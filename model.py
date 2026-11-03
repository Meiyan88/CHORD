"""CHORD model.

This is the minimal PMF4a symbol closure, renamed to CHORD.  The only
structural change relative to the original PMF4a is the adapter configuration:

* visual encoder: Preadapter ahead of MHSA and ahead of FFN, on 0-based layers
  1, 10 and 11 (1-based 2, 11, 12), with an external residual add;
* textual encoder: Preadapter after ``attention.output`` and after ``output``
  on the same layers, installed through forward hooks on the stock
  transformers ``BertLayer``;
* only the 12 Preadapters are added to the trainable set.

Everything else (prompts, cross-modal alignment, knowledge fusion, heads and
the forward data flow) is unchanged.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from ViT_3d import ViT
from adapters import (ADAPTED_LAYERS, BOTTLENECK, attach_text_preadapters,
                      build_preadapters)
from graph_model import SeHGNN2

DEFAULT_BERT_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)


class MLP_adapter(nn.Module):
    """Non-linear translator between modalities (unchanged from PMF4a)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttentionx(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CHORD(nn.Module):
    """CHORD (formerly PMF4a)."""

    def __init__(self, args, knowledge_ckpt=None, vit_ckpt=None,
                 bert_model=DEFAULT_BERT_MODEL, local_files_only=False,
                 random_init=False):
        super().__init__()

        self.knowledge_model = SeHGNN2(
            'Prognosis', 768, 512, 3,
            ['P', 'PC', 'PD', 'PI', 'PT', 'PCC', 'PCD', 'PCT', 'PIC', 'PCTT',
             'PICTT'],
            [],
            'P',
            0.5, 0.1, 0.0,
            2, 3, 'leaky_relu', False,
            data_size={'P': 768, 'PC': 768, 'PCC': 768, 'PCD': 768,
                       'PCT': 768, 'PCTT': 768, 'PD': 768, 'PI': 768,
                       'PIC': 768, 'PICTT': 768, 'PT': 768})

        if not random_init:
            if knowledge_ckpt is None:
                raise ValueError(
                    "knowledge_ckpt is required unless random_init=True")
            self.knowledge_model.load_state_dict(
                torch.load(knowledge_ckpt, map_location='cpu',
                           weights_only=False))

        if random_init:
            # Shape-compatible stand-in for the BiomedBERT checkpoint, used by
            # the synthetic smoke test only (no Hugging Face download).
            bert_config = BertConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
                type_vocab_size=2,
                vocab_size=30522,
            )
            self.bert_encoder = BertModel(bert_config)
        else:
            self.bert_encoder = BertModel.from_pretrained(
                bert_model, local_files_only=local_files_only)

        self.bert_encoder.heads = nn.Linear(
            self.bert_encoder.config.hidden_size, args.n_classes)

        self.heads = nn.ModuleList()
        num_heads = 5
        # Different head shapes are kept exactly as in PMF4a.
        for i in range(num_heads):
            if i == 0:
                head = nn.Linear(
                    self.bert_encoder.config.hidden_size + 32, 2)
            elif i == 1:
                head = nn.Sequential(
                    nn.Linear(self.bert_encoder.config.hidden_size + 32, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 2)
                )
            elif i == 2:
                head = nn.Sequential(
                    nn.Linear(self.bert_encoder.config.hidden_size + 32, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)
                )
            elif i == 3:
                head = nn.Sequential(
                    nn.Linear(self.bert_encoder.config.hidden_size + 32, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
            else:
                head = nn.Sequential(
                    nn.Linear(self.bert_encoder.config.hidden_size + 32, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
            self.heads.append(head)

        self.vit_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=32,
            frame_patch_size=4,
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        if not random_init:
            if vit_ckpt is None:
                raise ValueError("vit_ckpt is required unless random_init=True")
            vit_state = torch.load(vit_ckpt, map_location='cpu',
                                   weights_only=False)
            self.vit_encoder.load_state_dict(vit_state, strict=False)

        self.vit_encoder.heads = nn.Linear(
            self.vit_encoder.dim * 4, args.n_classes)
        self.vit_encoder.cls_head1 = nn.Linear(self.vit_encoder.dim * 4, 2)

        self.args = self.check_args(args)

        self.align_text = nn.ParameterList(
            [nn.Parameter(torch.empty(
                1, args.n_ctx, self.bert_encoder.config.hidden_size)
                .normal_(std=0.02)) for _ in range(args.n_fusion_layers)])

        self.align_trans = nn.ModuleList(
            [MLP_adapter(self.bert_encoder.config.hidden_size,
                         args.mlp_hidden_sz, self.vit_encoder.dim)
             for _ in range(args.n_fusion_layers)])

        # ------------------------------------------------------------------
        # Adapter configuration: 3 layers x 2 adapters, text and vision.
        # ------------------------------------------------------------------
        self.vis_attn_preadapters = build_preadapters(
            self.vit_encoder.dim, BOTTLENECK, ADAPTED_LAYERS)
        self.vis_ffn_preadapters = build_preadapters(
            self.vit_encoder.dim, BOTTLENECK, ADAPTED_LAYERS)
        self.txt_attn_preadapters = build_preadapters(
            self.bert_encoder.config.hidden_size, BOTTLENECK, ADAPTED_LAYERS)
        self.txt_ffn_preadapters = build_preadapters(
            self.bert_encoder.config.hidden_size, BOTTLENECK, ADAPTED_LAYERS)
        attach_text_preadapters(
            self.bert_encoder,
            self.txt_attn_preadapters,
            self.txt_ffn_preadapters,
            ADAPTED_LAYERS,
        )

        # Initial prompts / context vectors (unchanged).
        self.ctx_vectors = torch.empty(self.args.n_ctx, 768,)
        nn.init.normal_(self.ctx_vectors, std=0.02)
        self.proj = nn.Linear(768, 768)
        self.ctx_vectors = nn.Parameter(self.ctx_vectors)
        self.ctx_mask = torch.ones([1, self.args.n_ctx])

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(self.args.n_ctx, 768))
             for _ in range(self.args.n_align_layers - 1)])

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(768, 768)
        self.compound_prompt_projections0 = _get_clones(
            single_layer, self.args.n_align_layers - 1)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.args.n_align_layers - 1)
        self.compound_prompt_projections_liver = _get_clones(
            single_layer, self.args.n_align_layers - 1)
        self.compound_prompt_projections_tumor = _get_clones(
            single_layer, self.args.n_align_layers - 1)

        self.learnable_txt_cls = nn.Parameter(torch.randn(1, 1, 768))
        self.cross_attention = CrossAttention(768, 768, 8, 64)
        self.cross_fusion = CrossAttention(768, 768, 8, 64)

        self.linear_graph = nn.Linear(512, 768)
        self.knowledge_attention = CrossAttentionx(768, 768, 8, 64)

        self.grad_control()

    @staticmethod
    def _pre(adapters, layer_id, x):
        """Residual pre/post adapter: x + Preadapter(x), identity if not adapted."""
        key = str(layer_id)
        if key in adapters:
            return x + adapters[key](x)
        return x

    def forward(self, image_view0, image_view1, image_view2, image_view3,
                txt_input_ids, txt_token_type_ids, txt_attn_mask, graph_feat,
                clinical_data):
        n = image_view1.shape[0]
        device = image_view1.device
        assert image_view1.shape[0] == txt_input_ids.shape[0]

        ctx_vectors = self.ctx_vectors

        # Pre-processing before the two encoders.
        txt_tokens = self.bert_encoder.embeddings(
            txt_input_ids, txt_token_type_ids)
        graph_feat = self.knowledge_model.forward_feat(graph_feat, {}, None)
        graph_feat = self.linear_graph(graph_feat)
        graph_feat = graph_feat.unsqueeze(1).expand(-1, 256, -1)
        txt_tokens = self.knowledge_attention(graph_feat, txt_tokens) + txt_tokens
        length_text_tokens = txt_tokens.shape[1]
        txt_tokens = torch.cat(
            [txt_tokens[:, :-1], ctx_vectors.expand(n, -1, -1),
             txt_tokens[:, -1:]], dim=1)
        vis_align_first = self.proj(ctx_vectors)

        img_tokens0, pos_embedding0 = self.vit_encoder.embeddings(image_view0)
        img_tokens1, pos_embedding1 = self.vit_encoder.embeddings(image_view1)
        img_tokens2, pos_embedding2 = self.vit_encoder.embeddings(image_view2)
        img_tokens3, pos_embedding3 = self.vit_encoder.embeddings(image_view3)
        length_img_tokens1 = img_tokens1.shape[1]

        img_tokens0 = torch.cat(
            [img_tokens0, vis_align_first.expand(n, -1, -1)], dim=1)
        img_tokens1 = torch.cat(
            [img_tokens1, vis_align_first.expand(n, -1, -1)], dim=1)
        img_tokens2 = torch.cat(
            [img_tokens2, vis_align_first.expand(n, -1, -1)], dim=1)
        img_tokens3 = torch.cat(
            [img_tokens3, vis_align_first.expand(n, -1, -1)], dim=1)
        length_img_tokens2 = img_tokens1.shape[1]

        # Extra text attention mask.
        txt_attn_mask = torch.cat(
            [txt_attn_mask, self.ctx_mask.expand(n, -1,).to(device)], dim=1)
        txt_attn_mask = self.get_extended_txt_attn_mask(txt_attn_mask)
        max_prompt_length = self.args.n_qp + self.args.n_qcp + self.args.n_fcp
        batch_extra_attn_mask = torch.ones(n, max_prompt_length).to(device)
        batch_extra_attn_mask = self.get_extended_txt_attn_mask(
            batch_extra_attn_mask)

        # Main forward: unimodal feature extraction.
        visual_deep_prompts0 = []
        visual_deep_prompts = []
        visual_deep_prompts_liver = []
        visual_deep_prompts_tumor = []
        for bert_layer_id in range(
                self.bert_encoder.config.num_hidden_layers
                - self.args.n_fusion_layers):
            if bert_layer_id == 0:
                txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](
                    txt_tokens, txt_attn_mask)[0]
            else:
                txt_tokens = torch.cat(
                    [txt_tokens[:, :length_text_tokens - 1],
                     self.compound_prompts_text[bert_layer_id - 1].expand(
                         n, -1, -1),
                     txt_tokens[:, -1:]], dim=1)
                txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](
                    txt_tokens, txt_attn_mask)[0]
                visual_deep_prompts0.append(
                    self.compound_prompt_projections0[bert_layer_id - 1](
                        self.compound_prompts_text[bert_layer_id - 1]))
                visual_deep_prompts.append(
                    self.compound_prompt_projections[bert_layer_id - 1](
                        self.compound_prompts_text[bert_layer_id - 1]))
                visual_deep_prompts_liver.append(
                    self.compound_prompt_projections_liver[bert_layer_id - 1](
                        self.compound_prompts_text[bert_layer_id - 1]))
                visual_deep_prompts_tumor.append(
                    self.compound_prompt_projections_tumor[bert_layer_id - 1](
                        self.compound_prompts_text[bert_layer_id - 1]))

        for vit_layer_id in range(
                self.vit_encoder.depth - self.args.n_fusion_layers):
            if vit_layer_id == 0:
                [attn, ff] = self.vit_encoder.transformer.layers[vit_layer_id]

                img_tokens0 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = attn(img_tokens0) + img_tokens0
                img_tokens0 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = ff(img_tokens0) + img_tokens0

                img_tokens1 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = attn(img_tokens1) + img_tokens1
                img_tokens1 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = ff(img_tokens1) + img_tokens1

                img_tokens2 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = attn(img_tokens2) + img_tokens2
                img_tokens2 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = ff(img_tokens2) + img_tokens2

                img_tokens3 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = attn(img_tokens3) + img_tokens3
                img_tokens3 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = ff(img_tokens3) + img_tokens3

            elif 0 < vit_layer_id < self.args.local_depth - 1:
                [attn, ff] = self.vit_encoder.transformer.layers[vit_layer_id]

                img_tokens0 = torch.cat(
                    [img_tokens0[:, :length_img_tokens1, :],
                     visual_deep_prompts0[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens0 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = attn(img_tokens0) + img_tokens0
                img_tokens0 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = ff(img_tokens0) + img_tokens0

                img_tokens1 = torch.cat(
                    [img_tokens1[:, :length_img_tokens1, :],
                     visual_deep_prompts[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens1 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = attn(img_tokens1) + img_tokens1
                img_tokens1 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = ff(img_tokens1) + img_tokens1

                img_tokens2 = torch.cat(
                    [img_tokens2[:, :length_img_tokens1, :],
                     visual_deep_prompts_liver[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens2 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = attn(img_tokens2) + img_tokens2
                img_tokens2 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = ff(img_tokens2) + img_tokens2

                img_tokens3 = torch.cat(
                    [img_tokens3[:, :length_img_tokens1, :],
                     visual_deep_prompts_tumor[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens3 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = attn(img_tokens3) + img_tokens3
                img_tokens3 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = ff(img_tokens3) + img_tokens3

            elif vit_layer_id == self.args.local_depth - 1:
                [attn, ff] = self.vit_encoder.transformer.layers[vit_layer_id]

                img_tokens0 = torch.cat(
                    [img_tokens0[:, :length_img_tokens1, :],
                     visual_deep_prompts0[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens0 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = attn(img_tokens0) + img_tokens0
                img_tokens0 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens0)
                img_tokens0 = ff(img_tokens0) + img_tokens0

                img_tokens1 = torch.cat(
                    [img_tokens1[:, :length_img_tokens1, :],
                     visual_deep_prompts[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens1 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = attn(img_tokens1) + img_tokens1
                img_tokens1 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens1)
                img_tokens1 = ff(img_tokens1) + img_tokens1

                img_tokens2 = torch.cat(
                    [img_tokens2[:, :length_img_tokens1, :],
                     visual_deep_prompts_liver[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens2 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = attn(img_tokens2) + img_tokens2
                img_tokens2 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens2)
                img_tokens2 = ff(img_tokens2) + img_tokens2

                img_tokens3 = torch.cat(
                    [img_tokens3[:, :length_img_tokens1, :],
                     visual_deep_prompts_tumor[vit_layer_id - 1].expand(
                         n, -1, -1)], dim=1)
                img_tokens3 = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = attn(img_tokens3) + img_tokens3
                img_tokens3 = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_tokens3)
                img_tokens3 = ff(img_tokens3) + img_tokens3

                cross_token0 = self.cross_attention(
                    img_tokens0, (img_tokens1 + img_tokens2 + img_tokens3) / 3)
                cross_token1 = self.cross_attention(
                    img_tokens1, (img_tokens2 + img_tokens0 + img_tokens3) / 3)
                cross_token2 = self.cross_attention(
                    img_tokens2, (img_tokens1 + img_tokens3 + img_tokens0) / 3)
                cross_token3 = self.cross_attention(
                    img_tokens3, (img_tokens1 + img_tokens2 + img_tokens0) / 3)

                # Kept as in the original model (doubling, not residual add).
                img_tokens0 = cross_token0 + cross_token0
                img_tokens1 = cross_token1 + cross_token1
                img_tokens2 = cross_token2 + cross_token2
                img_tokens3 = cross_token3 + cross_token3

                img_token_mv = torch.cat(
                    [img_tokens0, img_tokens1, img_tokens2, img_tokens3],
                    dim=1)

            else:
                [attn, ff] = self.vit_encoder.transformer.layers[vit_layer_id]
                img_token_mv = torch.cat(
                    [img_token_mv[:, :length_img_tokens1, :],
                     visual_deep_prompts0[vit_layer_id - 1].expand(n, -1, -1),
                     img_token_mv[:, length_img_tokens2:
                                  length_img_tokens2 + length_img_tokens1, :],
                     visual_deep_prompts[vit_layer_id - 1].expand(n, -1, -1),
                     img_token_mv[:, length_img_tokens2 * 2:
                                  length_img_tokens2 * 2
                                  + length_img_tokens1, :],
                     visual_deep_prompts_liver[vit_layer_id - 1].expand(
                         n, -1, -1),
                     img_token_mv[:, length_img_tokens2 * 3:
                                  length_img_tokens2 * 3
                                  + length_img_tokens1, :],
                     visual_deep_prompts_tumor[vit_layer_id - 1].expand(
                         n, -1, -1)
                     ], dim=1)
                img_token_mv = self._pre(
                    self.vis_attn_preadapters, vit_layer_id, img_token_mv)
                img_token_mv = attn(img_token_mv) + img_token_mv
                img_token_mv = self._pre(
                    self.vis_ffn_preadapters, vit_layer_id, img_token_mv)
                img_token_mv = ff(img_token_mv) + img_token_mv

        # Multimodal fusion layers.
        for fusion_layer_id in range(self.args.n_fusion_layers):
            cross_f1 = self.cross_fusion(txt_tokens, img_token_mv)
            cross_f2 = self.cross_fusion(img_token_mv, txt_tokens)

            txt_tokens = txt_tokens + cross_f1
            img_token_mv = img_token_mv + cross_f2

            img_token_mv = torch.cat(
                [img_tokens0, img_tokens1, img_tokens2, img_tokens3],
                dim=1) + img_token_mv

            batch_align_txt = self.align_text[fusion_layer_id].expand(
                n, -1, -1).to(device)

            txt_tokens = torch.cat(
                [txt_tokens[:, :length_text_tokens - 1], batch_align_txt,
                 txt_tokens[:, -1:]], dim=1)
            txt_tokens = self.bert_encoder.encoder.layer[
                bert_layer_id + fusion_layer_id + 1](
                    txt_tokens, txt_attn_mask)[0]
            batch_align_vis = self.align_trans[fusion_layer_id](
                batch_align_txt)

            img_token_mv = torch.cat(
                [img_token_mv[:, :length_img_tokens1, :],
                 batch_align_vis,
                 img_token_mv[:, length_img_tokens2:
                              length_img_tokens2 + length_img_tokens1, :],
                 batch_align_vis,
                 img_token_mv[:, length_img_tokens2 * 2:
                              length_img_tokens2 * 2 + length_img_tokens1, :],
                 batch_align_vis,
                 img_token_mv[:, length_img_tokens2 * 3:
                              length_img_tokens2 * 3 + length_img_tokens1, :],
                 batch_align_vis
                 ], dim=1)

            [attn, ff] = self.vit_encoder.transformer.layers[
                vit_layer_id + fusion_layer_id + 1]
            img_token_mv = self._pre(
                self.vis_attn_preadapters,
                vit_layer_id + fusion_layer_id + 1, img_token_mv)
            img_token_mv = attn(img_token_mv) + img_token_mv
            img_token_mv = self._pre(
                self.vis_ffn_preadapters,
                vit_layer_id + fusion_layer_id + 1, img_token_mv)
            img_token_mv = ff(img_token_mv) + img_token_mv

        # After the main forward.
        self.cls_txt_token = txt_tokens[:, 0]
        cls_txt_token = self.cls_txt_token
        img_token_mv = self.vit_encoder.layernorm(img_token_mv)
        img_token0_f, img_token1_f, img_token2_f, img_token3_f = torch.chunk(
            img_token_mv, 4, dim=1)
        cls_img_token = torch.cat(
            (img_token0_f[:, 0], img_token1_f[:, 0], img_token2_f[:, 0],
             img_token3_f[:, 0]), dim=1)

        txt_pred = self.bert_encoder.heads(cls_txt_token)
        txt_pred = F.sigmoid(txt_pred)
        img_pred = self.vit_encoder.heads(cls_img_token)
        img_pred = F.sigmoid(img_pred)

        txt_cls_pred1 = self.heads[0](
            torch.cat([cls_txt_token, clinical_data], dim=-1))
        txt_cls_pred1 = F.softmax(txt_cls_pred1, -1)
        txt_cls_pred2 = self.heads[1](
            torch.cat([cls_txt_token, clinical_data], dim=-1))
        txt_cls_pred2 = F.softmax(txt_cls_pred2, -1)
        txt_cls_pred3 = self.heads[2](
            torch.cat([cls_txt_token, clinical_data], dim=-1))
        txt_cls_pred3 = F.softmax(txt_cls_pred3, -1)
        txt_cls_pred4 = self.heads[3](
            torch.cat([cls_txt_token, clinical_data], dim=-1))
        txt_cls_pred4 = F.softmax(txt_cls_pred4, -1)
        txt_cls_pred5 = self.heads[4](
            torch.cat([cls_txt_token, clinical_data], dim=-1))
        txt_cls_pred5 = F.softmax(txt_cls_pred5, -1)

        img_pred2 = self.vit_encoder.cls_head1(cls_img_token)
        img_pred2 = F.sigmoid(img_pred2)

        return ((txt_pred + img_pred) / 2, txt_cls_pred1, txt_cls_pred2,
                txt_cls_pred3, txt_cls_pred4, txt_cls_pred5)

    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def check_args(self, args):
        assert args.n_qp > 0
        assert args.n_fusion_layers <= min(
            self.bert_encoder.config.num_hidden_layers, self.vit_encoder.dim)
        if args.mlp_hidden_sz == -1:
            args.mlp_hidden_sz = max(
                int(self.vit_encoder.dim / 2),
                int(self.bert_encoder.config.hidden_size / 2))
        return args

    def grad_control(self):
        for param in self.parameters():
            param.requires_grad = False

        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = True
        for param in self.vit_encoder.cls_head1.parameters():
            param.requires_grad = True

        # Adapter-tuning: only the newly configured Preadapters are trainable
        # in addition to the heads above.
        for adapters in (self.vis_attn_preadapters,
                         self.vis_ffn_preadapters,
                         self.txt_attn_preadapters,
                         self.txt_ffn_preadapters):
            for param in adapters.parameters():
                param.requires_grad = True
