"""Synthetic CPU smoke test for CHORD.

This test needs no dataset and no pretrained weights.  It instantiates CHORD
with ``random_init=True`` (randomly initialised BiomedBERT-shaped encoder,
random ViT and knowledge model), runs a forward pass on synthetic tensors,
checks the adapter placement and identity initialisation, and performs one
optimizer step.

Run from the project root or scripts directory:

    python scripts/smoke_test.py --batch-size 2
"""

import argparse
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from adapters import ADAPTED_LAYERS, BOTTLENECK, Preadapter
from model import CHORD

GRAPH_KEYS = ['P', 'PC', 'PD', 'PI', 'PT', 'PCC', 'PCD', 'PCT', 'PIC',
              'PCTT', 'PICTT']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(1998)

    model_args = SimpleNamespace(
        n_classes=1,
        local_depth=2,
        n_fusion_layers=3,
        n_align_layers=9,
        n_ctx=3,
        n_qp=4,
        n_qcp=4,
        n_fcp=4,
        mlp_hidden_sz=-1,
        batch_size=args.batch_size,
    )

    model = CHORD(model_args, random_init=True).to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 1. Adapter structure checks
    # ------------------------------------------------------------------
    assert tuple(ADAPTED_LAYERS) == (1, 10, 11)
    for name in ("vis_attn_preadapters", "vis_ffn_preadapters",
                 "txt_attn_preadapters", "txt_ffn_preadapters"):
        adapters = getattr(model, name)
        assert sorted(adapters.keys()) == ["1", "10", "11"], name
        for adapter in adapters.values():
            assert isinstance(adapter, Preadapter)
            assert adapter.down.out_features == BOTTLENECK
            assert adapter.up.out_features == 768

    # Identity at initialisation: x + Preadapter(x) == x.
    probe = Preadapter().to(device)
    x = torch.randn(2, 5, 768, device=device)
    assert torch.allclose(x + probe(x), x, atol=1e-6), \
        "Preadapter must be an identity at initialisation"

    # BERT forward hooks must exist on exactly the adapted layers.
    for i in range(model.bert_encoder.config.num_hidden_layers):
        layer = model.bert_encoder.encoder.layer[i]
        expected = 1 if i in ADAPTED_LAYERS else 0
        assert len(layer.attention.output._forward_hooks) == expected
        assert len(layer.output._forward_hooks) == expected

    # ------------------------------------------------------------------
    # 2. Trainable parameter surface
    # ------------------------------------------------------------------
    trainable_names = [n for n, p in model.named_parameters()
                       if p.requires_grad]
    assert all("preadapters" in n or "heads" in n or "cls_head1" in n
               for n in trainable_names), \
        f"Unexpected trainable parameter: {trainable_names[:5]}"
    adapter_params = sum(p.numel() for n, p in model.named_parameters()
                         if "preadapters" in n)
    assert adapter_params == 12 * (768 * BOTTLENECK + BOTTLENECK
                                   + BOTTLENECK * 768 + 768), adapter_params
    print(f"Adapter parameters: {adapter_params:,} "
          f"({12} modules, bottleneck={BOTTLENECK})")
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # 3. Synthetic forward / backward / optimizer step
    # ------------------------------------------------------------------
    B = args.batch_size
    S = args.image_size
    T = args.frames
    L = args.text_length

    views = [torch.randn(B, 3, S, S, T, device=device) for _ in range(4)]
    input_ids = torch.randint(0, 30522, (B, L), device=device)
    token_type_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    graph_feat = {k: torch.randn(B, 768, device=device) for k in GRAPH_KEYS}
    clinical = torch.randn(B, 32, device=device)

    outputs = model(*views, input_ids, token_type_ids, attention_mask,
                    graph_feat, clinical)
    assert len(outputs) == 6
    assert outputs[0].shape == (B, 1)
    for out in outputs[1:]:
        assert out.shape == (B, 2)

    loss = outputs[0].mean()
    for out in outputs[1:]:
        loss = loss + out.mean()
    loss.backward()

    nonzero_up = 0
    for name in ("vis_attn_preadapters", "vis_ffn_preadapters",
                 "txt_attn_preadapters", "txt_ffn_preadapters"):
        adapters = getattr(model, name)
        for adapter in adapters.values():
            grad = adapter.up.weight.grad
            assert grad is not None, f"{name}: adapter received no gradient"
            if grad.abs().sum().item() > 0:
                nonzero_up += 1
    assert nonzero_up > 0, "No adapter received a non-zero gradient"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    print(f"Forward/backward/optimizer step OK: loss={loss.item():.6f}, "
          f"adapters_with_nonzero_grad={nonzero_up}/12")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
