"""Preadapter modules used by CHORD.

Configuration agreed for this project:

    bottleneck dimension      48 (768 / 16)
    structure                 Linear(768, 48) -> GELU -> Linear(48, 768)
    adapters per layer        2 (attention + feed-forward)
    adapted layers            3 per encoder: 1-based layers 2, 11, 12
                              (0-based 1, 10, 11), identical for text and vision
    visual insertion point    ahead of MHSA and ahead of FFN
    textual insertion point   after the complete attention-output sub-layer and
                              after the complete feed-forward-output sub-layer

The adapter itself has no residual, no LayerNorm and no dropout.  The caller
adds the adapter output to the sub-layer input/output, and the up-projection is
zero-initialised, so every adapter starts as an exact identity (x + 0).
"""

import torch.nn as nn

# 0-based indices.  These correspond to 1-based layers 2, 11 and 12.
ADAPTED_LAYERS = (1, 10, 11)
HIDDEN_SIZE = 768
BOTTLENECK = 48  # 768 // 16


class Preadapter(nn.Module):
    """Down-project -> GELU -> up-project bottleneck adapter."""

    def __init__(self, hidden_size=HIDDEN_SIZE, bottleneck=BOTTLENECK):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        # Identity at initialisation: x + adapter(x) == x.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


def build_preadapters(hidden_size=HIDDEN_SIZE, bottleneck=BOTTLENECK,
                      layers=ADAPTED_LAYERS):
    """Create one adapter per adapted layer, keyed by the 0-based layer index."""
    return nn.ModuleDict({str(i): Preadapter(hidden_size, bottleneck)
                          for i in layers})


def _install_post_adapter(module, adapter):
    """Append ``adapter`` to the complete output of ``module`` (residual add)."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + adapter(output[0]),) + output[1:]
        return output + adapter(output)

    module.register_forward_hook(hook)


def attach_text_preadapters(bert_encoder, attn_preadapters, ffn_preadapters,
                            layers=ADAPTED_LAYERS):
    """Attach post-sub-layer adapters to the selected ``BertLayer``s.

    In a stock transformers ``BertLayer``:

        attention.output = Dense -> Dropout -> LayerNorm(+residual)
        output           = Dense -> Dropout -> LayerNorm(+residual)

    Hooking those modules keeps the original computation order untouched and
    avoids copying transformers internals, which makes the code stable across
    transformers releases and keeps every pre-existing state-dict key intact.
    """
    for i in layers:
        layer = bert_encoder.encoder.layer[i]
        _install_post_adapter(layer.attention.output, attn_preadapters[str(i)])
        _install_post_adapter(layer.output, ffn_preadapters[str(i)])
