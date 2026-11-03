# CHORD

This is a official repository for "Bridging Cross-Stage Cirrhosis and Hepatocellular Carcinoma in Different Patients for Improved Cancer Prognostic Prediction".


## Repository structure

```
.
├── train.py                 # training entry point
├── model.py                 # CHORD (formerly PMF4a) and its minimal dependencies
├── adapters.py              # Preadapter and BERT hook installation
├── dataset.py               # dataset construction (only the 4 symbols used by the entry point)
├── ViT_3d.py                # 3D ViT encoder
├── position_encoding.py     # learned 3D positional encoding
├── graph_model.py           # SeHGNN2 knowledge-graph model (minimal closure)
├── utils.py                 # Cox loss and c-index
├── requirements.txt
├── LICENSE                  # Apache License 2.0
├── NOTICE
├── scripts/smoke_test.py    # CPU smoke test; no data or pretrained weights required
├── data/                    # place datasets here (not distributed)
└── weights/                 # place pretrained weights here (not distributed)
```

## Environment

Validated with **Python 3.10**, **torch 2.0.1** and CUDA 11.7 (CPU is sufficient for the
smoke test).

### CPU (for the smoke test)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# dassl's setup.py imports numpy without declaring it as a build dependency.
pip install numpy==1.26.4 setuptools wheel
pip install --no-build-isolation -r requirements.txt
```

### CUDA 11.7

```bash
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install numpy==1.26.4 setuptools wheel
pip install --no-build-isolation -r requirements.txt
```

Notes:

- `torch-sparse` imports `torch_scatter` at runtime, so both must be installed from the
  PyG wheel index that matches the torch build.
- `dassl` is not published on PyPI. `requirements.txt` pins it to commit `c61a1b5` of
  `KaiyangZhou/Dassl.pytorch`. Because of the build-isolation issue described above,
  `numpy` must be installed first and `--no-build-isolation` must be used.

## Data preparation

All default paths are relative to the repository root and can be overridden from the
command line. 

## Pretrained weights

| Argument | Default | Description |
|---|---|---|
| `--knowledge_ckpt` | `weights/graph.pkl` | SeHGNN2 knowledge-graph weights |
| `--vit_ckpt` | `weights/vit_3d.bin` | 3D ViT weights |
| `--bert_model` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Hugging Face model ID |

The BERT backbone is downloaded from Hugging Face by default. Add
`--local_files_only` to use a local cache only.

## Training

```bash
python train.py \
    --data_root1 data/data_1/preprocess/tumor/data \
    --data_root2 data/data_2/preprocess/tumor/data \
    --knowledge_ckpt weights/graph.pkl \
    --vit_ckpt weights/vit_3d.bin \
    --device auto \
    --gpu 2
```


### Resuming from a checkpoint

The original experiment resumed from an external checkpoint. It can be supplied with
`--resume`:

```bash
python train.py --resume /path/to/checkpoint.pth ...
```

## Smoke test (no data or weights required)

```bash
python scripts/smoke_test.py --batch-size 2
```

## Platform
The testing platform will be made available in the near future; users are required to configure their own LLM and to provide the corresponding API key.


## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for
attribution information.
