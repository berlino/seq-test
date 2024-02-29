## Setup

```bash
pip install torch==2.2.0 opt_einsum pythomata
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
pip install -e .
```

## Running

```bash
python -m zoology.launch zoology/experiments/mqar/mha.py
```