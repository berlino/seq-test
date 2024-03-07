## Setup

Use virtual environment `seq-test`. 

```bash
pip install torch==2.2.0 opt_einsum pythomata
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
pip install -e .
```


## Running

```bash
python -m zoology.launch zoology/experiments/mqar/mha.py
```

## Sync with Upstream

```bash
git checkout -b HazyResearch-main main
git pull https://github.com/HazyResearch/zoology.git main
git checkout main
git merge --no-ff HazyResearch-main
git push origin main
```