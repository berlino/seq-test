## Setup

Use virtual environment `seq-test`. 

```bash
pip install torch==2.2.0 opt_einsum pythomata
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
python setup.py install
```


## Running

```bash
python -m zoology.launch zoology/experiments/mqar/mha.py
```

Known issues:

* forward of GLA needs to be adapted to output only the output vector
* RMSNorm in GLA is parameteric, which might have issue in the multi-head cases (multiple group share the same scale parameter does not make much sense)

## Sync with Upstream

```bash
git checkout -b HazyResearch-main main
git pull https://github.com/HazyResearch/zoology.git main
git checkout main
git merge --no-ff HazyResearch-main
git push origin main
```
