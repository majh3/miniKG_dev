# MiniKG

Minimal code for running MiniKG on Family, YAGO3-10, Wikidata5M, and Freebase.

## Setup

```bash
pip install -r requirements.txt
```

CUDA and a working C++/CUDA compiler are required.

## Data

Family and YAGO3-10 are included:

```text
data/family/all_id.txt
data/yago3-10/all_id.txt
```

Download the large datasets from:

- Wikidata5M: https://github.com/THU-KEG/KEPLER#pre-training
- Freebase: https://aws-dglke.readthedocs.io/en/latest/train.html

Prepare them as:

```text
data/wikidata5m/all_id.npy
data/freebase/train.npy
```

Rows must contain integer IDs in this order:

```text
head relation tail
```

## Run

```bash
python run.py --dataset family
python run.py --dataset yago3-10
python run.py --dataset wikidata5m
python run.py --dataset freebase
python run.py --dataset family --steps 1000
```

`--steps` overrides the default base-training steps. All datasets use the shared settings in `config.json`. Results are written to `runs/<dataset>/`.
`profile_threshold` controls relation profiling; `decode_prune_threshold` controls first-hop decode pruning. Both techniques are enabled in code only for Wikidata5M and Freebase. Small graphs use lightweight compression probes and run the full decode once. Large graphs decode once for profiling and once after focused continuation.

## Query

Training also writes a directly queryable representation to `runs/<dataset>/query/`.

```bash
python query.py --dataset family --head 0 --relations 1
python query.py --dataset family --head 0 --relations 1,2
```

The second command evaluates a two-hop relation path. Answers are returned as entity IDs in JSON.
