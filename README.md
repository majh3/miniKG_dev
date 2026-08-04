# MiniKG

Minimal code for running MiniKG on Family, YAGO3-10, CoDEx-S/M/L, Wikidata5M, and Freebase.

## Setup

```bash
pip install -r requirements.txt
```

CUDA and a working C++/CUDA compiler are required.

## Data

Family, YAGO3-10, and CoDEx-S/M/L are included as named triples:

```text
data/<dataset>/triples.tsv
```

Each dataset is one file containing only tab-separated `head name`, `relation name`, and `tail name` triples. Integer IDs are created only in memory.

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
python run.py --dataset codex-s
python run.py --dataset codex-m
python run.py --dataset codex-l
python run.py --dataset wikidata5m
python run.py --dataset freebase
```

Training steps are set per dataset in the shared `config.json`. Results are written to `runs/<dataset>/`.
Large-graph training automatically chooses the shortest frequency-ranked relation prefix covering 70% of Wikidata5M or 80% of Freebase facts. Relation profiling and first-hop decode pruning then use fixed defaults. Small graphs use lightweight compression probes and run the full decode once. Large graphs decode once for profiling and once after focused continuation.

## Query

Training also writes a directly queryable representation to `runs/<dataset>/query/`.

```bash
python query.py --dataset family --head 0 --relations 1
python query.py --dataset family --head 0 --relations 1,2
```

The second command evaluates a two-hop relation path. Answers are returned as entity IDs in JSON.
