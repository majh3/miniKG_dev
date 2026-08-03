# MiniKG

## Setup

```bash
pip install -r requirements.txt
```

CUDA and a working C++/CUDA compiler are required.

## Data

Place the datasets at:

```text
data/family/all_id.txt
data/yago3-10/all_id.txt
```

Each line must contain integer IDs in this order:

```text
head relation tail
```

## Run

```bash
python run.py --dataset family
python run.py --dataset yago3-10
```

Both datasets use the shared settings in `config.json`. Results are written to `runs/<dataset>/`.
