# G-CNN: Group-Equivariant Convolutional Neural Networks

This project investigates the benefits of Group-Equivariant CNNs (G-CNNs) for colorectal tissue classification. Rather than relying on data augmentation to handle rotational variation, G-CNNs encode rotational and reflective symmetry directly into the network architecture using the *p4m* symmetry group. The goal is to study whether this structural prior improves classification accuracy and, in particular, sample efficiency compared to a standard CNN with and without augmentation.

The full report is available at [`report/main.pdf`](report/main.pdf).

---

## Results

Experiments are run on the **NCT-CRC-HE-100K** colorectal tissue dataset (9 classes) across training-set fractions ranging from 5% to 100%.

| Model | Acc. @ 5% | F1 @ 5% | Acc. @ 25% | F1 @ 25% | Acc. @ 100% | F1 @ 100% |
|---|---|---|---|---|---|---|
| CNN | 0.56 | 0.51 | 0.67 | 0.64 | 0.83 | 0.77 |
| CNN + Augmentation | 0.75 | 0.66 | 0.76 | 0.72 | **0.89** | **0.84** |
| G-CNN (*p4m*) | **0.78** | **0.71** | **0.80** | **0.73** | 0.86 | 0.80 |

The G-CNN leads at low-data regimes, demonstrating superior sample efficiency. Notably, the G-CNN trained on just 25% of the data achieves accuracy and F1 comparable to the augmented CNN trained on the full dataset. At 100% of the data, the augmented CNN closes the gap and slightly outperforms the G-CNN overall.

---

## Setup

### Option 1: Docker (recommended)

A `Dockerfile` and `docker-compose.yaml` are provided. Docker handles all dependencies, including CUDA.

Build the image once:

```bash
docker compose build
```

Then use the compose services described in the sections below.

### Option 2: pip

Requires Python 3.10+ and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

Trains a model using the configuration in `config/`. The default model is the G-CNN with the *p4m* group on the CRC dataset.

```bash
# Direct
python3 -m src.train

# Docker
docker compose run train
```

The model and mode can be overridden via Hydra:

```bash
python3 -m src.train mode=base_cnn model=base_cnn
python3 -m src.train data.use_augmentation=true
python3 -m src.train data.data_fraction=0.1
```

Checkpoints and TensorBoard logs are written to `outputs/`.

### Evaluation

Evaluates a trained checkpoint and writes metrics to a `metrics/metrics.json` file next to the run directory.

```bash
# Direct
python3 -m src.evaluate eval.model_path=<path/to/checkpoint.pt>

# Docker
docker compose run eval
```

### Training Experiment

Runs the full sweep used in the paper: all combinations of model, augmentation flag, and training-data fraction (1%, 5%, 10%, 25%, 50%, 100%), saving each run under `outputs/final/`.

```bash
# Direct
python3 -m src.train_experiment

# Docker
docker compose run train-experiment
```

---

## Configuration

All configuration is managed with [Hydra](https://hydra.cc). Config files live in `config/`:

```
config/
  config.yaml          # top-level defaults
  model/
    gecnn.yaml         # G-CNN (p4m) architecture
    base_cnn.yaml      # standard CNN baseline
  train/
    default.yaml       # training hyperparameters
  eval/
    default.yaml       # evaluation settings
  data/
    crc.yaml           # NCT-CRC-HE-100K dataset
    celeba.yaml
    chess.yaml
```

Key parameters you may want to override:

| Parameter | Default | Description |
|---|---|---|
| `mode` | `gecnn` | `gecnn` or `base_cnn` |
| `seed` | `21` | Global random seed |
| `data.use_augmentation` | `false` | Enable rotation/reflection augmentation |
| `data.data_fraction` | `1.0` | Fraction of training data to use |
| `train.num_epochs` | `5` | Number of training epochs |
| `train.batch_size` | `64` | Batch size |
| `train.lr` | `0.0001` | Learning rate |

---

## Monitoring

TensorBoard and Optuna Dashboard services are also included in the compose file:

```bash
docker compose up tensorboard       # → http://localhost:6067
docker compose up optuna-dashboard  # → http://localhost:8067
```

---

## Project Structure

```
src/
  models/
    gecnn.py          # G-CNN model (p4m group convolutions)
    modules.py        # Group-equivariant layer implementations
    base_cnn.py       # Standard CNN baseline
  training/
    train_gecnn.py    # G-CNN training loop
    train_base_cnn.py # CNN training loop
    train_functions.py
  data/
    crc.py            # NCT-CRC-HE-100K dataset loader
    celeba.py
    chess.py
  train.py            # Entry point: single training run
  evaluate.py         # Entry point: evaluation
  train_experiment.py # Entry point: full paper experiment sweep
  plots.py            # Result plotting utilities
config/               # Hydra configuration
report/               # LaTeX source and compiled paper (main.pdf)
tests/                # Unit and invariance tests
```

---

## Tests

```bash
pytest tests/
```

The test suite covers group-equivariance invariance properties and model layer correctness.