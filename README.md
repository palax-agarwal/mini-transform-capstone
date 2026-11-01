# mini-transform-capstone

## Mini Transformer (Encoder–Decoder) in NumPy

A complete encoder–decoder Transformer implemented from scratch with NumPy. Every layer
has a hand-derived `backward()` — there is no autograd and no deep-learning framework.
The toy task translates English number words into digit strings (`two thousand six` →
`2006`) for all numbers from 0 to 9999.

## What is in this project

| File | Purpose |
| --- | --- |
| `transformer.py` | Model library: embeddings, positional encoding, multi-head attention, encoder/decoder layers, Adam, greedy decoding. |
| `train.py` | **Main entry point.** Builds the dataset, trains the model, evaluates it, and saves results/weights. |
| `tests.py` | Self-contained verification suite (shapes, causal masking, padding invariance, batching, overfitting, reproducibility). |
| `gradcheck.py` | Finite-difference gradient check for every hand-written backward pass. |
| `analysis.py` | Aggregates the experiment JSON files, writes `summary.json`, and regenerates `fig/curves.png` and `fig/attn.png`. |
| `diagrams.py` | Regenerates the architecture diagram `fig/arch.png`. |
| `fig/` | Architecture, attention, curve, class and sequence diagrams. |
| `logs/` | Training logs for each experiment. |
| `results_*.json` | Evaluation results (loss, exact match, token accuracy, history) for each run. |
| `summary.json` | Aggregated results across configurations and seeds. |
| `model_*.npz` | Saved model parameters used by `analysis.py` for attention visualisation. |

## Requirements

- Python 3.10 or newer (verified with Python 3.13.2 on Windows)
- NumPy
- Matplotlib (only for `analysis.py` and `diagrams.py`)

Dependencies are listed in `requirements.txt`:

```text
numpy>=2.0
matplotlib>=3.8
```

## Run it in VS Code (PowerShell, from start to finish)

Open the project folder in VS Code: **File → Open Folder… →
`C:\Users\palax\Downloads\mini_transformer_code`**.

Then open **Terminal → New Terminal** (it should be PowerShell) and run:

```powershell
# 1. Go to the project folder
cd C:\Users\palax\Downloads\mini_transformer_code

# 2. Create the virtual environment (skip if .venv already exists)
python -m venv .venv

# 3. Activate it. If PowerShell blocks the script, run the Set-ExecutionPolicy line once.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 5. Verify the implementation (about 1–2 minutes; must print 10/10 tests passed)
python tests.py

# 6. Verify the hand-derived gradients (must print PASS)
python gradcheck.py

# 7. Train and evaluate the base model (a few minutes to ~20 minutes depending on
#    CPU; writes results_main_s0.json and model_main_s0.npz). Omit steps=3000 for
#    the 6000-step default.
python train.py main_s0 steps=3000

# 8. Regenerate the aggregate summary and figures
python analysis.py
python diagrams.py
```

To make VS Code use the environment, press `Ctrl+Shift+P` → **Python: Select
Interpreter** → choose `.\.venv\Scripts\python.exe`. The included
`.vscode/launch.json` (local, not committed) adds **Run and Debug** configurations
for the tests and a quick training run; otherwise use the integrated terminal
commands above.

### Run the main script directly

`train.py` is the training entry point and takes a run name followed by optional
`key=value` overrides:

```powershell
python train.py <run-name> [key=value ...]
```

Examples:

```powershell
# Base model used in the report
python train.py main_s0 steps=3000

# Ablations (one seed shown; the report uses seeds 0, 1 and 2)
python train.py head1_s0 h=1
python train.py layer1_s0 n_enc=1 n_dec=1
python train.py nope_s0 use_pe=False

# Fast smoke test
python train.py smoke steps=10 log_every=10
```

Training writes `results_<run-name>.json` and `model_<run-name>.npz` into the
project folder. The JSON contains the test loss, exact-match and token accuracy,
the training history, configuration and runtime.

## Results

The base model reaches **99.9% exact-match accuracy** on the held-out 1,000
numbers. `summary.json` and `fig/curves.png` compare the base model with three
ablations (one attention head, one encoder + one decoder layer, and no positional
encoding). Removing positional encoding causes the largest drop, confirming that
order information is essential for this task.

## Notes

- Arrays are batch-first: `(batch, sequence, features)`.
- Training is deterministic for a fixed seed (`tests.py`, test T8).
- Padding is masked in both the encoder self-attention and decoder cross-attention.
- The decoder uses a causal mask, so it cannot look ahead at future target tokens.
