# FairVFL

A modular Python implementation of FairVFL for CelebA, adapted from a Kaggle notebook for local use and GitHub deployment.

## Project Structure

- `main.py` — Entry point for training and evaluation
- `data.py` — Data loading and preprocessing
- `model.py` — Model architectures
- `train.py` — Training routines
- `evaluate.py` — Evaluation routines
- `requirements.txt` — Python dependencies
- `.gitignore` — Standard ignores for Python and data
- `/data/` — Place the CelebA dataset here (same structure as Kaggle)

## Setup

1. Clone this repository.
2. Download the CelebA dataset and place it in the `/data` folder, preserving all subfolders and CSVs.
3. Install dependencies:

```bash
pip3 install -r requirements.txt
```