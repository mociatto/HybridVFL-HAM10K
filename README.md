# FairVFL

A modular Python implementation of FairVFL for CelebA for local use and GitHub deployment.

## Project Structure

- `main.py` — Entry point for training and evaluation
- `data.py` — Data loading and preprocessing
- `model.py` — Model architectures
- `train.py` — Training routines
- `evaluate.py` — Evaluation routines
- `requirements.txt` — Python dependencies
- `.gitignore` — Standard ignores for Python and data
- `/data/` — Place the CelebA dataset here

## Setup

1. Clone this repository.
2. Download the CelebA dataset and place it in the `/data` folder, preserving all subfolders and CSVs as follow:
```
data/
├── img_align_celeba/                    # Directory with all CelebA face images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── list_attr_celeba.csv                 # Attribute labels (gender, age, etc.)
├── list_bbox_celeba.csv                 # Bounding box coordinates
└── list_landmarks_align_celeba.csv      # Facial landmark positions
```
3. Install dependencies:

```bash
pip3 install -r requirements.txt
```