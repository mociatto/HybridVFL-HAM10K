# MyVFL: Fairness-Aware Vertical Federated Learning for skin cancer detection

A modular Python implementation of **FairVFL** adapted for the HAM10000 skin lesion dataset.  
This project provides a research-ready codebase for exploring fairness and privacy in vertical federated learning (VFL) using both image and tabular medical data.

---

## 📂 Project Structure

- `main.py` — Entry point for training and evaluation  
- `debug.py` — Quick debug mode using a reduced subset for fast checks  
- `data.py` — Data loading and preprocessing (HAM10000 images + metadata)  
- `model.py` — Model architectures (CNN, tabular encoder, fairness heads)  
- `train.py` — Training routines  
- `evaluate.py` — Evaluation and fairness audit routines  
- `requirements.txt` — Python dependencies  
- `.gitignore` — Standard ignores for Python and data  
- `/data/` — Place the HAM10000 dataset here

---

## 📥 Setup

1. **Clone this repository.**

2. **Download the HAM10000 dataset** from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
   Place the extracted folders and CSV in your `/data` directory as follows:

    ```
    data/
    ├── HAM10000_images_part_1/         # First batch of skin lesion images
    │   ├── ISIC_0024306.jpg
    │   ├── ...
    ├── HAM10000_images_part_2/         # Second batch of images
    │   ├── ISIC_0032012.jpg
    │   ├── ...
    ├── HAM10000_metadata.csv           # Metadata for each image (age, sex, dx, localization)
    ```


3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Run

To start training and evaluating the FairVFL model, run from the project root:

```bash
python main.py
```
Or alternatively for faster testing, try running the debug script:

```bash
python debug.py
```