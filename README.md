# MyVFL: Fairness-Aware Vertical Federated Learning for Skin Cancer Detection

A modular Python implementation of **FairVFL** adapted for the HAM10000 skin lesion dataset.  
This project provides a research-ready codebase for exploring fairness and privacy in vertical federated learning using both image and tabular medical data.

---

## Project Structure

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

## Setup

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

## Run

To start training and evaluating the FairVFL model, run from the project root:

```bash
python main.py
```
Note: For faster testing, you can reduce the percentage of data used by setting the `PERCENTAGE` variable in `main.py` to a lower value (e.g., `PERCENTAGE = 0.1` for 10% of the data). This will significantly speed up training and evaluation, making it ideal for quick experiments or debugging.**

```python
PERCENTAGE = 0.1  # Use only 10% of data for fast testing
