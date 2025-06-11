# HybridVFL-HAM10K: Fairness-Aware Vertical Federated Learning for Skin Cancer Detection

A modular Python implementation of **FairVFL** adapted for the HAM10000 skin lesion dataset.  
This project provides a research-ready codebase for exploring fairness and privacy in vertical federated learning using both image and tabular medical data and demonstrates metrics and plots on a modern front-end dashboard.

---

## Project Structure

- `main.py` — Entry point for training and evaluation  
- `data.py` — Data loading and preprocessing (HAM10000 images + metadata)  
- `model.py` — Model architectures (CNN, tabular encoder, fairness heads)  
- `train.py` — Training routines  
- `evaluate.py` — Evaluation and fairness audit routines  
- `dashboard.py` — Flask/SocketIO backend for dashboard live metrics  
- `templates/dashboard.html` — Dashboard front-end template  
- `statics/dashboard.css` — Dashboard CSS styles  
- `statics/dashboard.js` — Dashboard interactive JS  
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
    ├── HAM10000_images_part_1/
    │   ├── ISIC_0024306.jpg
    │   ├── ...
    ├── HAM10000_images_part_2/
    │   ├── ISIC_0032012.jpg
    │   ├── ...
    ├── HAM10000_metadata.csv
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Run

### **Run the Project with Dashboard**

To launch the live dashboard and start FairVFL training with real-time metrics, run:

```bash
python dashboard.py
```

It opens http://localhost:5050 in your browser to view the dashboard.
Dashboard front-end is served via Flask/SocketIO and will automatically update with training progress.

### **Run CLI only**

If you only want to run the core training and evaluation via CLI, use:

```bash
python main.py
```
Note: For faster testing, you can reduce the percentage of data used by setting the `PERCENTAGE` variable in `main.py` to a lower value (e.g., `PERCENTAGE = 0.1` for 10% of the data). This will significantly speed up training and evaluation, making it ideal for quick experiments or debugging.**

```python
PERCENTAGE = 0.1  # Use only 10% of data for fast testing
