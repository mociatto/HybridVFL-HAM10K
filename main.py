from data import load_data, data_partition
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation
import numpy as np
import os
import time

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

if __name__ == "__main__":
    print("Loading data...")
    image_files, attributes, img_dir = load_data(DATA_ROOT)
    print("Partitioning data...")
    train_data, train_label, train_attr, val_data, val_label, val_attr, test_data, test_label, test_attr = data_partition(image_files, attributes, img_dir, DATA_ROOT)

    print(f"\n Dataset sizes:")
    print(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

    lr, hyper_gender, hyper_age = 0.001, 0.005, 0.001
    results_summary = {}
    start = time.time()
    modes = {'FairVFL': True, 'VanillaFL': False}

    for label, fairness_flag in modes.items():
        print(f"\n======= Training Mode: {label} (with_fairness={fairness_flag}) =======")
        model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv = \
            get_model_variant(train_data, lr, hyper_gender, hyper_age, with_fairness=fairness_flag)
        print("Training...")
        FairVFL_train(train_data, val_data, test_data, train_label, val_label, test_label, mode=label, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age, epochs=5)
        print("Evaluating...")
        test_result = evaluation(model, rep_model, test_data, test_label, test_attr)
        val_result = evaluation(model, rep_model, val_data, val_label, val_attr)
        print(f"Test result: {test_result}")
        print(f"Validation result: {val_result}")
        results_summary[label] = {"test": test_result, "val": val_result}

    np.save('FairVFL_results.npy', results_summary)
    print(f"\n🕒 Total training time: {round(time.time() - start, 2)} seconds")