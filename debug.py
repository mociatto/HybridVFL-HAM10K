from data import load_data, data_partition, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation, fairness_evaluation
import numpy as np
import os
import time

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

def preprocess_images(paths, target_size=(218, 178)):
    return np.array([load_and_preprocess_image(p, target_size) for p in paths])

if __name__ == "__main__":
    print("Loading data...")
    image_files, attributes, img_dir = load_data(DATA_ROOT)

    print("Partitioning data...")
    train_data, train_label, train_attr, val_data, val_label, val_attr, test_data, test_label, test_attr = \
        data_partition(image_files, attributes, img_dir, DATA_ROOT)

    # Define reduced sample sizes
    n_train = int(len(train_data[0]) * 0.01)
    n_val = int(len(val_data[0]) * 0.01)
    n_test = int(len(test_data[0]) * 0.01)

    # Slice data
    train_data = (train_data[0][:n_train], train_data[1][:n_train])
    val_data = (val_data[0][:n_val], val_data[1][:n_val])
    test_data = (test_data[0][:n_test], test_data[1][:n_test])
    train_label = train_label[:n_train]
    val_label = val_label[:n_val]
    test_label = test_label[:n_test]
    train_attr = (train_attr[0][:n_train], train_attr[1][:n_train])
    val_attr = (val_attr[0][:n_val], val_attr[1][:n_val])
    test_attr = (test_attr[0][:n_test], test_attr[1][:n_test])

    print(f"\n Using a reduced subset for quick debug: {n_train} train, {n_val} val, {n_test} test")
    print(f"\n Subset sizes — Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

    lr, hyper_gender, hyper_age = 0.001, 0.005, 0.001
    results_summary = {}
    start = time.time()
    modes = {'FairVFL': True, 'VanillaFL': False}

    for label, fairness_flag in modes.items():
        print(f"\n======= Training Mode: {label} (with_fairness={fairness_flag}) =======")

        # Inspect fairness attributes
        print("Gender attribute shape:", train_attr[0].shape, " Unique:", np.unique(train_attr[0], return_counts=True))
        print("Age attribute shape:", train_attr[1].shape, " Unique:", np.unique(train_attr[1], return_counts=True))

        print("Training...")
        model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv = FairVFL_train(
            train_data + (train_attr,), val_data + (val_attr,), test_data + (test_attr,),
            train_label, val_label, test_label,
            mode=label, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age, epochs=1
        )

        print("Evaluating...")
        test_images = preprocess_images(test_data[0])
        val_images = preprocess_images(val_data[0])
        test_input = (test_images, test_data[1], test_label, test_attr[0], test_attr[1])
        val_input = (val_images, val_data[1], val_label, val_attr[0], val_attr[1])

        test_result = evaluation(model, test_input)
        val_result = evaluation(model, val_input)

        if fairness_flag:
            gender_test = fairness_evaluation(rep_model, gender_model, test_input)
            gender_val = fairness_evaluation(rep_model, gender_model, val_input)
            age_test = fairness_evaluation(rep_model, age_model, test_input)
            age_val = fairness_evaluation(rep_model, age_model, val_input)
        else:
            gender_test = gender_val = age_test = age_val = {"f1": None}

        print(f"Test result: {test_result}")
        print(f"Validation result: {val_result}")
        print(f"Gender Test: {gender_test}, Gender Val: {gender_val}")
        print(f"Age Test: {age_test}, Age Val: {age_val}")

        results_summary[label] = {
            "test": {**test_result, "gender_f1": gender_test["f1"], "age_f1": age_test["f1"]},
            "val":  {**val_result,  "gender_f1": gender_val["f1"],  "age_f1": age_val["f1"]}
        }

    np.save('FairVFL_debug_results.npy', results_summary)
    print(f"\n Total training time: {round(time.time() - start, 2)} seconds")