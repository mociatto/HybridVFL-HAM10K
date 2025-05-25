import os
import time
import numpy as np
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation, fairness_evaluation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 16
TARGET_SIZE = (224, 224)

def preprocess_images(paths):
    return np.array([load_and_preprocess_image(p, TARGET_SIZE) for p in paths])

if __name__ == "__main__":
    print("Loading HAM10000 dataset...")
    data = load_ham10000(DATA_DIR)
    (train_imgs, train_feats, train_labels, train_attrs) = data['train']
    (test_imgs, test_feats, test_labels, test_attrs) = data['test']

    n_train = max(1, int(len(train_imgs) * 0.01))
    n_val = max(1, int(n_train * 0.2))
    n_test = max(1, int(len(test_imgs) * 0.01))

    val_imgs, val_feats, val_labels, val_attrs = (
        train_imgs[:n_val], train_feats[:n_val], train_labels[:n_val], train_attrs[:n_val]
    )
    train_imgs, train_feats, train_labels, train_attrs = (
        train_imgs[n_val:n_val + n_train], train_feats[n_val:n_val + n_train],
        train_labels[n_val:n_val + n_train], train_attrs[n_val:n_val + n_train]
    )
    test_imgs, test_feats, test_labels, test_attrs = (
        test_imgs[:n_test], test_feats[:n_test], test_labels[:n_test], test_attrs[:n_test]
    )

    print(f"\nUsing reduced subset: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    results_summary = {}
    start = time.time()

    fair_models = {}

    for label, with_fairness in {"FairVFL": True, "VanillaFL": False}.items():
        print(f"\n======= Training Mode: {label} (with_fairness={with_fairness}) =======")

        model, rep_model, gender_model, age_model, _, _, _, _ = FairVFL_train(
            (train_imgs, train_feats, train_attrs),
            (val_imgs, val_feats, val_attrs),
            (test_imgs, test_feats, test_attrs),
            train_labels, val_labels, test_labels,
            mode=label, lr=0.001, hyper_gender=0.005, hyper_age=0.001, epochs=1, batch_size=BATCH_SIZE
        )

        test_images = preprocess_images(test_imgs)
        val_images = preprocess_images(val_imgs)

        test_input = (test_images, test_feats, test_labels, test_attrs[:, 0], test_attrs[:, 1])
        val_input = (val_images, val_feats, val_labels, val_attrs[:, 0], val_attrs[:, 1])

        print("Evaluating...")
        test_result = evaluation(model, test_input)
        val_result = evaluation(model, val_input)

        if with_fairness:
            gender_test = fairness_evaluation(rep_model, gender_model, test_input)
            gender_val = fairness_evaluation(rep_model, gender_model, val_input)
            age_test = fairness_evaluation(rep_model, age_model, test_input)
            age_val = fairness_evaluation(rep_model, age_model, val_input)

            # Save fairness models for reuse
            fair_models["gender_model"] = gender_model
            fair_models["age_model"] = age_model
        else:
            # Passive audit using FairVFL's fairness models
            gender_test = fairness_evaluation(rep_model, fair_models["gender_model"], test_input)
            gender_val = fairness_evaluation(rep_model, fair_models["gender_model"], val_input)
            age_test = fairness_evaluation(rep_model, fair_models["age_model"], test_input)
            age_val = fairness_evaluation(rep_model, fair_models["age_model"], val_input)

        print(f"Test Result: {test_result}")
        print(f"Validation Result: {val_result}")
        print(f"Gender Fairness - Test: {gender_test}, Val: {gender_val}")
        print(f"Age Fairness - Test: {age_test}, Val: {age_val}")

        results_summary[label] = {
            "test": {**test_result, "gender_f1": gender_test["f1"], "age_f1": age_test["f1"]},
            "val":  {**val_result,  "gender_f1": gender_val["f1"],  "age_f1": age_val["f1"]}
        }

    np.save("FairVFL_debug_results.npy", results_summary)
    print(f"\nTotal debug training time: {round(time.time() - start, 2)} seconds")
