import os
import time
import numpy as np
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 32
EPOCHS = 1
ROUNDS = 2
PERCENTAGE = 0.1
LR = 0.001
HYPER_GENDER = 0.005
HYPER_AGE = 0.001

if __name__ == "__main__":
    print("Loading HAM10000 dataset...")
    data = load_ham10000(DATA_DIR)
    (train_imgs, train_feats, train_labels, train_attrs) = data['train']
    (test_imgs, test_feats, test_labels, test_attrs) = data['test']

    # Use only a percentage of data if PERCENTAGE < 1.0
    if PERCENTAGE < 1.0:
        n_train = int(len(train_imgs) * PERCENTAGE)
        n_val = int(len(train_imgs) * 0.2 * PERCENTAGE)
        n_test = int(len(test_imgs) * PERCENTAGE)

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
    else:
        # Usual split: 20% of train for validation
        val_split = int(0.2 * len(train_imgs))
        val_imgs, val_feats, val_labels, val_attrs = (
            train_imgs[:val_split], train_feats[:val_split], train_labels[:val_split], train_attrs[:val_split]
        )
        train_imgs, train_feats, train_labels, train_attrs = (
            train_imgs[val_split:], train_feats[val_split:], train_labels[val_split:], train_attrs[val_split:]
        )

    print("\nDataset sizes:")
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)} | Percentage: {PERCENTAGE*100:.0f}%")
    print(f"Rounds: {ROUNDS} | Epochs per round: {EPOCHS} | Batch size: {BATCH_SIZE}\n")

    results_summary = {}
    start = time.time()
    modes = {"FairVFL": True, "VanillaFL": False}

    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

        # Initialize ALL models at first round
        model, rep_model, gender_model, age_model, _, _, gender_cons_adv, age_cons_adv = get_model_variant(
            (train_imgs, train_feats, train_labels, train_attrs),
            LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness
        )

        for round_idx in range(ROUNDS):
            print(f"\n--- FL Round {round_idx + 1}/{ROUNDS} ---")
            model, rep_model, gender_model, age_model, _, _, gender_cons_adv, age_cons_adv = FairVFL_train(
                (train_imgs, train_feats, train_labels, train_attrs),
                (val_imgs, val_feats, val_labels, val_attrs),
                (test_imgs, test_feats, test_labels, test_attrs),
                train_labels, val_labels, test_labels,
                lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE,
                epochs=EPOCHS, batch_size=BATCH_SIZE, mode=mode_label,
                model=model, rep_model=rep_model, gender_model=gender_model, age_model=age_model,
                gender_cons_adv=gender_cons_adv, age_cons_adv=age_cons_adv
            )

        print("\nEvaluating after final round...")
        test_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in test_imgs])
        val_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in val_imgs])
        test_input = (test_images, test_feats, test_labels, test_attrs[:, 0], test_attrs[:, 1])
        val_input = (val_images, val_feats, val_labels, val_attrs[:, 0], val_attrs[:, 1])
        test_result = evaluation(model, test_input)
        val_result = evaluation(model, val_input)

        print(f"Test result: {test_result}")
        print(f"Validation result: {val_result}")
        results_summary[mode_label] = {"test": test_result, "val": val_result}

    np.save("FairVFL_results.npy", results_summary)
    print(f"\nTotal training time: {round(time.time() - start, 2)} seconds")
