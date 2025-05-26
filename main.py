import os
import time
import numpy as np
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation

from sklearn.metrics import f1_score

def passive_fairness_audit(rep_model, fairness_model, img_paths, tabular, targets, batch_size=32):
    # Preprocess images and compute embeddings
    images = np.array([load_and_preprocess_image(p, (224, 224)) for p in img_paths])
    embeddings = rep_model.predict([images, tabular], batch_size=batch_size)
    preds = np.argmax(fairness_model.predict(embeddings, batch_size=batch_size), axis=1)
    acc = np.mean(preds == targets)
    f1 = f1_score(targets, preds, average='weighted')
    return {"acc": acc, "f1": f1}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 32
EPOCHS = 1
ROUNDS = 2
PERCENTAGE = 1
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
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)} | Percentage: {int(PERCENTAGE*100)}%")
    print(f"Rounds: {ROUNDS} | Epochs per round: {EPOCHS} | Batch size: {BATCH_SIZE}\n")

    results_summary = {}
    start = time.time()
    modes = {"FairVFL": True, "VanillaFL": False}

    # To track FairVFL fairness models for passive audit
    fair_gender_model = None
    fair_age_model = None
    fair_rep_model = None

    # For storing both rep_models for probing
    vanilla_rep_model = None

    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

        model, rep_model, gender_model, age_model, _, _, _, _ = get_model_variant(
            (train_imgs, train_feats, train_labels, train_attrs),
            LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness
        )

        for round_idx in range(ROUNDS):
            print(f"\n--- FL Round {round_idx + 1}/{ROUNDS} ---")
            model, rep_model, gender_model, age_model, _, _, _, _ = FairVFL_train(
                (train_imgs, train_feats, train_labels, train_attrs),
                (val_imgs, val_feats, val_labels, val_attrs),
                (test_imgs, test_feats, test_labels, test_attrs),
                train_labels, val_labels, test_labels,
                lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE,
                epochs=EPOCHS, batch_size=BATCH_SIZE, mode=mode_label,
                model=model, rep_model=rep_model, gender_model=gender_model, age_model=age_model
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

        # If this is FairVFL, keep the fairness models/rep_model for passive audit
        if with_fairness:
            fair_gender_model = gender_model
            fair_age_model = age_model
            fair_rep_model = rep_model

        # If VanillaFL, keep the rep_model for probing
        if not with_fairness:
            vanilla_rep_model = rep_model

    # After both loops, probe both representations with FairVFL fairness heads
    print("\n--- Passive Fairness Audit: Probing both embeddings with FairVFL fairness heads ---")
    if fair_gender_model is not None and fair_age_model is not None:
        # Probing FairVFL
        fair_gender_audit = passive_fairness_audit(
            fair_rep_model, fair_gender_model, test_imgs, test_feats, test_attrs[:, 0], batch_size=BATCH_SIZE
        )
        fair_age_audit = passive_fairness_audit(
            fair_rep_model, fair_age_model, test_imgs, test_feats, test_attrs[:, 1], batch_size=BATCH_SIZE
        )
        print(f"FairVFL: Gender leakage: acc={fair_gender_audit['acc']:.3f}, f1={fair_gender_audit['f1']:.3f}")
        print(f"FairVFL: Age leakage: acc={fair_age_audit['acc']:.3f}, f1={fair_age_audit['f1']:.3f}")

    if vanilla_rep_model is not None and fair_gender_model is not None and fair_age_model is not None:
        vanilla_gender_audit = passive_fairness_audit(
            vanilla_rep_model, fair_gender_model, test_imgs, test_feats, test_attrs[:, 0], batch_size=BATCH_SIZE
        )
        vanilla_age_audit = passive_fairness_audit(
            vanilla_rep_model, fair_age_model, test_imgs, test_feats, test_attrs[:, 1], batch_size=BATCH_SIZE
        )
        print(f"VanillaFL: Gender leakage: acc={vanilla_gender_audit['acc']:.3f}, f1={vanilla_gender_audit['f1']:.3f}")
        print(f"VanillaFL: Age leakage: acc={vanilla_age_audit['acc']:.3f}, f1={vanilla_age_audit['f1']:.3f}")

    np.save("FairVFL_results.npy", results_summary)
    print(f"\nTotal training time: {round(time.time() - start, 2)} seconds")
