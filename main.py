import os
import time
import numpy as np
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation

from sklearn.metrics import f1_score

def passive_fairness_audit(rep_model, fairness_model, img_paths, tabular, targets, batch_size=32):
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
PERCENTAGE = 0.05
LR = 0.001
HYPER_GENDER = 0.005
HYPER_AGE = 0.001

if __name__ == "__main__":
    print("Loading HAM10000 dataset...")
    data = load_ham10000(DATA_DIR)
    image_client = data['image_client']
    vertical_client = data['vertical_client']

    (img_train, feat_train, label_train, sens_train) = image_client['train']
    (img_test, feat_test, label_test, sens_test) = image_client['test']
    (vert_train, vert_label_train) = vertical_client['train']
    (vert_test, vert_label_test) = vertical_client['test']

    # Reduce dataset for faster debug (percentage)
    if PERCENTAGE < 1.0:
        n_train = int(len(img_train) * PERCENTAGE)
        n_val = int(len(img_train) * 0.2 * PERCENTAGE)
        n_test = int(len(img_test) * PERCENTAGE)

        val_imgs, val_feats, val_labels, val_attrs = (
            img_train[:n_val], feat_train[:n_val], label_train[:n_val], sens_train[:n_val]
        )
        train_imgs, train_feats, train_labels, train_attrs = (
            img_train[n_val:n_val + n_train], feat_train[n_val:n_val + n_train],
            label_train[n_val:n_val + n_train], sens_train[n_val:n_val + n_train]
        )
        test_imgs, test_feats, test_labels, test_attrs = (
            img_test[:n_test], feat_test[:n_test], label_test[:n_test], sens_test[:n_test]
        )
        train_metadata, val_metadata, test_metadata = (
            vert_train[n_val:n_val + n_train], vert_train[:n_val], vert_test[:n_test]
        )
        train_metadata_labels, val_metadata_labels, test_metadata_labels = (
            vert_label_train[n_val:n_val + n_train], vert_label_train[:n_val], vert_label_test[:n_test]
        )
    else:
        val_split = int(0.2 * len(img_train))
        val_imgs, val_feats, val_labels, val_attrs = (
            img_train[:val_split], feat_train[:val_split], label_train[:val_split], sens_train[:val_split]
        )
        train_imgs, train_feats, train_labels, train_attrs = (
            img_train[val_split:], feat_train[val_split:], label_train[val_split:], sens_train[val_split:]
        )
        test_imgs, test_feats, test_labels, test_attrs = img_test, feat_test, label_test, sens_test
        train_metadata, val_metadata, test_metadata = (
            vert_train[val_split:], vert_train[:val_split], vert_test
        )
        train_metadata_labels, val_metadata_labels, test_metadata_labels = (
            vert_label_train[val_split:], vert_label_train[:val_split], vert_label_test
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

    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

        model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv = get_model_variant(
            (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
            LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness
        )

        for round_idx in range(ROUNDS):
            print(f"\n--- FL Round {round_idx + 1}/{ROUNDS} ---")
            model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv = FairVFL_train(
                (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
                (val_imgs, val_feats, val_labels, val_attrs, val_metadata),
                (test_imgs, test_feats, test_labels, test_attrs, test_metadata),
                train_labels, val_labels, test_labels,
                lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE,
                epochs=EPOCHS, batch_size=BATCH_SIZE, mode=mode_label,
                model=model, rep_model=rep_model, gender_model=gender_model, age_model=age_model,
                image_embedding_model=image_embedding_model, tabular_embedding_model=tabular_embedding_model, fusion_head=fusion_head,
                gender_cons_adv=gender_cons_adv, age_cons_adv=age_cons_adv
            )

        print("\nEvaluating after final round...")

        test_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in test_imgs])
        val_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in val_imgs])
        test_input = (test_images, test_feats, test_labels, test_attrs[:, 0], test_attrs[:, 1], test_metadata)
        val_input = (val_images, val_feats, val_labels, val_attrs[:, 0], val_attrs[:, 1], val_metadata)
        test_result = evaluation(fusion_head, image_embedding_model, tabular_embedding_model, test_input)
        val_result = evaluation(fusion_head, image_embedding_model, tabular_embedding_model, val_input)

        print(f"Test result: {test_result}")
        print(f"Validation result: {val_result}")
        results_summary[mode_label] = {"test": test_result, "val": val_result}

        # If this is FairVFL, keep the fairness models/rep_model for passive audit
        if with_fairness:
            fair_gender_model = gender_model
            fair_age_model = age_model
            fair_rep_model = rep_model

        # After VanillaFL, perform passive audit
        if not with_fairness and fair_gender_model is not None and fair_age_model is not None:
            print("\n--- Passive Fairness Audit: Probing VanillaFL and FairVFL embeddings with FairVFL fairness heads ---")
            # Gender/Age on VanillaFL
            audit_gender = passive_fairness_audit(
                rep_model,
                fair_gender_model,
                test_imgs, test_feats, test_attrs[:, 0],
                batch_size=BATCH_SIZE
            )
            audit_age = passive_fairness_audit(
                rep_model,
                fair_age_model,
                test_imgs, test_feats, test_attrs[:, 1],
                batch_size=BATCH_SIZE
            )
            print(f"VanillaFL: Gender leakage: acc={audit_gender['acc']:.3f}, f1={audit_gender['f1']:.3f}")
            print(f"VanillaFL: Age leakage: acc={audit_age['acc']:.3f}, f1={audit_age['f1']:.3f}")
            # Gender/Age on FairVFL
            audit_gender_fairvfl = passive_fairness_audit(
                fair_rep_model,
                fair_gender_model,
                test_imgs, test_feats, test_attrs[:, 0],
                batch_size=BATCH_SIZE
            )
            audit_age_fairvfl = passive_fairness_audit(
                fair_rep_model,
                fair_age_model,
                test_imgs, test_feats, test_attrs[:, 1],
                batch_size=BATCH_SIZE
            )
            print(f"FairVFL: Gender leakage: acc={audit_gender_fairvfl['acc']:.3f}, f1={audit_gender_fairvfl['f1']:.3f}")
            print(f"FairVFL: Age leakage: acc={audit_age_fairvfl['acc']:.3f}, f1={audit_age_fairvfl['f1']:.3f}")

    np.save("FairVFL_results.npy", results_summary)
    print(f"\nTotal training time: {round(time.time() - start, 2)} seconds")
