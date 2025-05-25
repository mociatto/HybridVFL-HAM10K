import os
import time
import numpy as np
from data import load_ham10000, DataGenerator
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
HYPER_GENDER = 0.005
HYPER_AGE = 0.001

if __name__ == "__main__":
    print("Loading HAM10000 dataset...")
    data = load_ham10000(DATA_DIR)
    (train_imgs, train_feats, train_labels, train_attrs) = data['train']
    (test_imgs, test_feats, test_labels, test_attrs) = data['test']

    # Split some data for validation
    val_split = int(0.2 * len(train_imgs))
    val_imgs, val_feats, val_labels, val_attrs = (
        train_imgs[:val_split], train_feats[:val_split], train_labels[:val_split], train_attrs[:val_split]
    )
    train_imgs, train_feats, train_labels, train_attrs = (
        train_imgs[val_split:], train_feats[val_split:], train_labels[val_split:], train_attrs[val_split:]
    )

    # Print dataset summary
    print("\nDataset sizes:")
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Create DataGenerators
    train_gen = DataGenerator(train_imgs, train_feats, train_labels, batch_size=BATCH_SIZE)
    val_gen = DataGenerator(val_imgs, val_feats, val_labels, batch_size=BATCH_SIZE)
    test_gen = DataGenerator(test_imgs, test_feats, test_labels, batch_size=BATCH_SIZE)

    results_summary = {}
    start = time.time()
    modes = {"FairVFL": True, "VanillaFL": False}

    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

        model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv = \
            get_model_variant(train_gen, LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness)

        print("Training...")
        FairVFL_train(train_gen, val_gen, test_gen, train_labels, val_labels, test_labels,
                      mode=mode_label, lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE, epochs=EPOCHS)

        print("Evaluating...")
        test_result = evaluation(model, rep_model, test_gen, test_labels, test_attrs)
        val_result = evaluation(model, rep_model, val_gen, val_labels, val_attrs)

        print(f"Test result: {test_result}")
        print(f"Validation result: {val_result}")
        results_summary[mode_label] = {"test": test_result, "val": val_result}

    np.save("FairVFL_results.npy", results_summary)
    print(f"\nTotal training time: {round(time.time() - start, 2)} seconds")
