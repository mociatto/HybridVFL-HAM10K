import os
import time
import numpy as np
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import FairVFL_train
from evaluate import evaluation

from sklearn.metrics import f1_score

# Import status configuration
from status_config import get_status, get_training_status, get_completion_status, get_evaluation_status

# SocketIO dashboard support
from socketio import Client

sio = Client()
try:
    sio.connect('http://localhost:5050')
except Exception as e:
    print("Dashboard not connected:", e)


def send_metrics(metrics):
    try:
        sio.emit('metrics_update', metrics)
    except Exception:
        pass


def passive_fairness_audit(rep_model, fairness_model, img_paths, tabular, targets, batch_size=32):
    images = np.array([load_and_preprocess_image(p, (224, 224)) for p in img_paths])
    embeddings = rep_model.predict([images, tabular], batch_size=batch_size)
    preds = np.argmax(fairness_model.predict(embeddings, batch_size=batch_size), axis=1)
    acc = np.mean(preds == targets)
    f1 = f1_score(targets, preds, average='weighted')
    return {"acc": acc, "f1": f1}


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 8
EPOCHS = 5
ROUNDS = 3
PERCENTAGE = 0.05
LR = 0.001
HYPER_GENDER = 0.001
HYPER_AGE = 0.0005

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
    print(
        f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)} | Percentage: {int(PERCENTAGE * 100)}%")
    print(f"Rounds: {ROUNDS} | Epochs per round: {EPOCHS} | Batch size: {BATCH_SIZE}\n")

    results_summary = {}
    start = time.time()
    modes = {"FairVFL": True} # , "VanillaFL": False}

    # To track FairVFL fairness models for passive audit
    fair_gender_model = None
    fair_age_model = None
    fair_rep_model = None

    # Initialize histories for dashboard (example for FairVFL)
    fusion_acc_history = []
    fusion_loss_history = []
    client1_acc_history = []
    client2_acc_history = []
    gender_acc = [0, 0]  # [male, female]
    fairness_radar_labels = ["≤30", "31–45", "46–60", "61–75", "≥76"]
    fairness_radar_values = [0, 0, 0, 0, 0]

    # After loading data and before training loop:
    initial_metrics = {
        "batch_size": BATCH_SIZE,
        "round": 0,
        "rounds_total": ROUNDS,
        "epochs_per_round": EPOCHS,
        "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
        "feature_class": 7,
        "running_time": "00:00",
        "start_time": time.time(),
        "current_status": get_status("LOADING_DATASET"),
        "train_acc_history": [],
        "val_acc_history": [],
        "train_loss_history": [],
        "gender_acc_history": [],
        "age_acc_history": [],
        "adv_loss_history": [],
        "leak_gender_image": 0,
        "leak_age_image": 0,
        "leak_gender_tabular": 0,
        "leak_age_tabular": 0,
    }
    send_metrics(initial_metrics)

    # Initialize accumulated histories across all rounds
    accumulated_train_acc_history = []
    accumulated_val_acc_history = []
    accumulated_train_loss_history = []
    accumulated_gender_acc_history = []
    accumulated_age_acc_history = []
    accumulated_adv_loss_history = []
    
    # Initialize leakage scores
    leak_gender_image = 0
    leak_age_image = 0
    leak_gender_tabular = 0
    leak_age_tabular = 0
    
    # Initialize final performance metrics
    final_test_accuracy = 0
    final_f1_score = 0

    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

        # Send status update
        status_metrics = {
            "current_status": get_status("TRAINING_START", mode=mode_label),
            "batch_size": BATCH_SIZE,
            "round": 0,
            "rounds_total": ROUNDS,
            "epochs_per_round": EPOCHS,
            "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
            "feature_class": 7,
            "start_time": start,
        }
        send_metrics(status_metrics)

        model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv = get_model_variant(
            (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
            LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness
        )

        for round_idx in range(ROUNDS):
            print(f"\n--- FL Round {round_idx + 1}/{ROUNDS} ---")
            
            # Send round start status
            round_status_metrics = {
                "current_status": get_training_status(mode_label, round_idx + 1, ROUNDS),
                "batch_size": BATCH_SIZE,
                "round": round_idx + 1,
                "rounds_total": ROUNDS,
                "epochs_per_round": EPOCHS,
                "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
                "feature_class": 7,
                "start_time": start,
            }
            send_metrics(round_status_metrics)

            (
                model, rep_model, gender_model, age_model,
                image_embedding_model, tabular_embedding_model, fusion_head,
                gender_cons_adv, age_cons_adv,
                train_acc_history, val_acc_history, train_loss_history, gender_acc_history, age_acc_history, adv_loss_history
            ) = FairVFL_train(
                (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
                (val_imgs, val_feats, val_labels, val_attrs, val_metadata),
                (test_imgs, test_feats, test_labels, test_attrs, test_metadata),
                train_labels, val_labels, test_labels,
                lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE,
                epochs=EPOCHS, batch_size=BATCH_SIZE, mode=mode_label,
                model=model, rep_model=rep_model, gender_model=gender_model, age_model=age_model,
                image_embedding_model=image_embedding_model, tabular_embedding_model=tabular_embedding_model,
                fusion_head=fusion_head,
                gender_cons_adv=gender_cons_adv, age_cons_adv=age_cons_adv,
                metrics_callback=send_metrics, current_round=round_idx + 1, total_rounds=ROUNDS, start_time=start,
                accumulated_histories=(accumulated_train_acc_history, accumulated_val_acc_history, accumulated_train_loss_history, 
                                     accumulated_gender_acc_history, accumulated_age_acc_history, accumulated_adv_loss_history)
            )

            # Accumulate histories across rounds
            accumulated_train_acc_history.extend(train_acc_history)
            accumulated_val_acc_history.extend(val_acc_history)
            accumulated_train_loss_history.extend(train_loss_history)
            accumulated_gender_acc_history.extend(gender_acc_history)
            accumulated_age_acc_history.extend(age_acc_history)
            accumulated_adv_loss_history.extend(adv_loss_history)

            # Send metrics to dashboard after each round (live update)
            metrics = {
                "batch_size": BATCH_SIZE,
                "round": round_idx + 1,
                "rounds_total": ROUNDS,
                "epochs_per_round": EPOCHS,
                "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
                "feature_class": 7,
                "running_time": "{:02d}:{:02d}".format(int((time.time() - start) // 60), int((time.time() - start) % 60)),
                "current_status": get_completion_status(mode_label, round_idx + 1, ROUNDS),
                "train_acc_history": accumulated_train_acc_history,
                "val_acc_history": accumulated_val_acc_history,
                "train_loss_history": accumulated_train_loss_history,
                "gender_acc_history": accumulated_gender_acc_history,
                "age_acc_history": accumulated_age_acc_history,
                "adv_loss_history": accumulated_adv_loss_history,
                # Leakage scores will be sent after training completes
                "leak_gender_image": 0,
                "leak_age_image": 0,
                "leak_gender_tabular": 0,
                "leak_age_tabular": 0,
            }
            send_metrics(metrics)

        print("\nEvaluating after final round...")
        
        # Send evaluation status
        eval_status_metrics = {
            "current_status": get_evaluation_status(mode_label),
            "batch_size": BATCH_SIZE,
            "round": ROUNDS,
            "rounds_total": ROUNDS,
            "epochs_per_round": EPOCHS,
            "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
            "feature_class": 7,
            "start_time": start,
        }
        send_metrics(eval_status_metrics)

        test_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in test_imgs])
        val_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in val_imgs])
        test_input = (test_images, test_feats, test_labels, test_attrs[:, 0], test_attrs[:, 1], test_metadata)
        val_input = (val_images, val_feats, val_labels, val_attrs[:, 0], val_attrs[:, 1], val_metadata)
        test_result = evaluation(fusion_head, image_embedding_model, tabular_embedding_model, test_input)
        val_result = evaluation(fusion_head, image_embedding_model, tabular_embedding_model, val_input)

        from evaluate import (
            evaluate_image_only, evaluate_tabular_only, evaluate_fusion, fairness_leakage
        )

        img_only_metrics = evaluate_image_only(image_embedding_model, test_images, test_labels, batch_size=BATCH_SIZE)
        tab_only_metrics = evaluate_tabular_only(tabular_embedding_model, test_metadata, test_labels,
                                                 batch_size=BATCH_SIZE)

        # -- Fusion improvement over solo clients
        fusion_acc = test_result["acc"]
        fusion_f1 = test_result["f1"]
        best_solo_acc = max(img_only_metrics["acc"], tab_only_metrics["acc"])
        best_solo_f1 = max(img_only_metrics["f1"], tab_only_metrics["f1"])
        print(f"Fusion - Best solo Δaccuracy: {fusion_acc - best_solo_acc:.4f}, ΔF1: {fusion_f1 - best_solo_f1:.4f}")

        # -- Fairness/leakage on image and tabular embeddings (if fairness models exist)
        if gender_model is not None:
            print("\nLeakage (Gender) on image embeddings:")
            
            # Send status update for leakage calculation
            leakage_status_metrics = {
                "current_status": get_status("CALCULATING_LEAKAGE"),
                "batch_size": BATCH_SIZE,
                "round": ROUNDS,
                "rounds_total": ROUNDS,
                "epochs_per_round": EPOCHS,
                "start_time": start,
            }
            send_metrics(leakage_status_metrics)
            
            leak_gender_image = fairness_leakage(
                rep_model,
                gender_model,
                test_images,
                np.zeros_like(test_metadata),  # only image input, zero tabular
                test_attrs[:, 0],
                batch_size=BATCH_SIZE,
                name="GenderLeakage-Image"
            )["acc"]
            print("Leakage (Gender) on tabular embeddings:")
            leak_gender_tabular = fairness_leakage(
                rep_model,
                gender_model,
                np.zeros_like(test_images),  # only tabular input, zero image
                test_metadata,
                test_attrs[:, 0],
                batch_size=BATCH_SIZE,
                name="GenderLeakage-Tabular"
            )["acc"]
        else:
            leak_gender_image = 0
            leak_gender_tabular = 0

        if age_model is not None:
            print("Leakage (Age) on image embeddings:")
            leak_age_image = fairness_leakage(
                rep_model,
                age_model,
                test_images,
                np.zeros_like(test_metadata),
                test_attrs[:, 1],
                batch_size=BATCH_SIZE,
                name="AgeLeakage-Image"
            )["acc"]
            print("Leakage (Age) on tabular embeddings:")
            leak_age_tabular = fairness_leakage(
                rep_model,
                age_model,
                np.zeros_like(test_images),
                test_metadata,
                test_attrs[:, 1],
                batch_size=BATCH_SIZE,
                name="AgeLeakage-Tabular"
            )["acc"]
        else:
            leak_age_image = 0
            leak_age_tabular = 0

        # Send final metrics with leakage scores after training completes
        final_metrics = {
            "batch_size": BATCH_SIZE,
            "round": ROUNDS,
            "rounds_total": ROUNDS,
            "epochs_per_round": EPOCHS,
            "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
            "feature_class": 7,
            "running_time": "{:02d}:{:02d}".format(int((time.time() - start) // 60), int((time.time() - start) % 60)),
            "current_status": get_status("EVALUATION_COMPLETED"),
            "start_time": start,
            "train_acc_history": accumulated_train_acc_history,
            "val_acc_history": accumulated_val_acc_history,
            "train_loss_history": accumulated_train_loss_history,
            "gender_acc_history": accumulated_gender_acc_history,
            "age_acc_history": accumulated_age_acc_history,
            "adv_loss_history": accumulated_adv_loss_history,
            "leak_gender_image": leak_gender_image,
            "leak_age_image": leak_age_image,
            "leak_gender_tabular": leak_gender_tabular,
            "leak_age_tabular": leak_age_tabular,
            "test_accuracy": final_test_accuracy,
            "f1_score": final_f1_score,
        }
        
        send_metrics(final_metrics)

        results_summary[mode_label] = {"test": test_result, "val": val_result}

        # If this is FairVFL, keep the fairness models/rep_model for passive audit
        if with_fairness:
            fair_gender_model = gender_model
            fair_age_model = age_model
            fair_rep_model = rep_model

        # After VanillaFL, perform passive audit
        # if not with_fairness and fair_gender_model is not None and fair_age_model is not None:
        #     print(
        #         "\n--- Passive Fairness Audit: Probing VanillaFL and FairVFL embeddings with FairVFL fairness heads ---")
        #     # Gender/Age on VanillaFL
        #     audit_gender = passive_fairness_audit(
        #         rep_model,
        #         fair_gender_model,
        #         test_imgs, test_feats, test_attrs[:, 0],
        #         batch_size=BATCH_SIZE
        #     )
        #     audit_age = passive_fairness_audit(
        #         rep_model,
        #         fair_age_model,
        #         test_imgs, test_feats, test_attrs[:, 1],
        #         batch_size=BATCH_SIZE
        #     )
        #     print(f"VanillaFL: Gender leakage: acc={audit_gender['acc']:.3f}, f1={audit_gender['f1']:.3f}")
        #     print(f"VanillaFL: Age leakage: acc={audit_age['acc']:.3f}, f1={audit_age['f1']:.3f}")
        #     # Gender/Age on FairVFL
        #     audit_gender_fairvfl = passive_fairness_audit(
        #         fair_rep_model,
        #         fair_gender_model,
        #         test_imgs, test_feats, test_attrs[:, 0],
        #         batch_size=BATCH_SIZE
        #     )
        #     audit_age_fairvfl = passive_fairness_audit(
        #         fair_rep_model,
        #         fair_age_model,
        #         test_imgs, test_feats, test_attrs[:, 1],
        #         batch_size=BATCH_SIZE
        #     )
        #     print(
        #         f"FairVFL: Gender leakage: acc={audit_gender_fairvfl['acc']:.3f}, f1={audit_gender_fairvfl['f1']:.3f}")
        #     print(f"FairVFL: Age leakage: acc={audit_age_fairvfl['acc']:.3f}, f1={audit_age_fairvfl['f1']:.3f}")

        # Capture final performance metrics
        final_test_accuracy = test_result["acc"]
        final_f1_score = test_result["f1"]

    np.save("FairVFL_results.npy", results_summary)
    total_time = time.time() - start
    print(f"\nTotal training time: {round(total_time, 2)} seconds")
    
    # Send final completion status with actual training time
    completion_metrics = {
        "training_completed": True,
        "current_status": get_status("TRAINING_COMPLETED"),
        "final_running_time": "{:02d}:{:02d}".format(int(total_time // 60), int(total_time % 60)),
        "batch_size": BATCH_SIZE,
        "round": ROUNDS,
        "rounds_total": ROUNDS,
        "epochs_per_round": EPOCHS,
        "sample_count": len(train_imgs) + len(val_imgs) + len(test_imgs),
        "feature_class": 7,
        "start_time": start,
        "train_acc_history": accumulated_train_acc_history,
        "val_acc_history": accumulated_val_acc_history,
        "train_loss_history": accumulated_train_loss_history,
        "gender_acc_history": accumulated_gender_acc_history,
        "age_acc_history": accumulated_age_acc_history,
        "adv_loss_history": accumulated_adv_loss_history,
        "leak_gender_image": leak_gender_image,
        "leak_age_image": leak_age_image,
        "leak_gender_tabular": leak_gender_tabular,
        "leak_age_tabular": leak_age_tabular,
        "test_accuracy": final_test_accuracy,
        "f1_score": final_f1_score,
    }
    send_metrics(completion_metrics)
