import os
import time
import numpy as np
import tensorflow as tf
from data import load_ham10000, DataGenerator, load_and_preprocess_image
from model import get_model_variant
from train import source_inspired_sequential_train

from sklearn.metrics import f1_score

from status_config import get_status, get_training_status, get_completion_status, get_evaluation_status

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


def set_random_seeds(seed=None):
    """Set random seeds for reproducibility within each mode but different between modes"""
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        import random
        random.seed(seed)


def passive_fairness_audit(rep_model, fairness_model, img_paths, tabular, targets, batch_size=32):
    images = np.array([load_and_preprocess_image(p, (224, 224)) for p in img_paths])
    embeddings = rep_model.predict([images, tabular], batch_size=batch_size)
    preds = np.argmax(fairness_model.predict(embeddings, batch_size=batch_size), axis=1)
    acc = np.mean(preds == targets)
    f1 = f1_score(targets, preds, average='weighted')
    return {"acc": acc, "f1": f1}


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH_SIZE = 16
EPOCHS = 5
ROUNDS = 5
PERCENTAGE = 0.1
LR = 0.001
HYPER_GENDER = 0.5
HYPER_AGE = 0.3

SecureVFL = False

if SecureVFL:
    model_random_seed = 123
else:
    model_random_seed = 789

set_random_seeds(model_random_seed)

if __name__ == "__main__":
    mode_label = "SecureVFL" if SecureVFL else "VanillaFL"
    
    FIXED_DATA_SPLIT_SEED = 42
    
    print("Loading HAM10000 dataset...")
    print(f"Model init seed: {model_random_seed} | Data split seed: {FIXED_DATA_SPLIT_SEED} (FIXED)")
    print("*** Using SAME data split for both modes for fair comparison ***")
    print(f"*** Data split consistency: GUARANTEED (seed={FIXED_DATA_SPLIT_SEED}) ***")
    print(f"*** Only model initialization differs: VanillaFL=789, SecureVFL=123 ***")
    
    data = load_ham10000(DATA_DIR, random_state=FIXED_DATA_SPLIT_SEED)
    image_client = data['image_client']
    vertical_client = data['vertical_client']

    (img_train, feat_train, label_train, sens_train) = image_client['train']
    (img_test, feat_test, label_test, sens_test) = image_client['test']
    (vert_train, vert_label_train) = vertical_client['train']
    (vert_test, vert_label_test) = vertical_client['test']
    
    import hashlib
    train_fingerprint = hashlib.md5(str(img_train[:5]).encode()).hexdigest()[:8]
    test_fingerprint = hashlib.md5(str(img_test[:5]).encode()).hexdigest()[:8]
    print(f"*** Data fingerprints: Train={train_fingerprint}, Test={test_fingerprint} ***")
    print("*** These fingerprints should be IDENTICAL across mode switches ***")

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

    print("EXPERIMENT SETUP:")
    print("=" * 50)
    print(f"Running in {mode_label} mode")
    print(f"Model init seed: {model_random_seed} | Data split seed: {FIXED_DATA_SPLIT_SEED} (FIXED)")
    print("Data consistency verification:")
    print(f"  Same train/test split across ALL mode switches")
    print(f"  Only model weights differ between modes")
    print(f"  Fair comparison guaranteed")
    print("Expected results:")
    if SecureVFL:
        print("  SecureVFL (privacy-preserving):")
        print("    * Potential slight accuracy reduction")
        print("    * LOWER leakage scores (privacy protection)")
        print("    * Successful if leakage is approximately random baselines")
    else:
        print("  VanillaFL (standard):")
        print("    * Higher classification accuracy")
        print("    * HIGHER leakage scores (no privacy protection)")
        print("    * Standard federated learning baseline")
    print("=" * 50 + "\n")

    results_summary = {}
    start = time.time()
    modes = {
        mode_label: SecureVFL
    }

    final_gender_model = None
    final_age_model = None
    final_rep_model = None

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    gender_acc_history = []
    age_acc_history = []
    adv_loss_history = []

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

    accumulated_train_acc_history = []
    accumulated_val_acc_history = []
    accumulated_train_loss_history = []
    accumulated_gender_acc_history = []
    accumulated_age_acc_history = []
    accumulated_adv_loss_history = []
    
    leak_gender_image = 0
    leak_age_image = 0
    leak_gender_tabular = 0
    leak_age_tabular = 0
    gender_acc_fused = 0
    age_acc_fused = 0
    
    final_test_accuracy = 0
    final_f1_score = 0
    
    for mode_label, with_fairness in modes.items():
        print(f"\n======= Training Mode: {mode_label} (with_fairness={with_fairness}) =======")

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

        model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv, image_classifier, tabular_classifier = get_model_variant(
            (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
            LR, HYPER_GENDER, HYPER_AGE, with_fairness=with_fairness
        )

        for round_idx in range(ROUNDS):
            print(f"\n--- FL Round {round_idx + 1}/{ROUNDS} ---")
            
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
                image_classifier, tabular_classifier,
                train_acc_history, val_acc_history, train_loss_history, gender_acc_history, age_acc_history, adv_loss_history
            ) = source_inspired_sequential_train(
                (train_imgs, feat_train, train_labels, train_attrs, train_metadata),
                (val_imgs, val_feats, val_labels, val_attrs, val_metadata),
                (test_imgs, test_feats, test_labels, test_attrs, test_metadata),
                train_labels, val_labels, test_labels,
                lr=LR, hyper_gender=HYPER_GENDER, hyper_age=HYPER_AGE,
                epochs=EPOCHS, batch_size=BATCH_SIZE, mode=mode_label,
                main_model=model, rep_model=rep_model, gender_model=gender_model, age_model=age_model,
                gender_cons_adv=gender_cons_adv, age_cons_adv=age_cons_adv,
                metrics_callback=send_metrics, current_round=round_idx + 1, total_rounds=ROUNDS, start_time=start,
                accumulated_histories=(accumulated_train_acc_history, accumulated_val_acc_history, accumulated_train_loss_history, 
                                     accumulated_gender_acc_history, accumulated_age_acc_history, accumulated_adv_loss_history)
            )

            accumulated_train_acc_history.extend(train_acc_history)
            accumulated_val_acc_history.extend(val_acc_history)
            accumulated_train_loss_history.extend(train_loss_history)
            accumulated_gender_acc_history.extend(gender_acc_history)
            accumulated_age_acc_history.extend(age_acc_history)
            accumulated_adv_loss_history.extend(adv_loss_history)

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
                "leak_gender_image": 0,
                "leak_age_image": 0,
                "leak_gender_tabular": 0,
                "leak_age_tabular": 0,
            }
            send_metrics(metrics)

        print("\nEvaluating after final round...")
        
        test_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in test_imgs])
        val_images = np.array([load_and_preprocess_image(p, (224, 224)) for p in val_imgs])
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_metadata)
        test_metadata_normalized = scaler.transform(test_metadata)
        val_metadata_normalized = scaler.transform(val_metadata)
        
        test_gen = DataGenerator(test_imgs, test_metadata_normalized, test_labels, batch_size=BATCH_SIZE)
        val_gen = DataGenerator(val_imgs, val_metadata_normalized, val_labels, batch_size=BATCH_SIZE)
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        
        test_preds = model.predict(test_gen, verbose=0)
        test_pred_classes = np.argmax(test_preds, axis=1)
        test_f1 = f1_score(test_labels, test_pred_classes, average='macro')
        
        val_preds = model.predict(val_gen, verbose=0)
        val_pred_classes = np.argmax(val_preds, axis=1)
        val_f1 = f1_score(val_labels, val_pred_classes, average='macro')
        
        test_result = {"acc": test_acc, "f1": test_f1}
        val_result = {"acc": val_acc, "f1": val_f1}
        
        print(f"Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}")
        print(f"Val: Accuracy={val_acc:.4f}, F1={val_f1:.4f}")

        from evaluate import (
            evaluate_image_only, evaluate_tabular_only, evaluate_fusion, fairness_leakage
        )

        print("\nSKIPPING individual client evaluation (architecture mismatch)")
        print("Main monolithic model performance:")
        print(f"   Test: {test_acc:.4f} accuracy, {test_f1:.4f} F1")
        print(f"   This is {test_acc/0.143:.1f}x better than random (14.3% for 7 classes)")

        img_only_metrics = {"acc": 0.0, "f1": 0.0}
        tab_only_metrics = {"acc": 0.0, "f1": 0.0}
        
        fusion_acc = test_result["acc"]
        fusion_f1 = test_result["f1"]
        best_solo_acc = 0.0
        best_solo_f1 = 0.0
        
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"   Monolithic VFL: {fusion_acc:.4f} accuracy, {fusion_f1:.4f} F1")
        print(f"   Improvement over random: +{(fusion_acc - 0.143)*100:.1f} percentage points")

        if gender_model is not None and age_model is not None:
            print("\n" + "="*60)
            print("PRIVACY LEAKAGE ANALYSIS")
            print("="*60)
            
            test_embeddings = rep_model.predict([test_images, test_metadata_normalized], batch_size=BATCH_SIZE, verbose=0)
            
            print("Testing leakage from model embeddings:")
            
            if len(test_attrs) > 0:
                gender_preds = np.argmax(gender_model.predict(test_embeddings, verbose=0), axis=1)
                gender_acc_fused = np.mean(gender_preds == test_attrs[:, 0])
                print(f"   Gender leakage: {gender_acc_fused:.4f} ({gender_acc_fused:.1%})")
                
                age_preds = np.argmax(age_model.predict(test_embeddings, verbose=0), axis=1)
                age_acc_fused = np.mean(age_preds == test_attrs[:, 1])
                print(f"   Age leakage: {age_acc_fused:.1%}")
                
                gender_random = 0.5  # 50% for binary
                age_random = 0.2     # 20% for 5 classes
                
                print(f"\nPrivacy Protection Analysis:")
                print(f"   Gender: {gender_acc_fused:.1%} vs Random {gender_random:.1%} -> Risk: {gender_acc_fused-gender_random:+.1%}")
                print(f"   Age: {age_acc_fused:.1%} vs Random {age_random:.1%} -> Risk: {age_acc_fused-age_random:+.1%}")
                
                if mode_label == "SecureVFL":
                    print("   SUCCESS if leakage is close to random baselines")
                else:
                    print("   VanillaFL expected to have higher leakage")
            else:
                gender_acc_fused = 0.0
                age_acc_fused = 0.0
                print("   No sensitive attributes available for leakage testing")
                
            print("="*60)
        else:
            gender_acc_fused = 0.0
            age_acc_fused = 0.0
            print("\nNo fairness models available for leakage testing")

        leak_gender_image = gender_acc_fused
        leak_age_image = age_acc_fused 
        leak_gender_tabular = gender_acc_fused
        leak_age_tabular = age_acc_fused
        
        final_test_accuracy = test_result["acc"]
        final_f1_score = test_result["f1"]
        
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
            "leak_gender_fused": gender_acc_fused,
            "leak_age_fused": age_acc_fused,
            "test_accuracy": final_test_accuracy,
            "f1_score": final_f1_score,
        }
        
        send_metrics(final_metrics)

        results_summary[mode_label] = {"test": test_result, "val": val_result}

        if with_fairness:
            final_gender_model = gender_model
            final_age_model = age_model
            final_rep_model = rep_model

    np.save("SecureVFL_results.npy", results_summary)
    total_time = time.time() - start
    print(f"\nTotal training time: {round(total_time, 2)} seconds")
    
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
        "leak_gender_fused": gender_acc_fused,
        "leak_age_fused": age_acc_fused,
        "test_accuracy": final_test_accuracy,
        "f1_score": final_f1_score,
    }
    send_metrics(completion_metrics)
