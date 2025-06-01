import numpy as np
from data import DataGenerator, load_and_preprocess_image
from model import get_model_variant
import tensorflow as tf
import time

# Import status configuration
from status_config import get_status

def preprocess_images(paths, target_size=(224, 224)):
    return np.array([load_and_preprocess_image(p, target_size) for p in paths])

def FairVFL_train(
    train_data, val_data, test_data,
    train_label, val_label, test_label,
    mode='FairVFL', lr=0.001, hyper_gender=0.005, hyper_age=0.001,
    batch_size=128, epochs=10,
    model=None, rep_model=None, gender_model=None, age_model=None,
    image_embedding_model=None, tabular_embedding_model=None, fusion_head=None,
    gender_cons_adv=None, age_cons_adv=None,
    metrics_callback=None, current_round=0, total_rounds=1, start_time=None,
    accumulated_histories=None
):
    with_fairness = (mode == 'FairVFL')
    train_attr = train_data[3] if with_fairness else None

    # Unpack metadata client input (fifth element in train_data/val_data/test_data tuple)
    metadata_train = train_data[4]
    metadata_val = val_data[4]
    metadata_test = test_data[4]

    if model is None or rep_model is None or \
       (with_fairness and (gender_model is None or age_model is None or gender_cons_adv is None or age_cons_adv is None)):
        model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv = \
            get_model_variant(train_data, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age, with_fairness=with_fairness)

    train_gen = DataGenerator(train_data[0], train_data[1], train_label, batch_size=batch_size)
    val_gen = DataGenerator(val_data[0], val_data[1], val_label, batch_size=batch_size)
    test_gen = DataGenerator(test_data[0], test_data[1], test_label, batch_size=batch_size)

    best_val_acc = 0
    best_weights = None

    # Initialize histories for dashboard
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    gender_acc_history = []
    age_acc_history = []
    adv_loss_history = []

    # Get accumulated histories from previous rounds for dashboard display
    if accumulated_histories is not None:
        (accumulated_train_acc_history, accumulated_val_acc_history, accumulated_train_loss_history, 
         accumulated_gender_acc_history, accumulated_age_acc_history, accumulated_adv_loss_history) = accumulated_histories
    else:
        accumulated_train_acc_history = []
        accumulated_val_acc_history = []
        accumulated_train_loss_history = []
        accumulated_gender_acc_history = []
        accumulated_age_acc_history = []
        accumulated_adv_loss_history = []

    if start_time is None:
        start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # --- 1. Local Training: Image Client (extract embeddings) ---
        train_images = preprocess_images(train_data[0], target_size=(224, 224))
        image_embeddings = image_embedding_model.predict(train_images, batch_size=batch_size, verbose=0)

        # --- 2. Local Training: Metadata Client (extract embeddings) ---
        tabular_embeddings = tabular_embedding_model.predict(metadata_train, batch_size=batch_size, verbose=0)

        # --- 3. Server-side Fusion + Classifier Training (end-to-end for simplicity) ---
        # Concatenate local embeddings and train fusion/classifier on them
        fused_embeddings = np.concatenate([image_embeddings, tabular_embeddings], axis=1)
        fusion_head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = fusion_head.fit(fused_embeddings, train_label, batch_size=batch_size, epochs=1, verbose=1)
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation
        val_images = preprocess_images(val_data[0], target_size=(224, 224))
        val_image_embeddings = image_embedding_model.predict(val_images, batch_size=batch_size, verbose=0)
        val_tabular_embeddings = tabular_embedding_model.predict(metadata_val, batch_size=batch_size, verbose=0)
        val_fused_embeddings = np.concatenate([val_image_embeddings, val_tabular_embeddings], axis=1)
        val_loss, val_acc = fusion_head.evaluate(val_fused_embeddings, val_label, batch_size=batch_size, verbose=0)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = fusion_head.get_weights()

        # --- 4. Fairness/adversarial training using fused embeddings ---
        if with_fairness:
            print("Generating embeddings for fairness heads...")
            gender_targets = train_attr[:, 0]
            age_targets = train_attr[:, 1]

            # Debug: Print shapes
            print(f"DEBUG: fused_embeddings shape: {fused_embeddings.shape}")
            print(f"DEBUG: gender_targets shape: {gender_targets.shape}")
            print(f"DEBUG: age_targets shape: {age_targets.shape}")

            print("Training gender model...")
            gender_hist = gender_model.fit(fused_embeddings, gender_targets, batch_size=batch_size, epochs=1, verbose=1)
            gender_acc = gender_hist.history['accuracy'][0]
            gender_acc_history.append(gender_acc)

            print("Training age model...")
            age_hist = age_model.fit(fused_embeddings, age_targets, batch_size=batch_size, epochs=1, verbose=1)
            age_acc = age_hist.history['accuracy'][0]
            age_acc_history.append(age_acc)

            print("Training adversarial models...")
            # Add safety check for adversarial training
            if fused_embeddings.shape[0] > 0:
                try:
                    # Use smaller batch size for adversarial training to avoid memory issues
                    adv_batch_size = min(batch_size, fused_embeddings.shape[0])
                    print(f"DEBUG: Training adversarial model with batch_size={adv_batch_size}")
                    
                    adv_hist = gender_cons_adv.fit(fused_embeddings, fused_embeddings, 
                                                 batch_size=adv_batch_size, epochs=1, verbose=1)
                    adv_loss = adv_hist.history['loss'][0]
                    adv_loss_history.append(adv_loss)
                    
                    age_cons_adv.fit(fused_embeddings, fused_embeddings, 
                                   batch_size=adv_batch_size, epochs=1, verbose=1)
                except Exception as e:
                    print(f"ERROR in adversarial training: {e}")
                    print(f"Skipping adversarial training for this epoch")
                    adv_loss_history.append(0.0)  # Use 0 as fallback
            else:
                print("WARNING: Empty fused_embeddings, skipping adversarial training")
                adv_loss_history.append(0.0)

            print("Adversarial representation confusion step...")
            with tf.GradientTape() as tape:
                tape.watch([image_embedding_model.trainable_variables, tabular_embedding_model.trainable_variables])
                rep_outputs = tf.concat([
                    image_embedding_model(train_images, training=True),
                    tabular_embedding_model(metadata_train, training=True)
                ], axis=1)
                gender_preds = gender_model(rep_outputs, training=False)
                age_preds = age_model(rep_outputs, training=False)
                gender_adv_loss = tf.keras.losses.sparse_categorical_crossentropy(gender_targets, gender_preds)
                age_adv_loss = tf.keras.losses.sparse_categorical_crossentropy(age_targets, age_preds)
                adv_loss = hyper_gender * tf.reduce_mean(gender_adv_loss) + hyper_age * tf.reduce_mean(age_adv_loss)
                total_loss = -adv_loss
            grads = tape.gradient(total_loss, image_embedding_model.trainable_variables + tabular_embedding_model.trainable_variables)
            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(
                zip(grads, image_embedding_model.trainable_variables + tabular_embedding_model.trainable_variables)
            )
        else:
            # For VanillaFL, append None or 0 to fairness/adversarial histories
            gender_acc_history.append(None)
            age_acc_history.append(None)
            adv_loss_history.append(None)

        # Per-epoch dashboard update
        if metrics_callback is not None:
            # Create current combined histories for dashboard display
            current_train_acc_history = accumulated_train_acc_history + train_acc_history
            current_val_acc_history = accumulated_val_acc_history + val_acc_history
            current_train_loss_history = accumulated_train_loss_history + train_loss_history
            current_gender_acc_history = accumulated_gender_acc_history + gender_acc_history
            current_age_acc_history = accumulated_age_acc_history + age_acc_history
            current_adv_loss_history = accumulated_adv_loss_history + adv_loss_history
            
            # Determine current status based on mode and fairness training
            if with_fairness:
                current_status = get_status("TRAINING_EPOCH", 
                                          mode=mode, round=current_round, 
                                          epoch=epoch + 1, total_epochs=epochs)
            else:
                current_status = get_status("TRAINING_EPOCH", 
                                          mode=mode, round=current_round, 
                                          epoch=epoch + 1, total_epochs=epochs)
            
            metrics_callback({
                "batch_size": batch_size,
                "round": current_round,
                "rounds_total": total_rounds,
                "epochs_per_round": epochs,
                "sample_count": len(train_data[0]) + len(val_data[0]) + len(test_data[0]),
                "feature_class": 7,
                "running_time": "{:02d}:{:02d}".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)),
                "current_status": current_status,
                "train_acc_history": current_train_acc_history,
                "val_acc_history": current_val_acc_history,
                "train_loss_history": current_train_loss_history,
                "gender_acc_history": current_gender_acc_history,
                "age_acc_history": current_age_acc_history,
                "adv_loss_history": current_adv_loss_history,
                "leak_gender_image": 0,
                "leak_age_image": 0,
                "leak_gender_tabular": 0,
                "leak_age_tabular": 0,
                "start_time": start_time,
            })

    if best_weights is not None:
        fusion_head.set_weights(best_weights)

    # --- Final test accuracy (on fused test embeddings) ---
    test_images = preprocess_images(test_data[0], target_size=(224, 224))
    test_image_embeddings = image_embedding_model.predict(test_images, batch_size=batch_size, verbose=0)
    test_tabular_embeddings = tabular_embedding_model.predict(metadata_test, batch_size=batch_size, verbose=0)
    test_fused_embeddings = np.concatenate([test_image_embeddings, test_tabular_embeddings], axis=1)
    test_loss, test_acc = fusion_head.evaluate(test_fused_embeddings, test_label, batch_size=batch_size, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Return models and histories for dashboard
    return (
        model, rep_model, gender_model, age_model,
        image_embedding_model, tabular_embedding_model, fusion_head,
        gender_cons_adv, age_cons_adv,
        train_acc_history, val_acc_history, train_loss_history, gender_acc_history, age_acc_history, adv_loss_history
    )
