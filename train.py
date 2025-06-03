import numpy as np
from data import DataGenerator, load_and_preprocess_image
from model import get_model_variant, get_fairness_model
import tensorflow as tf
import time
import traceback
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Import SMOTE and other resampling techniques
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Import status configuration
from status_config import get_status

def apply_smote_balancing(metadata_train, train_labels, balance_strategy='auto'):
    """
    Apply SMOTE to balance tabular/metadata features toward dominant class size
    
    Args:
        metadata_train: Tabular features to balance
        train_labels: Corresponding labels
        balance_strategy: 'auto' uses dominant class size as target
    
    Returns:
        balanced_metadata, balanced_labels
    """
    print("Applying SMOTE balancing to tabular data...")
    
    # Check current class distribution
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    print("   Original distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"     Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    # Determine target size based on dominant class
    max_class_size = max(class_counts.values())
    target_size = max_class_size  # Use dominant class size as target
    
    print(f"   Target samples per class: {target_size} (based on dominant class)")
    
    # Create sampling strategy dictionary
    sampling_strategy = {}
    for class_id, count in class_counts.items():
        if count < target_size:
            sampling_strategy[class_id] = target_size
    
    # Apply SMOTE with adaptive sampling strategy
    # Set k_neighbors based on smallest class size
    min_class_size = min(class_counts.values())
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    
    if k_neighbors < 1:
        print("   Warning: Very small classes detected, using ADASYN instead of SMOTE")
        sampler = ADASYN(random_state=42, n_neighbors=1, sampling_strategy=sampling_strategy)
    else:
        # Use SMOTETomek for better quality synthetic samples
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        sampler = SMOTETomek(smote=smote, random_state=42)
    
    try:
        balanced_metadata, balanced_labels = sampler.fit_resample(metadata_train, train_labels)
        
        # Show balanced distribution
        balanced_class_counts = Counter(balanced_labels)
        balanced_total = len(balanced_labels)
        
        print("   Balanced distribution:")
        for class_id, count in sorted(balanced_class_counts.items()):
            percentage = (count / balanced_total) * 100
            print(f"     Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        print(f"   Dataset size: {total_samples} -> {balanced_total} (+{balanced_total - total_samples} synthetic)")
        
        return balanced_metadata, balanced_labels
        
    except Exception as e:
        print(f"   Error in SMOTE: {e}")
        print("   Using original unbalanced data")
        return metadata_train, train_labels

def apply_image_augmentation_balancing(train_images, train_labels, balance_strategy='dominant'):
    """
    Apply image augmentation to balance image data toward dominant class size
    
    Args:
        train_images: Image data (paths or arrays)
        train_labels: Corresponding labels
        balance_strategy: 'dominant' uses dominant class size as target
    
    Returns:
        balanced_images, balanced_labels
    """
    print("Applying image augmentation for class balancing...")
    
    class_counts = Counter(train_labels)
    
    # Use dominant class size as target
    dominant_class_size = max(class_counts.values())
    print(f"   Target samples per class: {dominant_class_size} (dominant class size)")
    
    balanced_images = []
    balanced_labels = []
    
    for class_id in sorted(class_counts.keys()):
        class_mask = (train_labels == class_id)
        class_images = train_images[class_mask]
        current_count = len(class_images)
        
        # Add original images
        balanced_images.extend(class_images)
        balanced_labels.extend([class_id] * current_count)
        
        # Augment minority classes to reach dominant class size
        if current_count < dominant_class_size:
            needed = dominant_class_size - current_count
            print(f"   Class {class_id}: {current_count} -> {dominant_class_size} (+{needed} augmented)")
            
            # Simple augmentation: duplicate samples
            augment_count = 0
            while augment_count < needed:
                # Randomly select an image from this class
                source_idx = np.random.randint(0, len(class_images))
                source_image = class_images[source_idx]
                
                # For now, just duplicate (in a real implementation, you'd apply transformations)
                augmented_image = source_image
                
                balanced_images.append(augmented_image)
                balanced_labels.append(class_id)
                augment_count += 1
        else:
            print(f"   Class {class_id}: {current_count} samples (dominant class, no augmentation needed)")
    
    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)
    
    print(f"   Image dataset size: {len(train_images)} -> {len(balanced_images)}")
    
    return balanced_images, balanced_labels

def compute_balanced_class_weights(y_train):
    """
    Compute class weights inversely proportional to class frequency
    Most effective for severe imbalance like HAM10000
    """
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_classes, 
        y=y_train
    )
    
    # Convert to dictionary format for Keras
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print("COMPUTED CLASS WEIGHTS:")
    for class_id, weight in class_weight_dict.items():
        count = np.sum(y_train == class_id)
        percentage = (count / len(y_train)) * 100
        print(f"   Class {class_id}: Weight={weight:.2f} (Count={count}, {percentage:.1f}%)")
    
    return class_weight_dict

def focal_loss(alpha=1.0, gamma=2.0):
    """
    Focal Loss: Focuses training on hard examples
    Excellent for class imbalance + hard example mining
    
    alpha: Weighting factor for rare classes (0-1)
    gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def focal_loss_fn(y_true, y_pred):
        # Convert to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        ce_loss = -y_true_one_hot * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * tf.pow(1 - pt, gamma)
        
        focal_loss = focal_weight * ce_loss
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fn

def preprocess_images(paths, target_size=(224, 224)):
    return np.array([load_and_preprocess_image(p, target_size) for p in paths])

def source_inspired_sequential_train(
    train_data, val_data, test_data,
    train_label, val_label, test_label,
    mode='SecureVFL', lr=0.001, hyper_gender=0.005, hyper_age=0.001,
    batch_size=128, epochs=10,
    main_model=None, rep_model=None, gender_model=None, age_model=None,
    gender_cons_adv=None, age_cons_adv=None,
    metrics_callback=None, current_round=0, total_rounds=1, start_time=None,
    accumulated_histories=None
):
    """
    Source-inspired sequential training strategy:
    1. Train main monolithic model first
    2. Train fairness models on fixed embeddings
    3. No complex adversarial training loop
    """
    with_fairness = (mode == 'SecureVFL')
    train_attr = train_data[3] if with_fairness else None

    # Unpack metadata client input (fifth element in train_data/val_data/test_data tuple)
    metadata_train = train_data[4]
    metadata_val = val_data[4]
    metadata_test = test_data[4]

    if main_model is None or rep_model is None:
        result = get_model_variant(train_data, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age, with_fairness=with_fairness)
        main_model, rep_model, gender_model, age_model, image_embedding_model, tabular_embedding_model, fusion_head, gender_cons_adv, age_cons_adv, image_classifier, tabular_classifier = result

    # Ensure all variables are properly assigned for return statement
    if 'image_embedding_model' not in locals():
        print("Creating missing legacy models for compatibility...")
        from model import get_image_embedding_model, get_tabular_embedding_model, get_fusion_head, get_individual_classifiers
        image_embedding_model = get_image_embedding_model()
        tabular_embedding_model = get_tabular_embedding_model(train_data[4].shape[1])
        fusion_head = get_fusion_head(128 + 64, 7)  # Default dimensions
        image_classifier, tabular_classifier = get_individual_classifiers(image_embedding_model, tabular_embedding_model, 7)

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

    # Apply SMOTE balancing to training data
    print("\n" + "="*60)
    print("SMOTE DATA BALANCING")
    print("="*60)
    
    # FIXED: Manual feature normalization for tabular data
    print("Normalizing tabular features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    metadata_train_normalized = scaler.fit_transform(metadata_train)
    metadata_val_normalized = scaler.transform(metadata_val)
    metadata_test_normalized = scaler.transform(metadata_test)
    
    print(f"  Original metadata range: [{metadata_train.min():.3f}, {metadata_train.max():.3f}]")
    print(f"  Normalized metadata range: [{metadata_train_normalized.min():.3f}, {metadata_train_normalized.max():.3f}]")
    
    # Apply SMOTE to balance metadata/tabular data
    balanced_metadata_train, balanced_train_labels = apply_smote_balancing(
        metadata_train_normalized, train_label, balance_strategy='auto'
    )
    
    # FIXED: Use consistent indexing for image and tabular balancing
    print("Creating consistent balanced dataset...")
    
    # Get the indices used by SMOTE
    from collections import Counter
    original_size = len(train_label)
    balanced_size = len(balanced_train_labels)
    
    # Create consistent image paths based on balanced labels
    balanced_train_images_paths = []
    
    # Map each balanced sample to an original image
    class_indices = {}
    for i, label in enumerate(train_label):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    for balanced_label in balanced_train_labels:
        # Randomly pick an original image of the same class
        available_indices = class_indices[balanced_label]
        chosen_idx = np.random.choice(available_indices)
        balanced_train_images_paths.append(train_data[0][chosen_idx])
    
    balanced_train_images_paths = np.array(balanced_train_images_paths)
    
    print(f"  Consistent dataset size: {len(balanced_train_labels)} samples")
    print(f"  Image paths: {len(balanced_train_images_paths)}, Labels: {len(balanced_train_labels)}")
    
    # Show final balanced distribution
    print("\nFINAL BALANCED DATASET ANALYSIS:")
    analyze_class_distribution(balanced_train_labels)
    print("="*60 + "\n")
    
    # FIXED: Properly expand attributes to match balanced dataset size
    if train_data[3] is not None and len(balanced_train_labels) > len(train_data[3]):
        print(f"Expanding attributes from {len(train_data[3])} to {len(balanced_train_labels)} samples...")
        
        # Create mapping from new indices to original indices
        original_size = len(train_label)
        expanded_attrs = []
        
        for i in range(len(balanced_train_labels)):
            # For synthetic samples, use attributes from a randomly selected original sample of the same class
            if i < original_size:
                # Original sample
                expanded_attrs.append(train_data[3][i])
            else:
                # Synthetic sample - find a random original sample of the same class
                target_class = balanced_train_labels[i]
                original_class_indices = np.where(train_label == target_class)[0]
                if len(original_class_indices) > 0:
                    random_idx = np.random.choice(original_class_indices)
                    expanded_attrs.append(train_data[3][random_idx])
                else:
                    # Fallback: use first sample's attributes
                    expanded_attrs.append(train_data[3][0])
        
        balanced_train_attrs = np.array(expanded_attrs)
        print(f"   Expanded attributes: {balanced_train_attrs.shape}")
    else:
        balanced_train_attrs = train_data[3][:len(balanced_train_labels)] if train_data[3] is not None else None

    # SOURCE-INSPIRED SEQUENTIAL TRAINING
    print("="*60)
    print("SOURCE-INSPIRED SEQUENTIAL TRAINING")
    print("="*60)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # STEP 1: Train MAIN MODEL FIRST (like source paper)
        print("Step 1: Training main monolithic model...")
        
        # Prepare balanced training data
        train_images = preprocess_images(balanced_train_images_paths, target_size=(224, 224))
        
        # DEBUG: Check data shapes and ranges
        print(f"  DEBUG - Train images shape: {train_images.shape}, range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"  DEBUG - Metadata shape: {balanced_metadata_train.shape}, range: [{balanced_metadata_train.min():.3f}, {balanced_metadata_train.max():.3f}]")
        print(f"  DEBUG - Labels shape: {balanced_train_labels.shape}, unique: {np.unique(balanced_train_labels)}")
        
        # Create data generator for main model training
        train_gen = DataGenerator(balanced_train_images_paths, balanced_metadata_train, balanced_train_labels, batch_size=batch_size)
        val_gen = DataGenerator(val_data[0], metadata_val_normalized, val_label, batch_size=batch_size)
        
        # DEBUG: Test a single batch
        test_batch = train_gen[0]
        print(f"  DEBUG - Batch input shapes: {[x.shape for x in test_batch[0]]}")
        print(f"  DEBUG - Batch output shape: {test_batch[1].shape}")
        print(f"  DEBUG - Batch label range: [{test_batch[1].min()}, {test_batch[1].max()}]")
        print(f"  DEBUG - Normalized tabular range: [{test_batch[0][1].min():.3f}, {test_batch[0][1].max():.3f}]")
        
        # Train main model (monolithic architecture)
        history = main_model.fit(train_gen, epochs=1, verbose=1, validation_data=val_gen)
        
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_acc = history.history['val_accuracy'][0]
        
        # DEBUG: Check model predictions
        pred_sample = main_model.predict(test_batch[0], verbose=0)
        print(f"  DEBUG - Model predictions shape: {pred_sample.shape}")
        print(f"  DEBUG - Prediction probabilities: {pred_sample[0]}")
        print(f"  DEBUG - Max prediction class: {np.argmax(pred_sample[0])}, True class: {test_batch[1][0]}")
        
        print(f"  Main model: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        # STEP 2: Train fairness models on FIXED embeddings (like source paper)
        if with_fairness and gender_model is not None and age_model is not None:
            print("Step 2: Training fairness models on fixed embeddings...")
            
            # Get FIXED embeddings from the trained main model
            train_embeddings = rep_model.predict([train_images, balanced_metadata_train], batch_size=batch_size, verbose=0)
            
            # Prepare gender and age targets
            gender_targets = balanced_train_attrs[:, 0] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
            age_targets = balanced_train_attrs[:, 1] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
            
            # Train gender model on fixed embeddings
            print("  Training gender fairness model...")
            gender_hist = gender_model.fit(train_embeddings, gender_targets, batch_size=batch_size, epochs=1, verbose=0)
            gender_acc = gender_hist.history['accuracy'][0]
            
            # Train age model on fixed embeddings
            print("  Training age fairness model...")
            age_hist = age_model.fit(train_embeddings, age_targets, batch_size=batch_size, epochs=1, verbose=0)
            age_acc = age_hist.history['accuracy'][0]
            
            print(f"  Fairness models: Gender Acc={gender_acc:.3f}, Age Acc={age_acc:.3f}")
            
            # STEP 3: Adversarial training for SecureVFL
            print("Step 3: Adversarial training to reduce leakage...")
            
            try:
                if gender_cons_adv is not None and age_cons_adv is not None:
                    # ENHANCED: Multiple rounds of stronger adversarial training
                    print("  Training enhanced gradient reversal adversarial networks...")
                    
                    # Get current embeddings
                    current_embeddings = rep_model.predict([train_images, balanced_metadata_train], batch_size=batch_size, verbose=0)
                    
                    # Prepare targets for adversarial training
                    gender_targets = balanced_train_attrs[:, 0] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
                    age_targets = balanced_train_attrs[:, 1] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
                    
                    # ENHANCED: Multiple adversarial training rounds for stronger privacy
                    total_adv_loss = 0
                    for adv_round in range(1):  # Reduced from 3 to 1 round for stability
                        print(f"    Adversarial round {adv_round + 1}/1...")
                        
                        # Train gender adversarial model
                        gender_adv_hist = gender_cons_adv.fit(
                            current_embeddings, 
                            [current_embeddings, gender_targets],  # Both reconstruction and adversarial targets
                            epochs=2,  # Reduced from 5 to 2 epochs for stability
                            batch_size=batch_size,
                            verbose=0,
                            validation_split=0.1
                        )
                        
                        # Train age adversarial model  
                        age_adv_hist = age_cons_adv.fit(
                            current_embeddings,
                            [current_embeddings, age_targets],  # Both reconstruction and adversarial targets
                            epochs=2,  # Reduced from 5 to 2 epochs for stability
                            batch_size=batch_size,
                            verbose=0,
                            validation_split=0.1
                        )
                        
                        # Track adversarial loss (should become more negative = better obfuscation)
                        avg_adv_loss = (gender_adv_hist.history['loss'][-1] + age_adv_hist.history['loss'][-1]) / 2
                        total_adv_loss += avg_adv_loss
                        
                        # Skip embedding updates for stability
                        # (Previous version was causing instability by modifying embeddings)
                    
                    avg_adversarial_loss = total_adv_loss / 1  # Updated denominator
                    
                    # PRINT ADVERSARIAL LOSS - CRITICAL FIX
                    print(f"  Adversarial loss: {avg_adversarial_loss:.4f} (more negative = better privacy)")
                    
                    # ENHANCED: Test privacy protection effectiveness with stronger DP noise
                    print("  Testing privacy protection effectiveness...")
                    
                    # Get original and obfuscated embeddings for comparison
                    original_embeddings = rep_model.predict([train_images[:100], balanced_metadata_train[:100]], verbose=0)
                    
                    # Apply adversarial obfuscation
                    gender_protected, _ = gender_cons_adv.predict(original_embeddings, verbose=0)
                    age_protected, _ = age_cons_adv.predict(original_embeddings, verbose=0)
                    combined_protected = 0.5 * gender_protected + 0.5 * age_protected
                    
                    # Test leakage before and after protection
                    test_attrs = balanced_train_attrs[:100] if balanced_train_attrs is not None else np.random.randint(0, 2, (100, 2))
                    
                    # Original leakage
                    gender_model_temp = get_fairness_model(original_embeddings.shape[1], 2)
                    gender_model_temp.fit(original_embeddings, test_attrs[:, 0], epochs=5, verbose=0)
                    original_gender_acc = gender_model_temp.evaluate(original_embeddings, test_attrs[:, 0], verbose=0)[1]
                    
                    age_model_temp = get_fairness_model(original_embeddings.shape[1], 5) 
                    age_model_temp.fit(original_embeddings, test_attrs[:, 1], epochs=5, verbose=0)
                    original_age_acc = age_model_temp.evaluate(original_embeddings, test_attrs[:, 1], verbose=0)[1]
                    
                    # Protected leakage
                    gender_model_temp.fit(combined_protected, test_attrs[:, 0], epochs=5, verbose=0)
                    protected_gender_acc = gender_model_temp.evaluate(combined_protected, test_attrs[:, 0], verbose=0)[1]
                    
                    age_model_temp.fit(combined_protected, test_attrs[:, 1], epochs=5, verbose=0)
                    protected_age_acc = age_model_temp.evaluate(combined_protected, test_attrs[:, 1], verbose=0)[1]
                    
                    print("  Privacy protection results:")
                    print(f"    Gender: {original_gender_acc:.3f} -> {protected_gender_acc:.3f} (Δ={protected_gender_acc-original_gender_acc:+.3f})")
                    print(f"    Age: {original_age_acc:.3f} -> {protected_age_acc:.3f} (Δ={protected_age_acc-original_age_acc:+.3f})")
                    
                    # Use adversarial-protected embeddings as final result (DP noise was counterproductive)
                    final_gender_acc = protected_gender_acc
                    final_age_acc = protected_age_acc
                    
                    print(f"  Final privacy (adversarial training only):")
                    print(f"    Gender: {final_gender_acc:.3f}, Age: {final_age_acc:.3f}")
                    print(f"  Privacy improvement: Gender {final_gender_acc-original_gender_acc:+.3f}, Age {final_age_acc-original_age_acc:+.3f}")
                    
                    # CRITICAL FIX: Actually modify rep_model to include adversarial protection
                    print("  Applying adversarial protection to main representation model...")
                    
                    # Create a wrapper model that applies adversarial protection
                    from tensorflow.keras.models import Model
                    from tensorflow.keras.layers import Lambda
                    
                    def adversarial_protection_layer(embeddings):
                        """Apply adversarial protection to embeddings"""
                        # Get protected embeddings from both adversarial models
                        gender_protected, _ = gender_cons_adv(embeddings)
                        age_protected, _ = age_cons_adv(embeddings)
                        # Combine protections
                        return 0.5 * gender_protected + 0.5 * age_protected
                    
                    # Get original rep_model inputs and intermediate output
                    rep_inputs = rep_model.input
                    raw_embeddings = rep_model.output
                    
                    # Apply adversarial protection
                    protected_embeddings = Lambda(adversarial_protection_layer, name='adversarial_protection')(raw_embeddings)
                    
                    # Create new protected representation model
                    protected_rep_model = Model(inputs=rep_inputs, outputs=protected_embeddings, name='protected_rep_model')
                    
                    # Replace the original rep_model with protected version
                    rep_model = protected_rep_model
                    print("  ✅ Representation model now includes adversarial protection!")
                    
                    # Set the adversarial loss for tracking
                    adv_loss = avg_adversarial_loss
                    
                else:
                    print("  No adversarial models for VanillaFL mode")
                    adv_loss = 0.0  # No adversarial training in VanillaFL
                    
            except Exception as e:
                print(f"  Error in adversarial training: {e}")
                traceback.print_exc()
                adv_loss = 0.0  # Set default on error
        
        else:
            # VanillaFL mode - only leakage measurement
            print("Step 2: Training fairness models for leakage measurement...")
            
            # Get embeddings for leakage measurement
            train_embeddings = rep_model.predict([train_images, balanced_metadata_train], batch_size=batch_size, verbose=0)
            
            # Prepare gender and age targets
            gender_targets = balanced_train_attrs[:, 0] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
            age_targets = balanced_train_attrs[:, 1] if balanced_train_attrs is not None else np.zeros(len(balanced_train_labels))
            
            if gender_model is not None:
                gender_hist = gender_model.fit(train_embeddings, gender_targets, batch_size=batch_size, epochs=1, verbose=0)
                gender_acc = gender_hist.history['accuracy'][0]
            else:
                gender_acc = 0.0
                
            if age_model is not None:
                age_hist = age_model.fit(train_embeddings, age_targets, batch_size=batch_size, epochs=1, verbose=0)
                age_acc = age_hist.history['accuracy'][0]
            else:
                age_acc = 0.0
            
            adv_loss = 0.0  # No adversarial training in VanillaFL

        # Update histories
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        gender_acc_history.append(gender_acc)
        age_acc_history.append(age_acc)
        adv_loss_history.append(adv_loss)

        # Per-epoch dashboard update
        if metrics_callback is not None:
            # Create current combined histories for dashboard display
            current_train_acc_history = accumulated_train_acc_history + train_acc_history
            current_val_acc_history = accumulated_val_acc_history + val_acc_history
            current_train_loss_history = accumulated_train_loss_history + train_loss_history
            current_gender_acc_history = accumulated_gender_acc_history + gender_acc_history
            current_age_acc_history = accumulated_age_acc_history + age_acc_history
            current_adv_loss_history = accumulated_adv_loss_history + adv_loss_history
            
            # Determine current status based on mode
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

    # Final test accuracy (on original test data)
    test_images = preprocess_images(test_data[0], target_size=(224, 224))
    test_gen = DataGenerator(test_data[0], metadata_test_normalized, test_label, batch_size=batch_size)
    test_loss, test_acc = main_model.evaluate(test_gen, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Return models and histories for dashboard
    return (
        main_model, rep_model, gender_model, age_model,
        image_embedding_model, tabular_embedding_model, fusion_head,  # Legacy models for compatibility
        gender_cons_adv, age_cons_adv,
        image_classifier, tabular_classifier,  # Individual classifiers
        train_acc_history, val_acc_history, train_loss_history, gender_acc_history, age_acc_history, adv_loss_history
    )

def analyze_class_distribution(y_labels, model_accuracy=None):
    """
    Quick analysis of class distribution to detect imbalance issues
    """
    class_counts = Counter(y_labels)
    total_samples = len(y_labels)
    
    print("CLASS DISTRIBUTION ANALYSIS:")
    print(f"   Total samples: {total_samples}")
    
    # Sort by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_id, count in sorted_classes:
        percentage = (count / total_samples) * 100
        print(f"   Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    # Check if model is just predicting dominant class
    if model_accuracy is not None:
        dominant_class_percentage = (sorted_classes[0][1] / total_samples) * 100
        accuracy_diff = abs(dominant_class_percentage - (model_accuracy * 100))
        
        print(f"\nBIAS DETECTION:")
        print(f"   Dominant class: {dominant_class_percentage:.1f}%")
        print(f"   Model accuracy: {model_accuracy * 100:.1f}%")
        print(f"   Difference: {accuracy_diff:.1f}%")
        
        if accuracy_diff < 2:
            print("   WARNING: Model likely biased toward dominant class!")
        else:
            print("   Model learning beyond dominant class")
    
    return class_counts
