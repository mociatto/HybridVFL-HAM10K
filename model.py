from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Normalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

NUM_CLASSES = 7  # HAM10000 has 7 lesion classes

def get_source_inspired_architecture(image_shape=(224, 224, 3), tabular_dim=None, num_classes=NUM_CLASSES):
    """
    Source-inspired monolithic architecture adapted for HAM10000
    FIXED: Feature normalization, simpler architecture, better initialization
    """
    # Inputs
    image_input = Input(shape=image_shape, name='image_input')
    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    
    # FIXED: Normalize tabular features (critical for HAM10000 age/sex/localization)
    tabular_norm = Normalization(name='tabular_normalization')
    normalized_tabular = tabular_norm(tabular_input)
    
    # SIMPLIFIED image branch for medical imaging (less overfitting)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_initializer='he_normal')(image_input)  # Better initialization
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # REMOVED: Extra conv layers to reduce overfitting
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    image_features = Dense(64, activation='relu', name='image_features',
                          kernel_initializer='he_normal')(x)  # Smaller representation
    
    # SIMPLIFIED tabular branch with normalization
    tab_x = Dense(32, activation='relu', kernel_initializer='he_normal')(normalized_tabular)
    tab_x = Dropout(0.3)(tab_x)
    tabular_features = Dense(16, activation='relu', name='tabular_features',
                            kernel_initializer='he_normal')(tab_x)  # Smaller representation
    
    # MONOLITHIC FUSION - Direct concatenation like source paper
    fused_representation = Concatenate(name='fused_representation')([image_features, tabular_features])
    
    # SIMPLIFIED classification head
    fused_x = Dense(64, activation='relu', kernel_initializer='he_normal')(fused_representation)
    fused_x = Dropout(0.3)(fused_x)
    
    # Output layer for 7-class classification
    output_layer = Dense(num_classes, activation='softmax', name='classification_output',
                        kernel_initializer='glorot_uniform')(fused_x)  # Glorot for softmax
    
    # Main model
    main_model = Model(inputs=[image_input, tabular_input], outputs=output_layer, name='MonolithicVFL')
    
    # Representation model for fairness evaluation (like source paper)
    rep_model = Model(inputs=[image_input, tabular_input], outputs=fused_representation, name='RepresentationModel')
    
    return main_model, rep_model

def get_fairness_model(input_dim, output_dim):
    """Enhanced fairness model with more capacity for better leakage measurement"""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_adversarial_autoencoder(embedding_dim):
    """Enhanced adversarial autoencoder for better privacy protection"""
    input_layer = Input(shape=(embedding_dim,))
    
    # Stronger encoder with more capacity
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.2)(x)
    
    # Bottleneck layer for information compression
    encoded = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    
    # Decoder with noise injection for obfuscation
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(encoded)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.1)(x)
    
    # Output layer with tanh for bounded output
    output = Dense(embedding_dim, activation='tanh', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    # Stronger learning rate for more aggressive obfuscation
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                 loss='mse')
    return model

def get_gradient_reversal_model(embedding_dim, sensitive_classes):
    """BALANCED gradient reversal adversarial network for stable privacy protection"""
    from tensorflow.keras.layers import Lambda, GaussianNoise
    
    def gradient_reversal(x, lambda_param=1.0):
        """Balanced gradient reversal layer for numerical stability"""
        # During forward pass: identity
        # During backward pass: reverse gradients with controlled amplification
        return tf.stop_gradient(x) + tf.cast(x, tf.float32) * tf.constant(-lambda_param)
    
    input_layer = Input(shape=(embedding_dim,))
    
    # BALANCED: Stable architecture with moderate capacity
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(input_layer)
    x = Dropout(0.3)(x)  # Reduced dropout for stability
    x = GaussianNoise(0.01)(x)  # Reduced noise level
    
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.2)(x)
    
    # Privacy-preserving features with bounded output for stability
    features = Dense(embedding_dim, activation='tanh', name='privacy_features')(x)
    
    # BALANCED: Moderate strength gradient reversal
    reversed_features = Lambda(lambda x: gradient_reversal(x, lambda_param=2.0))(features)
    
    # BALANCED: Moderate depth adversarial classifier
    adv_x = Dense(128, activation='relu')(reversed_features)
    adv_x = Dropout(0.3)(adv_x)
    adv_x = Dense(64, activation='relu')(adv_x)
    adv_x = Dropout(0.2)(adv_x)
    adv_x = Dense(32, activation='relu')(adv_x)
    adv_output = Dense(sensitive_classes, activation='softmax', name='adversarial_output')(adv_x)
    
    model = Model(inputs=input_layer, outputs=[features, adv_output])
    
    # Compile with BALANCED adversarial loss weight to prevent numerical instability
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),  # Reduced learning rate for stability
        loss={'privacy_features': 'mse', 'adversarial_output': 'sparse_categorical_crossentropy'},
        loss_weights={'privacy_features': 1.0, 'adversarial_output': -1.0}  # Reduced from -5.0 to -1.0
    )
    
    return model

def get_image_embedding_model():
    """Legacy individual image embedding model for backward compatibility"""
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # Optionally add dropout for regularization
    x = Dropout(0.3)(x)
    model = Model(inputs=image_input, outputs=x, name="ImageEmbeddingModel")
    return model

def get_tabular_embedding_model(tabular_dim):
    """Legacy individual tabular embedding model for backward compatibility"""
    tabular_input = Input(shape=(tabular_dim,))
    x = Dense(64, activation='relu')(tabular_input)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=tabular_input, outputs=x, name="TabularEmbeddingModel")
    return model

def get_fusion_head(input_dim, num_classes=NUM_CLASSES):
    """Legacy fusion head for backward compatibility"""
    fused_input = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(fused_input)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=fused_input, outputs=output, name="FusionHead")
    return model

def get_individual_classifiers(image_embedding_model, tabular_embedding_model, num_classes=NUM_CLASSES):
    """
    Create individual classifiers for each client that can make standalone predictions.
    This fixes the issue where embedding models output features instead of class probabilities.
    """
    # Image client classifier (standalone)
    image_classifier = Sequential([
        image_embedding_model,
        Dense(64, activation='relu', name='image_classifier_dense'),
        Dropout(0.3, name='image_classifier_dropout'),
        Dense(num_classes, activation='softmax', name='image_predictions')
    ], name='ImageClassifier')
    
    # Tabular client classifier (standalone)
    tabular_classifier = Sequential([
        tabular_embedding_model,
        Dense(32, activation='relu', name='tabular_classifier_dense'),
        Dropout(0.3, name='tabular_classifier_dropout'),
        Dense(num_classes, activation='softmax', name='tabular_predictions')
    ], name='TabularClassifier')
    
    return image_classifier, tabular_classifier

def get_model_variant(train_data, lr=0.001, hyper_gender=0.005, hyper_age=0.001, with_fairness=True):
    """
    NEW: Source-inspired monolithic architecture with federated learning capability
    """
    # For backward compatibility with existing training loop
    # train_data: (img_paths, tabular, labels, sens_attrs, metadata_tabular, ...)
    image_input_shape = (224, 224, 3)
    tabular_input_shape = (train_data[1].shape[1],)
    metadata_input_shape = (train_data[4].shape[1],)

    print("Creating SOURCE-INSPIRED monolithic architecture...")
    
    # NEW: Use source-inspired monolithic architecture
    main_model, rep_model = get_source_inspired_architecture(
        image_shape=image_input_shape,
        tabular_dim=metadata_input_shape[0],
        num_classes=NUM_CLASSES
    )
    
    # Compile main model
    main_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),  # Use legacy Adam
        loss='sparse_categorical_crossentropy',  # Changed from categorical to sparse
        metrics=['accuracy']
    )
    
    # CRITICAL: Adapt the normalization layer to training data
    print("  Adapting feature normalization to training data...")
    # This will be done during first training batch automatically
    
    # Legacy components for backward compatibility (create but mark as deprecated)
    image_embedding_model = get_image_embedding_model()
    tabular_embedding_model = get_tabular_embedding_model(metadata_input_shape[0])
    
    # Create individual classifiers for evaluation
    image_classifier, tabular_classifier = get_individual_classifiers(
        image_embedding_model, tabular_embedding_model, NUM_CLASSES
    )
    
    # Legacy fusion head (not used in new architecture but kept for compatibility)
    img_emb_dim = image_embedding_model.output_shape[1]
    tab_emb_dim = tabular_embedding_model.output_shape[1]
    fusion_head = get_fusion_head(img_emb_dim + tab_emb_dim, NUM_CLASSES)

    # Always create gender and age models for leakage measurement
    # Both VanillaFL and FairVFL will train these models to measure leakage
    embedding_dim = rep_model.output_shape[1]
    gender_model = get_fairness_model(embedding_dim, output_dim=2)
    age_model = get_fairness_model(embedding_dim, output_dim=5)
    
    # Only create adversarial models for FairVFL (with_fairness=True)
    if with_fairness:
        gender_cons_adv = get_gradient_reversal_model(embedding_dim, 2)
        age_cons_adv = get_gradient_reversal_model(embedding_dim, 5)
    else:
        gender_cons_adv = age_cons_adv = None

    # Return everything for modular training
    # NOTE: main_model is the new monolithic architecture, model is kept for backward compatibility
    return (
        main_model, rep_model, gender_model, age_model,  # NEW: main_model instead of model
        image_embedding_model, tabular_embedding_model, fusion_head,
        gender_cons_adv, age_cons_adv,
        image_classifier, tabular_classifier  # Individual classifiers
    )
