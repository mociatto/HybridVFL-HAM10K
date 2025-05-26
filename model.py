from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 7  # HAM10000 has 7 lesion classes

def get_fairness_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_adversarial_autoencoder(embedding_dim):
    input_layer = Input(shape=(embedding_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(embedding_dim)(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def get_image_embedding_model():
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
    tabular_input = Input(shape=(tabular_dim,))
    x = Dense(64, activation='relu')(tabular_input)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=tabular_input, outputs=x, name="TabularEmbeddingModel")
    return model

def get_fusion_head(input_dim, num_classes=NUM_CLASSES):
    fused_input = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(fused_input)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=fused_input, outputs=output, name="FusionHead")
    return model

def get_model_variant(train_data, lr=0.001, hyper_gender=0.005, hyper_age=0.001, with_fairness=True):
    # For backward compatibility with existing training loop
    # train_data: (img_paths, tabular, labels, sens_attrs, metadata_tabular, ...)
    image_input_shape = (224, 224, 3)
    tabular_input_shape = (train_data[1].shape[1],)
    metadata_input_shape = (train_data[4].shape[1],)

    # Image embedding model (Client 1)
    image_embedding_model = get_image_embedding_model()

    # Tabular (vertical client) embedding model (Client 2)
    tabular_embedding_model = get_tabular_embedding_model(metadata_input_shape[0])

    # Fusion and classifier (Server)
    # Note: Embedding dims must match the output of both models
    img_emb_dim = image_embedding_model.output_shape[1]
    tab_emb_dim = tabular_embedding_model.output_shape[1]
    fusion_input = Input(shape=(img_emb_dim + tab_emb_dim,))

    fusion_head = get_fusion_head(img_emb_dim + tab_emb_dim, NUM_CLASSES)

    # Full pipeline model for end-to-end training (if needed)
    img_input = Input(shape=image_input_shape)
    tabular_input = Input(shape=metadata_input_shape)
    img_emb = image_embedding_model(img_input)
    tab_emb = tabular_embedding_model(tabular_input)
    fused = Concatenate()([img_emb, tab_emb])
    output = fusion_head(fused)
    model = Model(inputs=[img_input, tabular_input], outputs=output, name="FullVFLModel")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # The rep_model is the combined image+tabular embedding extractor, used for fairness heads
    rep_model = Model(inputs=[img_input, tabular_input], outputs=fused, name="RepModel")

    # Fairness/adversarial heads as before, now using the *fused* embeddings
    if with_fairness:
        embedding_dim = rep_model.output_shape[1]
        gender_model = get_fairness_model(embedding_dim, output_dim=2)
        age_model = get_fairness_model(embedding_dim, output_dim=5)
        gender_cons_adv = get_adversarial_autoencoder(embedding_dim)
        age_cons_adv = get_adversarial_autoencoder(embedding_dim)
    else:
        gender_model = age_model = gender_cons_adv = age_cons_adv = None

    # Return everything for modular training
    return (
        model, rep_model, gender_model, age_model,
        image_embedding_model, tabular_embedding_model, fusion_head,
        gender_cons_adv, age_cons_adv
    )
