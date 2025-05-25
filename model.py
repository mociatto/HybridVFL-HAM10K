from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def get_fairness_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_adversarial_autoencoder(embedding_dim):
    input_layer = Input(shape=(embedding_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(embedding_dim)(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def get_model_variant(train_data, lr=0.001, hyper_gender=0.005, hyper_age=0.001, with_fairness=True):
    model, rep_model, gender_model, age_model = None, None, None, None
    gender_mapper, age_mapper = None, None
    gender_cons_adv, age_cons_adv = None, None

    # Representation Network
    image_input = Input(shape=(218, 178, 3))
    feature_input = Input(shape=(train_data[1].shape[1],))

    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    raw_data_rep = Concatenate()([x, feature_input])  # [image features + tabular]

    rep_model = Model([image_input, feature_input], raw_data_rep)
    output_layer = Dense(2, activation='softmax')(raw_data_rep)
    model = Model([image_input, feature_input], output_layer)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if with_fairness:
        embedding_dim = rep_model.output_shape[1]

        gender_model = get_fairness_model(embedding_dim)
        age_model = get_fairness_model(embedding_dim)

        gender_cons_adv = get_adversarial_autoencoder(embedding_dim)
        age_cons_adv = get_adversarial_autoencoder(embedding_dim)

    return model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv
