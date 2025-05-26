import numpy as np
from data import DataGenerator, load_and_preprocess_image
from model import get_model_variant
import tensorflow as tf

def preprocess_images(paths, target_size=(224, 224)):
    return np.array([load_and_preprocess_image(p, target_size) for p in paths])

def FairVFL_train(
    train_data, val_data, test_data,
    train_label, val_label, test_label,
    mode='FairVFL', lr=0.001, hyper_gender=0.005, hyper_age=0.001,
    batch_size=128, epochs=10,
    model=None, rep_model=None, gender_model=None, age_model=None,
    gender_cons_adv=None, age_cons_adv=None
):
    with_fairness = (mode == 'FairVFL')
    train_attr = train_data[3] if with_fairness else None

    if model is None or rep_model is None or \
       (with_fairness and (gender_model is None or age_model is None or gender_cons_adv is None or age_cons_adv is None)):
        model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv = \
            get_model_variant(train_data, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age, with_fairness=with_fairness)
    else:
        gender_mapper = age_mapper = None

    train_gen = DataGenerator(train_data[0], train_data[1], train_label, batch_size=batch_size)
    val_gen = DataGenerator(val_data[0], val_data[1], val_label, batch_size=batch_size)
    test_gen = DataGenerator(test_data[0], test_data[1], test_label, batch_size=batch_size)

    best_val_acc = 0
    best_weights = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # 1. Train Main Classifier
        model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=1)
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"Validation accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

        if with_fairness:
            print("Preprocessing train images for fairness models...")
            train_images = preprocess_images(train_data[0], target_size=(224, 224))
            print("Generating embeddings...")
            train_embeddings = rep_model.predict([train_images, train_data[1]], batch_size=batch_size)
            gender_targets = train_attr[:, 0]
            age_targets = train_attr[:, 1]

            # 2. Train fairness (sensitive attribute) classifiers
            print("Training gender model...")
            gender_model.fit(train_embeddings, gender_targets, batch_size=batch_size, epochs=1, verbose=1)
            print("Training age model...")
            age_model.fit(train_embeddings, age_targets, batch_size=batch_size, epochs=1, verbose=1)

            # 3. Train adversarial mappers/consistency heads (reconstruct embedding)
            print("Training adversarial models...")
            gender_cons_adv.fit(train_embeddings, train_embeddings, batch_size=batch_size, epochs=1, verbose=1)
            age_cons_adv.fit(train_embeddings, train_embeddings, batch_size=batch_size, epochs=1, verbose=1)

            # 4. Adversarial update (representation confusion: maximize sensitive loss)
            # This part requires custom training using TensorFlow GradientTape
            # Update rep_model to maximize adversarial loss
            print("Adversarial representation confusion step...")
            with tf.GradientTape() as tape:
                tape.watch(rep_model.trainable_variables)
                rep_outputs = rep_model([train_images, train_data[1]], training=True)
                gender_preds = gender_model(rep_outputs, training=False)
                age_preds = age_model(rep_outputs, training=False)
                gender_adv_loss = tf.keras.losses.sparse_categorical_crossentropy(gender_targets, gender_preds)
                age_adv_loss = tf.keras.losses.sparse_categorical_crossentropy(age_targets, age_preds)
                adv_loss = hyper_gender * tf.reduce_mean(gender_adv_loss) + hyper_age * tf.reduce_mean(age_adv_loss)
                # maximize the adversarial loss: so minimize -adv_loss
                total_loss = -adv_loss

            grads = tape.gradient(total_loss, rep_model.trainable_variables)
            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(grads, rep_model.trainable_variables))

    if best_weights is not None:
        model.set_weights(best_weights)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv
