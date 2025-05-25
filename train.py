import numpy as np
from tensorflow.keras.utils import to_categorical
from data import DataGenerator, load_and_preprocess_image
from model import get_model_variant


def preprocess_images(paths, target_size=(218, 178)):
    return np.array([load_and_preprocess_image(p, target_size) for p in paths])


def FairVFL_train(train_data, val_data, test_data,
                  train_label, val_label, test_label,
                  mode='FairVFL', lr=0.001, hyper_gender=0.005, hyper_age=0.001,
                  batch_size=128, epochs=10):
    with_fairness = (mode == 'FairVFL')

    train_attr = train_data[2] if with_fairness else None

    model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv = \
        get_model_variant(train_data, lr=lr, hyper_gender=hyper_gender, hyper_age=hyper_age,
                          with_fairness=with_fairness)

    train_gen = DataGenerator(train_data[0], train_data[1], train_label, batch_size=batch_size)
    val_gen = DataGenerator(val_data[0], val_data[1], val_label, batch_size=batch_size)
    test_gen = DataGenerator(test_data[0], test_data[1], test_label, batch_size=batch_size)

    best_val_acc = 0
    best_weights = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=1)
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

        if with_fairness:
            print("Preprocessing train images for fairness models...")
            train_images = preprocess_images(train_data[0])

            print("Generating embeddings...")
            train_embeddings = rep_model.predict([train_images, train_data[1]], batch_size=batch_size)

            # Convert one-hot to class labels
            gender_targets = np.argmax(train_attr[0], axis=1)
            age_targets = np.argmax(train_attr[1], axis=1)

            print("Training gender model...")
            gender_model.fit(train_embeddings, gender_targets, batch_size=batch_size, epochs=1, verbose=1)

            print("Training age model...")
            age_model.fit(train_embeddings, age_targets, batch_size=batch_size, epochs=1, verbose=1)

            print("Training adversarial models...")
            gender_cons_adv.fit(train_embeddings, train_embeddings, batch_size=batch_size, epochs=1, verbose=1)
            age_cons_adv.fit(train_embeddings, train_embeddings, batch_size=batch_size, epochs=1, verbose=1)

    if best_weights is not None:
        model.set_weights(best_weights)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, rep_model, gender_model, age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv
