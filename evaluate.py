import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluation(model, data, batch_size=128):
    images, features, labels, gender_labels, age_labels = data
    preds = model.predict([images, features], batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='macro')
    print(f"Main Task - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"main_acc": acc, "main_f1": f1}

def fairness_evaluation(rep_model, fairness_model, data, batch_size=128):
    images, features, labels, gender_labels, age_labels = data
    embeddings = rep_model.predict([images, features], batch_size=batch_size)
    preds = fairness_model.predict(embeddings, batch_size=batch_size)

    # Determine which label to evaluate
    output_dim = fairness_model.output_shape[-1]
    if output_dim == 2:
        targets = gender_labels
        label_name = "Gender"
    else:
        targets = age_labels
        label_name = "Age"

    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(targets, y_pred)
    f1 = f1_score(targets, y_pred, average='macro')

    print(f"{label_name} Fairness - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"acc": acc, "f1": f1}
