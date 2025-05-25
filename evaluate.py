import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluation(model, data, batch_size=128):
    images, features, labels, gender_labels, age_labels = data
    preds = model.predict([images, features], batch_size=batch_size)
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Main Task - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"main_acc": acc, "main_f1": f1}

def fairness_evaluation(rep_model, fairness_model, data, batch_size=128):
    images, features, labels, gender_labels, age_labels = data
    reps = rep_model.predict([images, features], batch_size=batch_size)
    preds = fairness_model.predict(reps, batch_size=batch_size)
    y_true = np.argmax(gender_labels if fairness_model.output_shape[-1] == 2 else age_labels, axis=1)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    label = "Gender" if fairness_model.output_shape[-1] == 2 else "Age"
    print(f"{label} Fairness - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"acc": acc, "f1": f1}