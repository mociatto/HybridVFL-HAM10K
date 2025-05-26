import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"{name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"acc": acc, "f1": f1}

def evaluate_image_only(model, images, labels, batch_size=128):
    preds = model.predict(images, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    return compute_metrics(labels, y_pred, name="Image-only")

def evaluate_tabular_only(model, metadata, labels, batch_size=128):
    preds = model.predict(metadata, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    return compute_metrics(labels, y_pred, name="Tabular-only")

def evaluate_fusion(fusion_head, img_embeds, tab_embeds, labels, batch_size=128):
    fused = np.concatenate([img_embeds, tab_embeds], axis=1)
    preds = fusion_head.predict(fused, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    return compute_metrics(labels, y_pred, name="Fusion")

def evaluation(fusion_head, image_embedding_model, tabular_embedding_model, data, batch_size=128):
    images, features, labels, gender_labels, age_labels, metadata = data
    img_embeds = image_embedding_model.predict(images, batch_size=batch_size)
    tab_embeds = tabular_embedding_model.predict(metadata, batch_size=batch_size)
    return evaluate_fusion(fusion_head, img_embeds, tab_embeds, labels, batch_size)

def fairness_leakage(rep_model, fairness_model, X_img, X_tab, y_sensitive, batch_size=128, name="Fairness"):
    embeddings = rep_model.predict([X_img, X_tab], batch_size=batch_size)
    preds = fairness_model.predict(embeddings, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    return compute_metrics(y_sensitive, y_pred, name=name)
