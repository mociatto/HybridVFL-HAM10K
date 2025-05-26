import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluation(fusion_head, image_embedding_model, tabular_embedding_model, data, batch_size=128):
    # data: (images, features, labels, gender_labels, age_labels, metadata)
    images, features, labels, gender_labels, age_labels, metadata = data
    img_embeds = image_embedding_model.predict(images, batch_size=batch_size)
    tab_embeds = tabular_embedding_model.predict(metadata, batch_size=batch_size)
    fused_embeds = np.concatenate([img_embeds, tab_embeds], axis=1)
    preds = fusion_head.predict(fused_embeds, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='macro')
    print(f"Main Task - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"main_acc": acc, "main_f1": f1}
