import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"{name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"acc": acc, "f1": f1}

def compute_detailed_metrics(y_true, y_pred, name="", class_names=None):
    """
    Compute detailed metrics including per-class performance for debugging class imbalance
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n📊 {name} DETAILED METRICS:")
    print(f"   Overall Accuracy: {acc:.4f} ({acc:.1%})")
    print(f"   Macro F1: {f1_macro:.4f}")
    print(f"   Weighted F1: {f1_weighted:.4f}")
    
    # Per-class breakdown
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(7)]
    
    print(f"   Per-class Performance:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = report[str(i)]['support']
            print(f"     {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={support})")
    
    return {"acc": acc, "f1": f1_macro, "f1_weighted": f1_weighted}

def evaluate_image_only(image_classifier, images, labels, batch_size=128, detailed=False):
    """
    FIXED: Now uses image_classifier (which outputs probabilities) instead of embedding model
    """
    if image_classifier is None:
        print("Image classifier not available")
        return {"acc": 0.0, "f1": 0.0}
    
    try:
        preds = image_classifier.predict(images, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        if detailed:
            class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
            return compute_detailed_metrics(labels, y_pred, name="Image-only", class_names=class_names)
        else:
            return compute_metrics(labels, y_pred, name="Image-only")
            
    except Exception as e:
        print(f"Error in image evaluation: {e}")
        return {"acc": 0.0, "f1": 0.0}

def evaluate_tabular_only(tabular_classifier, metadata, labels, batch_size=128, detailed=False):
    """
    FIXED: Now uses tabular_classifier (which outputs probabilities) instead of embedding model
    """
    if tabular_classifier is None:
        print("Tabular classifier not available")
        return {"acc": 0.0, "f1": 0.0}
    
    try:
        preds = tabular_classifier.predict(metadata, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        if detailed:
            class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
            return compute_detailed_metrics(labels, y_pred, name="Tabular-only", class_names=class_names)
        else:
            return compute_metrics(labels, y_pred, name="Tabular-only")
            
    except Exception as e:
        print(f"Error in tabular evaluation: {e}")
        return {"acc": 0.0, "f1": 0.0}

def evaluate_fusion(fusion_head, img_embeds, tab_embeds, labels, batch_size=128, detailed=False):
    """
    Fusion evaluation (unchanged but added detailed option)
    """
    try:
        fused = np.concatenate([img_embeds, tab_embeds], axis=1)
        preds = fusion_head.predict(fused, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        if detailed:
            class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
            return compute_detailed_metrics(labels, y_pred, name="Fusion", class_names=class_names)
        else:
            return compute_metrics(labels, y_pred, name="Fusion")
            
    except Exception as e:
        print(f"Error in fusion evaluation: {e}")
        return {"acc": 0.0, "f1": 0.0}

def evaluation(fusion_head, image_embedding_model, tabular_embedding_model, data, batch_size=128):
    """
    Standard evaluation function (unchanged)
    """
    images, features, labels, gender_labels, age_labels, metadata = data
    img_embeds = image_embedding_model.predict(images, batch_size=batch_size, verbose=0)
    tab_embeds = tabular_embedding_model.predict(metadata, batch_size=batch_size, verbose=0)
    return evaluate_fusion(fusion_head, img_embeds, tab_embeds, labels, batch_size)

def fairness_leakage(rep_model, fairness_model, X_img, X_tab, y_sensitive, batch_size=128, name="Fairness"):
    """
    Fairness leakage evaluation (unchanged)
    """
    embeddings = rep_model.predict([X_img, X_tab], batch_size=batch_size, verbose=0)
    preds = fairness_model.predict(embeddings, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    return compute_metrics(y_sensitive, y_pred, name=name)
