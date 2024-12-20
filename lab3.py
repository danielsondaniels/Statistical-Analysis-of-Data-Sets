import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"\nMetrics:")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    }

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1])  

evaluation_results = evaluate_classification_metrics(y_true, y_pred)
def bayes_theorem(prior_prob, likelihood, marginal_prob):
    posterior_prob = (prior_prob * likelihood) / marginal_prob
    return posterior_prob


prior = 0.6  
likelihood = 0.7  
marginal = 0.5  

posterior = bayes_theorem(prior, likelihood, marginal)
print(f"\nPosterior Probability (Bayes Theorem): {posterior:.4f}")
