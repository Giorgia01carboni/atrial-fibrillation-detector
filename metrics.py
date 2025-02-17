import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


"""
Measure model's performance using Specificity (Sp) and Sensibility (Se)
Sp = TN / (TN + FP)
Se = TP / (TP + FN)
True Positives (TP), False Negatives (FN), True Negatives (TN), False Positives (FP)

Using Specificity and Sensibility we build the ROC curve (how much the sensibility changes 
based on false psoitives ratings. 
AUC = Area Under Curve, area under the ROC curve. If AUC close to 1 => model is very accurate.
"""


def performance(ground_truth, decisions_binary, window_size=8):
    ground_truth = ground_truth[window_size:]

    tn, fp, fn, tp = confusion_matrix(ground_truth, decisions_binary).ravel()

    # Sensitivity
    Se = tp / (tp + fn)
    # Specificity
    Sp = tn / (tn + fp)
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return Se, Sp, accuracy


def plot_roc_auc(ground_truth, decisions):
    min_len = min(len(ground_truth), len(decisions))
    ground_truth = ground_truth[:min_len]
    decisions = decisions[:min_len]

    fpr, tpr, thresholds = roc_curve(ground_truth, decisions)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve for AFIB Detection')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
