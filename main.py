import preprocessing
import data_manager
import rr_interval_irregularity
import bigeminy_suppression
import detector
import metrics
import numpy as np
import os

def main():
    ### Main parameters: ###
    ### N: windows length (must be even)
    ### gamma: threshold for rr-intervals (in seconds)
    ### delta: threshold to distinguish between Bigeminy and AF in ECG
    ### alpha: smoothing coefficient (used in exponential averager)
    ### eta: threshold to effectively check if irregularity is AF or non-AF.

    N = 8
    gamma = 0.03
    delta = 2e-4
    alpha = 0.03
    eta = 0.00035

    if N % 2 != 0:
        raise ValueError("N must be an even-valued integer.")

    ### Load dataset (MIT–BIH Atrial Fibrillation database)

    ecg_records = []
    # As suggested in the paper, do not consider these files during testing and evaluation
    broken_files = ["00735", "03665", "04936", "05091", "08405", "08434"]
    directory = '../afdb'
    for filename in os.listdir(directory):
        record_name = filename.split('.')[0]
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and record_name not in broken_files:
            if filename.endswith('.dat') or filename.endswith('.hea') or filename.endswith('.atr'):
                ecg_records.append(record_name)

    true_labels = []
    predicted_labels = []

    sensitivities = []
    specificities = []
    accuracies = []

    for record_name in ecg_records:
        signal, fs, rr_intervals, ground_truth = data_manager.load_ecg(record_name)

        # if record is shorter than window's length N, skip.
        if len(rr_intervals) < 8:
            continue

        ### dataset preprocessing
        filtered_rr = preprocessing.my_median_filter(rr_intervals)
        ema_rr = preprocessing.exponential_averager(filtered_rr, alpha)
        linear_filtered_rr = preprocessing.my_forward_backward_filtering(ema_rr, alpha)

        ### irregularities in the RR intervals
        m_normalized = rr_interval_irregularity.M_normalized(filtered_rr, N, gamma)
        i_t = rr_interval_irregularity.rr_irregularities(m_normalized, linear_filtered_rr, alpha)

        ### Bigeminy's suppression
        b_n = bigeminy_suppression.bigeminy_irregularity(rr_intervals, filtered_rr, N)
        b_t = bigeminy_suppression.bigeminy_exponential_averager(b_n, alpha)  #smoothed b_n

        ### signal fusion and detection
        decisions = detector.decision_func(i_t, b_t, delta)
        decisions_binary = detector.ground_truth_decision(decisions, eta) #predicted labels

        ### Model's performance
        se, sp, acc = metrics.performance(ground_truth, decisions_binary)
        sensitivities.append(se)
        specificities.append(sp)
        accuracies.append(acc)
        true_labels.append(ground_truth)
        predicted_labels.append(decisions_binary)
        print(f"Record {record_name}:")
        print(f"Specificity: {sp}")
        print(f"Sensitivity: {se}")
        print(f"Accuracy: {acc}")
        print("---------------------")
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)
    print("\n")

    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    avg_accuracy = np.mean(accuracies)
    metrics.plot_roc_auc(true_labels, predicted_labels)

    print(f"Average Sensitivity: {avg_sensitivity}")
    print(f"Average Specificity: {avg_specificity}")
    print(f"Average Accuracy: {avg_accuracy}")


if __name__ == "__main__":
    main()