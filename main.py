import preprocessing
import data_manager
import rr_interval_irregularity
import bigeminy_suppression
import detector
import numpy as np


def save_array_to_txt(filename, array_name, array, num_samples=15):
    with open(filename, "a") as f:  # "a" per aggiungere al file senza sovrascrivere
        f.write(f"{array_name}(first {num_samples} values):\n")
        np.savetxt(f, array[:num_samples], fmt="%.6f")  # Salva con 6 cifre decimali
        f.write("\n")  # Spazio tra i dati


def main():
    ### Main parameters: ###
    ### N: windows length (must be even)
    ### gamma: threshold for rr-intervals (in seconds)
    ### delta: threshold to distinguish between Bigeminy and AF in ECG
    ### alpha: smoothing coefficient (used in exponential averager)

    N = 8
    gamma = 0.03
    delta = 0.05
    alpha = 0.02

    if N % 2 != 0:
        raise ValueError("N must be an even-valued integer.")

    ### Load dataset (MIT–BIH Atrial Fibrillation database)
    ecg_name = "04043"
    signal, fs, rr_intervals = data_manager.load_ecg(ecg_name)
    save_array_to_txt("output.txt", "rr_intervals", rr_intervals)

    ### dataset preprocessing

    ''' 
    First remove ectopic beats with the median filter so that rr_intervals is
    not distorted, then use the exponential averager filter to smooth the signal 
    and to track the trend in the RR interval series. The output of the 
    exponential_averager function has a non-linear phase (signal can be delayed non-linearly).
    Solution: application of forward-backward filtering to achieve linear phase.
    How: Apply exponential averager forward, apply exponential averager on inverted signal,
    reverse signal again to have correct output (linear_filtered_rr). 
    '''

    filtered_rr = preprocessing.my_median_filter(rr_intervals)
    save_array_to_txt("output.txt", "filtered_rr", filtered_rr)

    exponential_avg_rr_forw = preprocessing.exponential_averager(filtered_rr, alpha)
    exponential_avg_rr_back = preprocessing.exponential_averager(exponential_avg_rr_forw[::-1], alpha)
    linear_filtered_rr = exponential_avg_rr_back[::-1]
    save_array_to_txt("output.txt", "linear_filtered_rr", linear_filtered_rr)

    ### irregularities in the RR intervals
    m_normalized = rr_interval_irregularity.M_normalized(filtered_rr, N, gamma)
    i_t = rr_interval_irregularity.rr_irregularities(m_normalized, linear_filtered_rr, alpha)
    save_array_to_txt("output.txt", "m_normalized", m_normalized)
    save_array_to_txt("output.txt", "i_t", i_t)

    ### Bigeminy's suppression
    b_n = bigeminy_suppression.bigeminy_irregularity(rr_intervals, filtered_rr, N)
    b_t = bigeminy_suppression.bigeminy_exponential_averager(b_n, alpha) #smoothed b_n
    save_array_to_txt("output.txt", "b_t", b_t)

    ### signal fusion and detection
    decisions = detector.decision_func(i_t, b_t, delta)
    save_array_to_txt("output.txt", "decisions ", decisions)

    print(decisions)


if __name__ == "__main__":
    main()