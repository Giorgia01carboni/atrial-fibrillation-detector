import numpy as np

def bigeminy_irregularity(rr_intervals, filtered_rr, N):
    """
    B(n) measures RR irregularity while being indifferent to Bigeminy.
    Slide a window of length N (for both the original RR intervals array and the filtered RR interval array
    and sum the values inside the window. Divide
    :param rr_intervals: Array of RR-intervals (in seconds).
    :param filtered_rr: rr intervals without ectopic beat (check preprocessing.py)
    :param N: Window length.
    :return bigeminy_list: list of B(n) for each window.
    """
    bigeminy_list = []

    for n in range(len(rr_intervals) - N + 1):
        rr_window = rr_intervals[n:n + N]
        filtered_rr_window = filtered_rr[n:n+N]

        rr_sum = sum(rr_window)
        filtered_rr_sum = sum(filtered_rr_window)
        b_n = ((filtered_rr_sum / rr_sum) - 1) ** 2
        bigeminy_list.append(b_n)
    return bigeminy_list


def bigeminy_exponential_averager(bigeminy_list, alpha):

    """
    Exponential Moving Average (ema) applied to bigeminy suppression list (B(n)).
    :param bigeminy_list: bigeminy suppression
    :param alpha: smoothing coefficient (0 < alpha < 1). A value closer to 1 gives more weight to the most recent data,
                  while a value closer to 0 gives more weight to past values.
    :return: smoothed bigeminy. B_t(n) increases if there are irregularities.
    """

    b_n_prev = bigeminy_list[0]
    ema_b_n = np.zeros(len(bigeminy_list)) #exponential moving average on b_n
    ema_b_n[0] = b_n_prev

    for n, val in enumerate(bigeminy_list[1:], start=1):
        ema_b_n[n] = b_n_prev + alpha * (val - b_n_prev)
        b_n_prev = ema_b_n[n]

    return ema_b_n
