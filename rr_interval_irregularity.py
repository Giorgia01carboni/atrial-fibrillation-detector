import numpy as np

def rr_sliding_window(rr_intervals, N, gamma):

    '''
    Creates a window sliding on the rr intervals array and look for irregular intervals
    by checking against a chosen value gamma (in seconds).
    :param rr_intervals: array of RR-intervals
    :param N: Sliding window's length (on the paper: 8 beats)
    :param gamma: Threshold for detecting irregular intervals (in seconds)
    :return irregular_pairs: number of intervals exceeding gamma (for each window)
    '''

    irregular_pairs = []

    for i in range(len(rr_intervals) - N + 1):
        window = rr_intervals[i: i + N]
        irregular_pair = sum(1 for j in range(len(window)) for k in range(j+1, len(window)) if abs(window[j] - window[k]) > gamma)
        irregular_pairs.append(irregular_pair)

    return irregular_pairs