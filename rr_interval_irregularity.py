import numpy as np

def rr_sliding_window(rr_intervals, N, gamma):
    '''
    Make a window sliding on the rr intervals array and look for irregular intervals
    by checking against a chosen value gamma (in seconds).
    :param rr_intervals:
    :param N: 8-beat sliding window length
    :param gamma: value used to look for irregular intervals (in seconds)
    :return greater_than_gamma: number of intervals exceeding gamma
    '''
    for i in range(len(rr_intervals) - N + 1):
        window = rr_intervals[i: i + N]
        greater_than_gamma = sum(1 for j in range(window) for k in range(j+1, window) if abs[j - k] > gamma)

    return greater_than_gamma