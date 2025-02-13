import numpy as np

def M_normalized(rr_intervals, N, gamma):
    '''
    Sliding detection widow of length N computes the number of all pairwise RR interval
    combinations differing more than \gamma seconds. The result is normalized.

    :param rr_intervals: Array of RR-intervals (in seconds)
    :param N: Sliding window's length (on the paper: 8 beats)
    :param gamma: Threshold for detecting irregular intervals (in seconds)
    :return m_normalized_list: List of M(n) values. NB: 0 ≤ M(n) ≤ 1.
    '''

    m_normalized_list = []

    for i in range(len(rr_intervals) - N + 1):
        window = rr_intervals[i:i+N]
        m = 0
        for j in range(len(window)):
            for k in range(j+1, len(window)):
                m += np.heaviside(abs(window[j] - window[k]) - gamma, 0)
        m_normalized = (2 * m) / (N * (N - 1))
        m_normalized_list.append(m_normalized)

    return m_normalized_list

def rr_irregularities(m_normalized_list, ema_rr, alpha):
    '''
    Find indicator of heartbeats irregularities by the ratio between M(n) smoothed (M_t(n))
    and RR interval trend r_t(n). I_t(n) = M_t(n) / r_t(n)
    :param m_normalized_list: List of M(n) values.
    :param ema_rr: array of same length of rr_intervals containing smoothed signal
    :param alpha: smoothing coefficient (0 < alpha < 1). A value closer to 1 gives more weight to the most recent data,
                  while a value closer to 0 gives more weight to past values.
    :return: I_t(n). Output is close to 0 for regular rhythms and approaches 1 during AF.
    '''

    mnt_prev = m_normalized_list[0]
    # Exponential Moving Average of M(n)
    ema_mn = np.zeros(len(m_normalized_list))
    ema_mn[0] = mnt_prev

    for n, val in enumerate(m_normalized_list, start=0):
        ema_mn[n] = mnt_prev + alpha * (val - mnt_prev)
        mnt_prev = ema_mn[n]

    return ema_mn / np.array(ema_rr)