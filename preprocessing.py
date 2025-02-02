import numpy as np
from scipy.ndimage import median_filter

def median_filter(rr_intervals):
    """
    3-point median filter r_m(n) = median{r(n-1),r(n),r(n+1)}
    :return: rr intervals without ectopic beats
    """

    filtered_rr = median_filter(rr_intervals, size=3)
    return filtered_rr


def exponential_averager(rr_intervals, alpha):
    """
    :param rr_intervals: array of RR intervals in seconds
    :param alpha: smoothing coefficient (0 < alpha < 1). A value closer to 1 gives more weight to the most recent data,
                  while a value closer to 0 gives more weight to past values.
    :return: ema_rr: array of same length of rr_intervals containing smoothed signal

    On main call it on rr_intervals and on rr_intervals[::-1] in order to perform a forward–backward filtering
     to achieve linear (null) phase.
    """

    rt_previous = rr_intervals[0]
    ema_rr = np.zeros(len(rr_intervals))
    ema_rr[0] = rt_previous

    for n, val in enumerate(rr_intervals, start=1):
        ema_rr[n] = rt_previous + alpha * (val - rt_previous)
        rt_previous = ema_rr[n]

    return ema_rr