import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import filtfilt

''' 
    First remove ectopic beats with the median filter so that rr_intervals is
    not distorted, then use the exponential averager filter to smooth the signal 
    and to track the trend in the RR interval series. The output of the 
    exponential_averager function has a non-linear phase (signal can be delayed non-linearly).
    Solution: application of forward-backward filtering to achieve linear phase.
    How: Apply exponential averager forward, apply exponential averager on inverted signal,
    reverse signal again to have correct output. Use scipy.signal function filtfilt(). 
'''


def my_median_filter(rr_intervals):
    """
    3-point median filter r_m(n) = median{r(n-1),r(n),r(n+1)}
    :return: rr intervals without ectopic beats
    """

    filtered_rr = median_filter(rr_intervals, size=3)
    return filtered_rr


def exponential_averager(rr_intervals, alpha):
    """
    Smooth signal and track the trend in the RR-intervals using en exponential moving average (ema):
    r_t(n) = r_t(n-1) + alpha * (r(n) - r_t(n-1))

    :param rr_intervals: array of RR intervals in seconds
    :param alpha: smoothing coefficient (0 < alpha < 1). A value closer to 1 gives more weight to the most recent data,
                  while a value closer to 0 gives more weight to past values.
    :return: ema_rr: array of same length of rr_intervals containing smoothed signal

    """

    ema_rr = np.zeros(len(rr_intervals))
    ema_rr[0] = rr_intervals[0]

    for n in range(1, len(rr_intervals)):
        ema_rr[n] = ema_rr[n-1] + alpha * (rr_intervals[n] - ema_rr[n-1])
    return ema_rr


def my_forward_backward_filtering(rr_intervals_filtered, alpha):
    """
    Forward-backward filter application to achieve linear phase.
    The filtfilt function is used, so the exponential moving average formula
    r_t(n) = r_t(n-1) + alpha * (r(n) - r_t(n-1))
    is rewritten as a first order IIR filter (standard form y(n) - a1 * y(n-1) = b0 * x(n)):
    r_t(n) - (1 - alpha) * r_t(n-1) = alpha * r(n).
    The parameters are:
    - a: coefficients used to weight y(n),
    - b: coefficients used to weight x(n).

    :param rr_intervals_filtered: result of pre-filtering on rr_intervals
    :param alpha: smoothing coefficient (0 < alpha < 1). A value closer to 1 gives more weight to the most recent data,
                  while a value closer to 0 gives more weight to past values.
    :return filtered_ema: rr intervals filtered with zero-phase.
    """
    if len(rr_intervals_filtered) > 6:
        b = [alpha]
        a = [1, -(1-alpha)]

        filtered_ema = filtfilt(b, a, rr_intervals_filtered)
        return filtered_ema
    else:
        print("rr_interval is too short to apply forward-backward filtering. Min length required: 6.")
        print("Returning ema_rr")
        return rr_intervals_filtered