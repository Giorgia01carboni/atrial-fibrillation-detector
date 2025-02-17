import numpy as np


def decision_func(i_t, b_t, delta):
    """
    Compute the decision output. Distinguish between AF and Bigeminy's.
    The functions returns:
    - i_t[n] if b_t[n] >= delta
    - b_t[n] if b_t[n] < delta.

    :param i_t: Irregularity of RR-intervals
    :param b_t: Irregularity caused by Bigeminy
    :param delta: threshold used to distinguish between AF and Bigeminy
    :return decisions: continuous array containing the i_t or b_t based on threshold.

    """
    if i_t.shape != b_t.shape:
        print("I_t and B_t need to have the same length.")
        return None

    decisions = [i_t[n] if b_t[n] >= delta else b_t[n] for n in range(len(i_t))]

    return decisions


def ground_truth_decision(decisions, eta):
    """
    Converts decisions array containing the detector outputs into a binary form
    (i.e: 1 is AFIB and 0 is non-AF). The output of this function is used to
    check against the dataset annotation to check the model's accuracy.

    :param decisions: Output of detector
    :param eta: Threshold to distinguish between AF and non-AF.
    :return true_decisions: Binary detector's output.

    """
    true_decisions = [1 if d >= eta else 0 for d in decisions]
    return true_decisions