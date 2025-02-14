import numpy as np

def decision_func(i_t, b_t, delta):
    """

    :param i_t:
    :param b_t:
    :param delta:
    :return:
    """
    if i_t.shape != b_t.shape:
        print("I_t and B_t need to have the same length.")

    decisions = [i_t[n] if b_t[n] >= delta else b_t[n] for n in range(len(i_t))]
    return decisions

