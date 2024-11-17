import numpy as np
from scipy import signal

def autocor(sig, biasnorm=False, demean=True):
    """
    Compute the autocorrelation of a signal.
    
    Args:
        sig (numpy.ndarray): The input signal.
        biasnorm (bool, optional): If True, the autocorrelation is normalized by the length of the signal. If False, the autocorrelation is normalized by the decreasing sequence of integers. True refers to the "unbiased" estimator, and False refers to the "biased" estimator. This is not the same "bias" as in the bias-corrected autocorrelation estimator in the acorcor.py module.
        demean (bool, optional): If True, the mean of the signal is subtracted before computing the autocorrelation.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    arr = sig.copy()
    if demean:
        arr -= np.mean(arr)
    if biasnorm:
        return(
            signal.correlate(arr, arr, mode='full')[len(arr)-1:-1] / len(arr)
        )
    else:
        return(
            signal.correlate(arr, arr, mode='full')[len(arr)-1:] /
            np.arange(len(arr)+1, 1, -1)
        )
