import numpy as np
from scipy import signal

def autocor(sig, weirdnorm=False, demean=True):
    arr = sig.copy()
    if demean:
        arr -= np.mean(arr)
    if weirdnorm:
        return(
            signal.correlate(arr, arr, mode='full')[len(arr)-1:-1] / len(arr)
        )
    else:
        return(
            signal.correlate(arr, arr, mode='full')[len(arr)-1:-1]
            / np.arange(len(arr), 1, -1)
        )
