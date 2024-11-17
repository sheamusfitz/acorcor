import numpy as np

def lr1(sig, n=None):
    """
    Compute the autocorrelation of a signal using a left-right mean approach. Specifically, $$\frac{1}{n}\sum_{i=0}^{n-t} (x_i - \bar{x}_l)(x_{i+t} - \bar{x}_r)$$ where $\bar{x}_l$ goes from 0 to t, and $\bar{x}_r$ goes from n-t to n.

    This is the incorrect method of computing the "left-right" autocorrelation. The correct method is lr2.
    
    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the autocorrelation. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    if n==None:
        n = len(sig)
    acor = np.zeros(len(sig)-1)
    # acor[0] = np.std(sig)
    lmean = sig[0]
    rmean = sig[-1]
    acor[0] = np.mean( (sig[0:n]-lmean)*(sig[0:n]-rmean) )
    for t in range(1, len(acor)-1):
        lmean = np.mean(sig[0:t])
        rmean = np.mean(sig[n-t:n])
        acor[t] = np.mean( (sig[0:n-t]-lmean)*(sig[t:n]-rmean) )
    return acor


def lr2(sig, n=None):
    """
    Compute the autocorrelation of a signal using a left-right mean approach. Specifically, $$\frac{1}{n-t}\sum_{i=0}^{n-t-1} (x_i - \bar{x}_l)(x_{i+t} - \bar{x}_r)$$, where $\bar{x}_l$ goes from 0 to n-t, and $\bar{x}_r$ goes from t to n.

    This is the correct method of computing the "left-right" autocorrelation, but is perhaps slower than the equivalent lr4.
    
    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the autocorrelation. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    if n==None:
        n = len(sig)
    acor = np.zeros(len(sig)-1)
    # acor[0] = np.std(sig)
    lmean = np.mean(sig)
    rmean = np.mean(sig)
    acor[0] = np.mean( (sig[0:n]-lmean)*(sig[0:n]-rmean) )
    for t in range(1, len(acor)-1):
        lmean = np.mean(sig[0:n-t])
        rmean = np.mean(sig[t:n])
        acor[t] = np.mean( (sig[0:n-t]-lmean)*(sig[t:n]-rmean) )
    return acor

def lr3(sig, n=None):
    """
    Compute the autocorrelation of a signal using an expanded left-right mean approach. Here, $$\frac{1}{n-t}\sum_{i=0}^{n-t-1} x_i x_{i+t} - \bar{x}_l \bar{x}_r$$, where $\bar{x}_l$ goes from 0 to n-t, and $\bar{x}_r$ goes from t to n. This ends up being the same as lr2.

    Again, this one is wrong. See lr2 and lr4.
    
    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the autocorrelation. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    if n==None:
        n = len(sig)
    acor = np.zeros(len(sig)-1)
    # acor[0] = np.std(sig)
    lmean = sig[0]
    rmean = sig[-1]
    acor[0] = np.mean( sig[0:n]*sig[0:n] ) - lmean*rmean
    for t in range(1, len(acor)-1):
        lmean = np.mean(sig[0:t])
        rmean = np.mean(sig[n-t:n])
        acor[t] = np.mean( sig[0:n-t]*sig[t:n] ) - lmean*rmean
    return acor

def lr4(sig, n=None):
    """
    Compute the autocorrelation of a signal using an expanded left-right mean approach. Here, $\frac{1}{n-t}\sum_{i=0}^{n-t-1} x_i x_{i+t} - \bar{x}_l \bar{x}_r$, where $\bar{x}_l$ goes from 0 to n-t, and $\bar{x}_r$ goes from t to n.
    
    This is the correct method of computing the "left-right" autocorrelation, and is equivalent to lr2.
    
    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the autocorrelation. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    if n==None:
        n = len(sig)
    acor = np.zeros(len(sig)-1)
    # acor[0] = np.std(sig)
    lmean = np.mean(sig)
    rmean = np.mean(sig)
    acor[0] = np.mean( sig[0:n]*sig[0:n] ) - lmean*rmean
    for t in range(1, len(acor)-1):
        lmean = np.mean(sig[0:n-t])
        rmean = np.mean(sig[t:n])
        acor[t] = np.mean( sig[0:n-t]*sig[t:n] ) - lmean*rmean
    return acor

def lr5(sig, n=None):
    """
    Compute the autocorrelation of a signal using a simplified left-right mean approach. This method computes the autocorrelation as:
    $\frac{1}{n-t}\sum_{i=0}^{n-t-1} x_i x_{i+t} - \bar{x}^2$, where $\bar{x}$ is the mean of the entire signal.
    
    This is actually equivalent to the default method in autocor.py.

    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the autocorrelation. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The autocorrelation of the signal.
    """
    if n==None:
        n = len(sig)
    acor = np.zeros(len(sig)-1)
    # acor[0] = np.std(sig)
    smean = np.mean(sig)**2
    acor[0] = np.mean( sig[0:n]*sig[0:n] ) - smean
    for t in range(1, len(acor)-1):
        # lmean = np.mean(sig[0:t])
        # rmean = np.mean(sig[n-t:n])
        acor[t] = np.mean( sig[0:n-t]*sig[t:n] ) - smean
    return acor

def msd(sig, n=None):
    """
    Compute the autocorrelation function of the signal using the mean squared deviation. $$\frac{1}{n-t}\sum_{i=0}^{n-t-1} (x_i - x_{i+t})^2$$
    
    Args:
        sig (numpy.ndarray): The input signal.
        n (int, optional): The length of the signal to use for the MSD. If not provided, the full length of the signal is used.
    
    Returns:
        numpy.ndarray: The mean squared difference of the signal.
    """
    if n==None:
        n=len(sig)
    out = np.zeros(len(sig))
    for t in range(n):
        out[t] = 1/(n-t) * np.sum((sig[t:] - sig[:n-t])**2)
    return out
