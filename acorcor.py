"""Module where I'm going to put the atuocorrelation corrected forms."""
import numpy as np
from scipy.special import gamma, gammaincc, expi
from numba import njit



def corrected_exp(t, tau, n, a=1):
    """
    This gives the bias-corrected autocorrelation estimator, for an assumed exponential
    underlying `true' autocorrelation function. This assumes the form: 
        c(t) = a*e^(-t/tau)

        Parameters:
            t (np.array): array of lagtimes of interest
            tau (float): autocorrelation decay time
            n (int): length of the original series used to calculate the autocorrelation  function
            a (float): amplitude of ansatz autocorrelation function

        Returns:
            Bias-corrected autocorrelation estimator asymptotic form, with the same shape 
                as `t`, at lagtimes `t`.
    """
    exp = np.exp(-t/tau)
    out = exp - 1/n - 2*tau/n + 2*tau**2/(n*(n-t)) * (
        t/n + exp - np.exp((t-n)/tau) - t/n * np.exp(-n/tau)
    )

    return a*out


def mygamma(b, x):
    b = np.float64(b)
    # print(type(b), type(x))
    # print(np.max(x))
    return gamma(b)*gammaincc(b, x)


def mygamma2(b, arg):
    b = np.float64(b)
    return(gamma(2*b)*gammaincc(2*b, arg**(1/b)))


def corrected_strex(t, tau, n, b, a=1):
    out = np.zeros_like(t)
    first_ind = 0
    first = np.array([])
    if t[0] == 0:
        first_ind = 1
        first = a*1/n**2 * (n**2 - n + 2*b*tau**2 * gamma(2*b) - 2*tau*n*gamma(1+b) + 2*tau*b*n*mygamma(b, (n/tau)**(1/b)) - 2*tau**2*b*mygamma(2*b, (n/tau)**(1/b)))

    x = t[first_ind:]
    ttaub = (x/tau)**(1/b)
    nttaub = ((n-x)/tau)**(1/b)
    something = a*(np.exp(-ttaub) - 1/n
              + 2*tau*b/n * (mygamma(b, nttaub) - gamma(b))
              + 2*b*x*tau/(n*(n-x)) * (mygamma(b, (n/tau)
                                               ** (1/b)) - mygamma(b, ttaub))
              + 2*b*x*tau**2/(n**2*(n-x)) * (
        gamma(2*b) + n/x*mygamma(2*b, ttaub) - n/x*mygamma(2*b, nttaub)
        - mygamma(2*b, (n/tau)**(1/b))
    ))
    out = np.append(first, something)
    return out

def c_one_over(t, t0, n, a=1):
    return a*(1+t/t0)**-1 - a/n * (1-2*t0) - 2*a/(n**2*(n-t)) * (
        - np.log(t0) * t0 * (n**2 - n*t + t*t0)
        + np.log(n+t0) * (n+t0) * t*t0
        + np.log(n-t+t0) * (n-t+t0) * n*t0
        - np.log(t+t0) * (t+t0) * n*t0
    )


def cstrex_int(n, a, b, tau, c, indices=None):
    """
    Gives the integral of the bias-corrected stretched exponential.
    a * gammastar(b, (t/tau)^(1/b)) + c
    **NOTE** if the integral you're fitting is subsampled, then the `indices' argument is a little tricky... Think about it for yourself, I haven't done that yet.
    """
    b = np.float64(b)
    # print(80)
    t = np.arange(1, n)
    # print(82)
    t1 = np.exp(-(t/tau)**(1/b))
    # print(84)
    t2 = -1/n 
    # print(86)
    t3 = 2/n*tau*b *(mygamma(b, (n-t)/tau) - gamma(b))
    # print(88)
    t4 = 2*b*t*tau/(n*(n-t)) * (mygamma(b, n/tau) - mygamma(b,t/tau))
    # print(90)
    t5 = 2*b*t*tau**2/(n**2*(n-t)) * (
        gamma(2*b)
        + n/t*mygamma2(b, t/tau)
        - n/t*mygamma2(b, (n-t)/tau)
        - mygamma2(b, n/tau)
    )
    # print("n, a, b, tau", n, a, b, tau,'\t'*10, 97)
    out = (t1+t2+t3+t4+t5)
    out = np.append(0, out)
    if type(indices)==type(None):
        return np.cumsum(a*out) + c
    return subcumsum(a*out, indices) + c


def subcumsum(x, indices):
    out = np.zeros_like(indices, float)
    prepended = False
    if indices[0] != 0:
        indices = np.append(0, indices)
        prepended = True
    previ = 0
    prevsum = 0
    for i, index in enumerate(indices):
        thissum = np.sum(x[previ:index])
        out[i] = prevsum + thissum
        previ = i
        prevsum = thissum
    if prepended:
        return out[1:]
    return out

def cexp_integral(t, tau, n, a=1):
    
    out = -t/n + tau * (1 - np.exp(-t/tau)) - 2*t*tau/n +\
        2/n**2 * (-1+np.exp(-n/tau)) * t * tau**2
    out += 2/n*tau**2*np.exp(-n/tau) * (expi(n/tau)-expi((n-t)/tau))
    out += 2/n*tau**2 * (expi((t-n)/tau) - expi(-n/tau))
    out -= 2*(1-np.exp(-n/tau)) * tau**2 / n *np.log((n-t)/n)
    return a*out
