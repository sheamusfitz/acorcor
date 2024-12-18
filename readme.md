# Autocorrelation Correction Module

This module provides implementations of bias-corrected autocorrelation estimators for various functional forms.

## Features

- Corrected autocorrelation estimators that account for finite-size effects
- Support for multiple functional forms:
  - Exponential: $`a\cdot\exp(-t/\tau)`$
  - Stretched exponential: $`a\cdot\exp(-(t/\tau)^{1/b})`$
  - Power law decay: $`\frac{a}{1+t/t0}`$

## Core Functions

### corrected_exp(t, tau, n, a=1)

Computes the bias-corrected autocorrelation estimator for an exponential decay.
- `t`: Array of lag times
- `tau`: Autocorrelation decay time
- `n`: Length of original time series
- `a`: Amplitude (default=1)

The full form of this function is:

$$
\left\langle c_t \right\rangle/a = 
        e^{-t/\tau} - \frac1n-\frac{2\tau}{n} + \frac{2\tau^2}{n(n-t)}\left(
            1+e^{-t/\tau}-e^{(t-n)/\tau}-\frac tn e^{-n/\tau}
    \right)
$$

### corrected_strex(t, tau, b, n, a=1)

Computes the bias-corrected autocorrelation estimator for a stretched exponential decay.
- `t`: Array of lag times
- `tau`: Autocorrelation decay time. This is **not** the 'mean relaxation time' (that is $`\tau \cdot \Gamma(b+1)`$).
- `b`: Stretching exponent
- `n`: Length of original time series
- `a`: Amplitude (default=1)

### corrected_powerlaw(t, t0, n, a=1)

Computes the bias-corrected autocorrelation estimator for a power law decay.
- `t`: Array of lag times
- `t0`: Power law decay time
- `n`: Length of original time series
- `a`: Amplitude (default=1)



## Usage

The module is designed to work with numerical analysis packages like NumPy and can be integrated with fitting libraries like `lmfit` and `emcee` for parameter estimation.

Example:
```python
import numpy as np
from acorcor import corrected_exp

# Generate lag times
t = np.random.normal(0, 1, 1000)
# Calculate corrected autocorrelation
corr = corrected_exp(t, tau=10, n=1000, a=1)

```

