# Autocorrelation Correction Module

This module provides implementations of bias-corrected autocorrelation estimators for various functional forms.

## Features

- Corrected autocorrelation estimators that account for finite-size effects
- Support for multiple functional forms:
  - Exponential: `a*exp(-t/tau)`
  - Stretched exponential: `a*exp(-(t/tau)^(1/b))`
  - Power law decay: `a/(1+t/t0)`

## Core Functions

### corrected_exp(t, tau, n, a=1)

Computes the bias-corrected autocorrelation estimator for an exponential decay.
- `t`: Array of lag times
- `tau`: Autocorrelation decay time
- `n`: Length of original time series
- `a`: Amplitude (default=1)

### corrected_strex(t, tau, b, n, a=1)

Computes the bias-corrected autocorrelation estimator for a stretched exponential decay.
- `t`: Array of lag times
- `tau`: Autocorrelation decay time. This is **not** the `mean relaxation time' (that is $\\tau \Gamma(b+1)$).
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

