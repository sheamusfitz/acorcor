This has a few functions loaded in that are useful for the bias-corrected autocorrelation function fitting.

Most of them take in an array of lagtimes (usually $t$), timescale, other parameters for the function, along with the total trajectory length ($n$).

I have the functions for an exponential $`a\cdot e^{-t/tau}`$, stretched exponential $`a\cdot e^{-(t/tau)^{1/b}}`$, and this thing: $a/(1+t/t0)$.

I can add more upon request.

Also included is a jupyter notebook which shows a recommended fitting method (using `lmfit` with `emcee`).
