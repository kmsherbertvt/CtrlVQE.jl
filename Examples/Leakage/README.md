# The Rotating Wave Approximation

How susceptible are typical pulses to leakage?
To what extent is the two-level approximation "safe"?
Can we do better using smooth windows?

This directory has three scripts:
1. `optimize.jl` optimizes a typical device, with the two-level approximation,
    to find parameters for a 7-window 21ns pulse preparing something close to the ground state of LiH.
    We do two optimizations, one with fixed windows and with smooth windows.
2. `evolve.jl` runs each of the two pulses with increasing number of levels per transmon.
3. `analysis.jl` plots energies and leakages as a function of m.

We find that, while smooth pulses are not *quite* as egregious as stepped pulses,
    neither pulse in the two-level approximation produces a normalized energy
    anywhere *near* a more exact treatment.
With the device and pulse parameters used here,
    one would need to account for at least four (prefereably 5) levels.