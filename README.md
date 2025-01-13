# CtrlVQE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kmsherbertvt.github.io/CtrlVQE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kmsherbertvt.github.io/CtrlVQE.jl/dev/)
[![Build Status](https://github.com/kmsherbertvt/CtrlVQE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kmsherbertvt/CtrlVQE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kmsherbertvt/CtrlVQE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kmsherbertvt/CtrlVQE.jl)


Code to simulate ctrl-VQE in Julia.

## Installation

To use this code in your Julia environment,
    simply add this Github repo from the Julia REPL.
Enter package mode by typing `]`, then:

```
pkg> add https://github.com/kmsherbertvt/CtrlVQE.jl
```

To use your own version of the code, first clone the repo into your Julia dev folder
  (mine is `~/.julia/dev`; yours may be different):

```
> cd ~/.julia/dev
> git clone https://github.com/kmsherbertvt/CtrlVQE.jl CtrlVQE
```

Then add the cloned directory to your Julia environment in development mode.
Start the Julia REPL, enter package mode by typing `]`, then:

```
pkg> dev CtrlVQE
```

Note that this works from any working directory,
    since we've used Julia's `dev` directory.


## Sub-Packages

The base package contains most functionality you'll need,
    but the repository itself includes a few extra sub-packages in the `pkgs` directory.
Each one has its own README.md.

To use a sub-package in your Julia environment,
    simply add this Github repo from the Julia REPL.
I haven't figured out how to do it from package mode, though,
    so you'll need to use the `Pkg` module:

```
julia> import Pkg
julia> Pkg.add(url="https://github.com/kmsherbertvt/CtrlVQE.jl", subdir="pkgs/{thesubpackage}")
```

If you are using your own local version of the repository:

```
julia> import Pkg
julia> Pkg.develop(url="$(ENV["HOME"])/.julia/dev/CtrlVQE", subdir="pkgs/{thesubpackage}")
```

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Things that must be done
- A `validate` method defined for each core type, plus doctests in each basic struct.
  - Go ahead and add in Test, FiniteDifferences as dependencies...
  - Emulate hierarchical testsets as done in ADAPT.jl
  - Copy in (and touch up) standard tests from CtrlVQE.jl
  - Write doctests

- A sub-package with convenient tools for plotting?
  - I think this only makes sense as an extension to the ModularFramework.
  - Maybe it can be included in `ModularFramework` itself, if we learn how to use the plot templates package.

- Update citations:
  - Ayush's paper is published
  - My complex ctrl-VQE paper is almost published...

### Examples to Write
- FourierAnalysis
  - Demonstrate that, in the old device parameters, gradient signals tend to peak at the detuning.
  - This shows that, so long as you let the phase vary, the optimizer will be perfectly happy to stay on resonance.
  - I have some hints that cross-resonance frequencies will show up in the new device paraemters.
    We should try and show that with the same framework.
- Adaptive Superposition/Subdivision/Concatentation
  - Mostly done but I'm not happy with how the code is organized. Too complicated for an Example!
  - I'm also not very happy with how concatenation is done. I think it wants a mutable IntegrationType.
  - These Examples may need to wait until after all three are done in a new
- GlobalOptimization
  - Using a sub-eMET pulse duration (say, 18ns on old device)
  - Do a local optimization from zero pulse
  - Do a global optimization from zero pulse
  - Do a shotgun survey of local optimizations from random initial parameters
- SPSAOptimizer
  - Demonstrate how to use my SPSAOptimizers.jl package
  - Compare trajectories for a few different hyperparameters, and compare against BFGS

### EvolutionKit subpackage
The idea is to have a weightier package that gives you convenient access to all sorts of time evolutions.
Well, mainly the idea is to have a separate package that gives you access to Julia's insanely weighty `DifferentialEquations` solver...
  - ODEEvolutions
  - LanczosEvolutions

### SignalKit subpackage
The idea is to have a weightier package that gives you convenient access to every sort of signal we've ever thought of.
  - So many. :)

### Other things to try out and maybe promote to basics
- `ConstrainedEnergyFunction`
  - penalties do not merely artificially inflate the cost function, but instead add their own Lagrange multipliers as parameters.
  - The idea is that the optimization's zero-gradient convergence criterion enforces the penalties being identically zero
    (or at least, small enough to dodge the threshold).
  - I did try this once and found it really didn't behave any differently, so it wasn't worth the degree it complicated the code.
    Even so, it seems theoretically more "right", so if we can get a good interface to do it, let's do it...
  - I anticipate a problem where, realistically, I would want an L2 norm on the gradient of the energy parameters,
    but L∞ norm on the gradient for each Lagrange multiplier.
    Maybe it can be paritally accounted for by normalizing each Lagrange multiplier with the number of parameters in the energy function.
