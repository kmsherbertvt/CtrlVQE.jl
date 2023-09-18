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

## Getting Started

Find well-documented tutorial scripts in `pkgs/Tutorials`.
Use them as a guide for writing your own scripts.
Alternatively, see that sub-package's README for more details on how to run the scripts.

If you are using your own local installation of the code,
    run tests from the Julia REPL, in package mode.
```
pkg> test CtrlVQE
```

You may also wish to run some benchmarks in `pkgs/Benchmarking`.
See that sub-package's README for more details on how to run the scripts.

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Development Guide

This repository is designed to be as modular and extensible as possible.
Think of the ctrl-VQE algorithm as divided into four distinct parts:
- `signals` - how to characterize the time-dependent control signals
- `devices` - how to characterize the dynamics of a computational state, ie. the physical Hamiltonian
- `evols` - how to implement time evolution under a given device Hamiltonian
- `costfns` - how to put the above three parts together into a function to variationally minimize

The `src` code has sub-directories for each of these four parts,
    which consists of a "root" file (eg. `signals` has `Signals.jl`)
    defining an abstract type (eg. `SignalType`)
  and its interface (eg. methods like `valueat`, `partial`, etc.),
    and a set of "leaf" files which define different concrete structs implementing the interface
    (eg. `ConstantSignals.jl` defines `Constant` and `ComplexConstant` signals).
To define a new pulse shape, or device, or evolution algorithm, or measurement protocol,
    you *should* only need to add in the relevant "leaf" file,
    and then make a few formulaic edits so that the package structure "knows about" your addition.

If your type has dependencies external to base Julia
    (eg. `LinearAlgebra` is in base Julia but `FiniteDifferences` requires an extra installation),
    put your code into a sub-package instead of following the instructions below.
That is more complicated and I haven't bothered to write a guide for it yet.
TODO: Write the guide for this. (I don't actually know yet how testing should work...)

Here's the complete checklist:
- Implement the concrete type:
  - Follow the "Implementation" documentation for the abstract type
    (ie. `SignalType`, `DeviceType`, `EvolutionType`, or `CostFunctionType`).
  - Write your implementation in its own "leaf" file.
  - Drop the leaf file into the appropriate subfolder of `src`.
- Add the type to the main module.
  - Create a new sub-module within `CtrlVQE`, and include your file within the sub-module.
  - Optionally, import the new concrete type into the `CtrlVQE` module so it is easily accessed.
- Add a test set for your concrete type. `test/Devices.jl` or `test/Signals.jl`.
  - Find the appropriate test file (eg. if you've implemented a new signal, find `test/Signals.jl`).
  - Add a new test-set for your type. You can follow the existing patterns.
  - Create a model object.
  - Run a sanity check: `StandardTests.validate(my_model_object)`.
    This will run a set of standardized consistency tests,
    eg. is the analytical gradient consistent with a finite difference?
  - Optionally, add additional tests specific to your particular type.
- Add documentation.
  - Document your type's constructor(s) thoroughly with doc-strings.
  - Find the appropriate doc file (eg. if you've implemented a new signal, find `doc/src/Signals.md`).
  - Add a new section for your type. You can follow the existing patterns.
  - Create an autodoc for the module you added to `CtrlVQE`.
    Find the syntax from the existing sections in the doc file.

TODO: Eventually we'll need a guide on how to use git to actually update the repo itself...