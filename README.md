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

This section gives a guide on how to create new device and signal types.

TODO: Actually we'll want development guides for new evolution algorithms and cost functions also.
    Probably the development guide deserves its own file.

If your type has dependencies external to base Julia,
    put your code into a sub-package instead of following the instructions below.
TODO: I'd better write a different set of instructions for sub-packages...

TODO: Make this neat.
- Implement the concrete type:
  - Follow implementation instructions in `Devices.jl`, `Signals.jl`, or `CostFunctions.jl`.
  - Each type should have its own file (I haven't been following this rule but I should).
  - Drop the file into the appropriate subfolder of `src`.
- Add the type to the main module.
  - Create a new sub-module within `Devices` or `Signals`, and include your file within the sub-module.
  - Import the type into the `Devices` or `Signals` module.
- Add a test set to `test/Devices.jl` or `test/Signals.jl`.
  - Create a model object.
  - Run the standard consistency check.
- Add documentation.
  - Document your type's constructor(s) thoroughly with doc-strings.
  - Add a section to `docs/src/Devices.md` or `docs/src/Signals.md`. Follow the existing patterns.

TODO: Eventually we'll need a guide on how to use git to actually update the repo itself...