module Benchmarking
    """
        benchmark(args...; kwargs...)

    Run a specific trial and produce a `BenchmarkTools.Trial` object.

    """
    function benchmark end

    """
        scaling(args...; kwargs...)

    Run a series of benchmarks for different parameters,
        and return arrays reporting best-case resource costs.

    """
    function scaling end

    """
        analysis(args...; outdir=".", kwargs...)

    Perform some standard analysis on the results of `scaling`.

    The signature is identical to `scaling` except for an additional `outdir` kwarg,
        which specifies a directory to save results and figures in.

    This function will load the results of <outdir>/scaling.npz,
        or if that file does not exist, run `scaling` and create the file.
    Note that this ASSUMES that file was generated with the same args and kwargs.

    """
    function analyze end

    export benchmark, scaling

    include("Evolution.jl")     # Implements package functions for time evolution.
    include("Evaluation.jl")    # Implements package functions for cost functions.

    include("Convergence.jl")
    import .Convergence         # Provides a suite of functions for convergence analysis.

end # module Benchmarking
