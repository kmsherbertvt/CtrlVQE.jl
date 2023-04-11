module Benchmarking

    const Nmn = sort(
        [(m^n, m, n) for m in 2:3 for n in 1:16 if m^n < 2^13],
        by=item->item[1]
    )

    function index(Nmax)
        N = [N for (N,m,n) in Nmn]
        return findlast( N .≤ Nmax)
    end

    # TODO: We don't need "Experiment" in these names...
    module ResourceBenchmarkExperiment
        include("ResourceBenchmarkExperiment.jl")
    end

    module OneQubitTrotterAccuracyExperiment
        include("OneQubitTrotterAccuracyExperiment.jl")
    end

    module TrotterConvergenceExperiment
        include("TrotterConvergenceExperiment.jl")
    end

    module GradientResourceBenchmarkExperiment
        include("GradientResourceBenchmarkExperiment.jl")
        # TODO: Actually please combine resource costs for evolve and gradient into one experiment. All we need to see is that slopes for both match expected scaling.
    end

    module GradientSignalAccuracyExperiment
        include("GradientSignalAccuracyExperiment.jl")
    end

    module GradientAccuracyExperiment
        include("GradientAccuracyExperiment.jl")
    end

end # module Benchmarking
