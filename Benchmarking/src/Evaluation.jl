import .Benchmarking

import CtrlVQE: CostFunctions

import BenchmarkTools: @benchmark
import LinearRegression: linregress, slope, bias
import NPZ

"""
    benchmark(costfn, x, niters; nf=1, ng=0)

Compute the resource costs to evaluate a cost function some number of times.

The keyword arguments allow you to easily model different sorts of optimizations.

# Parameters
- `costfn::CostFunctionType`: the cost function to benchmark.
- `niters::Int`: number of times to evaluate the cost function in a single benchmark.
    If you use the keyword arguments,
        this becomes the number of optimization iterations to emulate.

# Keyword Arguments
- `nf`: the number of function evaluations per iteration.
- `ng`: the number of gradient evaluations per iteration.

For example, to emulate first-order SPSA, use `nf=2, ng=0`.
To emulate `BFGS` with a minimalist linesearch algorithm, use `nf=1, ng=1`.

# Returns
- A `BenchmarkTools.Trial` object.

"""
function Benchmarking.benchmark(
    costfn::CostFunctions.CostFunctionType{F},
    niters::Int;
    nf::Int=1,
    ng::Int=0,
) where {F}
    # PRELIMINARY ALLOCATIONS
    x  = zeros(F, length(costfn))
    f  = CostFunctions.cost_function(costfn)
    g! = CostFunctions.grad!function(costfn)
    ∇f = similar(x)
    # WARM-UP RUN
    f(x); g!(∇f, x)
    # EXPERIMENT
    return @benchmark for _ in 1:$niters
        for _ in 1:$nf; $f($x); end
        for _ in 1:$ng; $g!($∇f, $x); end
    end
end


"""
    scaling(costfn; maxk=12, nf=1, ng=0)

Benchmark a specific cost-function for increasing iteration count.

# Parameters
- `costfn::CostFunctionType`: the cost function to benchmark.

# Keyword Arguments
- `maxk`: this function performs a `benchmark` trial with ``2^k`` iterations,
    where `k` counts up from 0 to `maxk`.
- `nf`: the number of function evaluations per iteration.
- `ng`: the number of gradient evaluations per iteration.

# Returns
- `time::Vector{Float64}` (ns)
- `gctime::Vector{Float64}` (ns)
- `memory::Vector{Int}` (bytes)
- `allocs::Vector{Int}`

Each element in these arrays gives the MINIMUM trial
    for the corresponding property of a `BenchmarkTools.Trial` object
    resulting from `benchmark` with a different choice of `niter`.
Index `i` corresponds to ``k=j-1``, ``2^k`` iterations.

"""
function Benchmarking.scaling(
    costfn::CostFunctions.CostFunctionType{F};
    maxk::Int=12,
    nf::Int=1,
    ng::Int=0,
) where {F}
    # CONSTRUCT METRIC MATRICES
    time   = zeros(Float64, 1+maxk) # ns
    gctime = zeros(Float64, 1+maxk) # ns
    memory = zeros(Int,     1+maxk) # bytes
    allocs = zeros(Int,     1+maxk) # (just a number)

    # RUN THE EXPERIMENTS
    for k in 0:maxk
        println("Benchmarking niters=$(1<<k)")
        trial = Benchmarking.benchmark(costfn, 1<<k; nf=nf, ng=ng)

        mintrial = minimum(trial)
        time[1+k]   = mintrial.time
        gctime[1+k] = mintrial.gctime
        memory[1+k] = mintrial.memory
        allocs[1+k] = mintrial.allocs
    end

    # RETURN MATRICES
    return (
        time   = time,
        gctime = gctime,
        memory = memory,
        allocs = allocs,
    )
end






function Benchmarking.analyze(
    costfn::CostFunctions.CostFunctionType{F};
    outdir::String=".",
    maxk::Int=12,
    nf::Int=1,
    ng::Int=0,
) where {F}
    # OBTAIN SCALING RESULTS
    scalingfile = "$outdir/scaling.npz"
    if isfile(scalingfile)
        scaling = NamedTuple(Symbol(K) => v for (K,v) in NPZ.npzread(scalingfile))
    else
        scaling = Benchmarking.scaling(costfn; maxk=maxk, nf=nf, ng=ng)
        NPZ.npzwrite(scalingfile; scaling...)
    end

    # COMPUTE SLOPE AND POWER FOR EACH METRIC
    #=

    We want to fit:

        ``y = A x^p``

    We will do so by taking a log and a linear regression:

        ``log y = p log x + log A``

    =#
    X = 0:maxk
    regressions = NamedTuple(K => linregress(X, log2.(dat)) for (K,dat) in pairs(scaling))
    p = NamedTuple(K => slope(regression)[] for (K,regression) in pairs(regressions))
    A = NamedTuple(K => 2^bias(regression) for (K,regression) in pairs(regressions))

    # SAVE RESULTS
    NPZ.npzwrite("$outdir/p.npz"; p...)
    NPZ.npzwrite("$outdir/A.npz"; A...)

    # REPORT ON RESULTS
    open("$outdir/report.txt", "a") do io
        println(io, """
        Cost-function scaling:
            y = A xᵖ

        x = # of iterations (nf=$nf, ng=$ng)
        y = time (ns):
            p = $(p.time)
            A = $(A.time)
        y = gctime (ns):
            p = $(p.gctime)
            A = $(A.gctime)
        y = memory (bytes):
            p = $(p.memory)
            A = $(A.memory)
        y = allocs:
            p = $(p.allocs)
            A = $(A.allocs)

        """)
    end
end