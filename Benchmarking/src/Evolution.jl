
import .Benchmarking

import CtrlVQE
import CtrlVQE: LAT
import CtrlVQE: Integrations, Devices, Evolutions

import BenchmarkTools: @benchmark
import LinearRegression: linregress, slope, bias
import NPZ
import Plots

"""
    benchmark(evolution::EvolutionType, device::DeviceType, grid::IntegrationType)

Compute the resource costs for a single time evolution, of the |0⟩ state.

The parameters are plugged directly into `CtrlVQE.evolve!`,
    along with the |0⟩ state.

# Returns
- A `BenchmarkTools.Trial` object.

"""
function Benchmarking.benchmark(
    evolution::Evolutions.EvolutionType,
    device::Devices.DeviceType{F},
    grid::Integrations.IntegrationType{F},
) where {F}
    # PRELIMINARY ALLOCATIONS
    ψ0 = LAT.basisvector(Devices.nstates(device), 1)
    ψ = similar(ψ0, Complex{F})
    # WARM-UP RUN
    Evolutions.evolve(evolution, device, grid, ψ0; result=ψ)
    # EXPERIMENT
    return @benchmark Evolutions.evolve($evolution, $device, $grid, $ψ0; result=$ψ)
end

"""
    scaling(evolution, devicetype, gridtype; maxk=12, maxn=10, T=1.0)

Benchmark an evolution algorithm for increasing time steps and qubit count.

# Parameters
- `evolution::EvolutionType`: the evolution algorithm to benchmark.
- `devicetype::Type{<:DeviceType}`: a concrete type for constructing the device.
    The `Prototype` constructor will be used to generate devices with specific `n`.
- `gridtype::Type{<:IntegrationType}`: a concrete type for constructing the time grid.
    The `Prototype` constructor will be used to generate grids with specific `r`.

# Keyword Arguments
- `maxk`: this function performs a `benchmark` trial with ``r=T⋅2^k`` time steps,
    where `k` counts up from 0 to `maxk`.
- `maxn`: this function performs a `benchmark` trial with `n` qubits,
    where `n` counts up from 1 to `maxn`.
- `T`: maximum pulse duration.
    Should be chosen to match the pulses programmed into `device`.

Note that `k` controls a per-unit-time quantity of steps,
    so if `T` is large, this function can take awhile!

# Returns
- `time::Matrix{Float64}` (ns)
- `gctime::Matrix{Float64}` (ns)
- `memory::Matrix{Int}` (bytes)
- `allocs::Matrix{Int}`

Each element in these arrays gives the MINIMUM trial
    for the corresponding property of a `BenchmarkTools.Trial` object
    resulting from `benchmark` with a different choice of `r` and `n`.
Row index `i` corresponds to ``k=j-1``, ``r=T⋅2^k`` time steps.
Column index `i` corresponds to ``n=i`` qubits.

"""
function Benchmarking.scaling(
    evolution::Evolutions.EvolutionType,
    devicetype::Type{<:Devices.DeviceType{F}},
    gridtype::Type{<:Integrations.IntegrationType{F}};
    maxk::Int=12,
    maxn::Int=10,
    T::F=1.0,
) where {F}
    # CONSTRUCT METRIC MATRICES
    time   = zeros(Float64, (1+maxk, maxn)) # ns
    gctime = zeros(Float64, (1+maxk, maxn)) # ns
    memory = zeros(Int,     (1+maxk, maxn)) # bytes
    allocs = zeros(Int,     (1+maxk, maxn)) # (just a number)

    # RUN THE EXPERIMENTS
    for n in 1:maxn
        device = Devices.Prototype(devicetype; n=n, T=T)
        for k in 0:maxk
            r = round(Int, T * 2^k)
            grid = Integrations.Prototype(gridtype; r=r, T=T)
            println("Benchmarking n=$n, r=$r...")
            trial = benchmark(evolution, device, grid)

            mintrial = minimum(trial)
            time[1+k,n]   = mintrial.time
            gctime[1+k,n] = mintrial.gctime
            memory[1+k,n] = mintrial.memory
            allocs[1+k,n] = mintrial.allocs
        end
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
    evolution::Evolutions.EvolutionType,
    devicetype::Type{<:Devices.DeviceType{F}},
    gridtype::Type{<:Integrations.IntegrationType{F}};
    outdir::String=".",
    maxk::Int=12,
    maxn::Int=10,
    T::F=1.0,
) where {F}
    # OBTAIN SCALING RESULTS
    scalingfile = "$outdir/scaling.npz"
    if isfile(scalingfile)
        scaling = NamedTuple(Symbol(K) => v for (K,v) in NPZ.npzread(scalingfile))
    else
        scaling = Benchmarking.scaling(evolution,
                devicetype, gridtype; maxk=maxk, maxn=maxn, T=T)
        NPZ.npzwrite(scalingfile; scaling...)
    end

    # COMPUTE SLOPE AND POWER FOR EACH METRIC
    #=

    We want to fit (for each choice of n):

        ``y = A r^p``

    We will do so by taking a log and a linear regression:

        ``log y = p log r + log A``

    =#
    r = [round(Int, T*2^k) for k in 0:maxk]
    regressions = NamedTuple(K => [
        linregress(r, log2.(dat[:,n])) for n in 1:maxn
    ] for (K,dat) in pairs(scaling))
    p = NamedTuple(K => [
        slope(regression[n])[] for n in 1:maxn
    ] for (K,regression) in pairs(regressions))
    A = NamedTuple(K => [
        2^bias(regression[n]) for n in 1:maxn
    ] for (K,regression) in pairs(regressions))

    # SAVE RESULTS
    NPZ.npzwrite("$outdir/p.npz"; p...)
    NPZ.npzwrite("$outdir/A.npz"; A...)

    # PLOT RESULTS
    N = [Devices.nstates(CtrlVQE.Prototype(devicetype; n=n)) for n in 1:maxn]

    plt = Plots.plot(;
        xlabel = "Size of Hilbert Space",
        ylabel = "Power (p in y=A⋅xᵖ)",
        framestyle = :origin,
        palette = :roma10,
        legend = :topright,
    )
    Plots.plot!(plt, N, p.time;   color=1, lw=3, shape=:circle, label="Time")
    Plots.plot!(plt, N, p.gctime; color=3, lw=3, shape=:circle, label="GC Time")
    Plots.plot!(plt, N, p.memory; color=5, lw=3, shape=:square, label="Memory")
    Plots.plot!(plt, N, p.allocs; color=7, lw=3, shape=:star,   label="Allocations")
    Plots.savefig(plt, "$outdir/p.pdf")

    plt = Plots.plot(;
        xlabel = "Size of Hilbert Space",
        ylabel = "Time (ns) per Time Step",
        framestyle = :origin,
        palette = :roma10,
        legend = false,
    )
    Plots.plot!(plt, N, A.time;   color=1, lw=3, shape=:circle, label=false)
    Plots.savefig(plt, "$outdir/time.pdf")

    plt = Plots.plot(;
        xlabel = "Size of Hilbert Space",
        ylabel = "GC Time (ns) per Time Step",
        framestyle = :origin,
        palette = :roma10,
        legend = false,
    )
    Plots.plot!(plt, N, A.gctime; color=3, lw=3, shape=:circle, label=false)
    Plots.savefig(plt, "$outdir/gctime.pdf")

    plt = Plots.plot(;
        xlabel = "Size of Hilbert Space",
        ylabel = "Memory (bytes) per Time Step",
        framestyle = :origin,
        palette = :roma10,
        legend = false,
    )
    Plots.plot!(plt, N, A.memory; color=5, lw=3, shape=:square, label=false)
    Plots.savefig(plt, "$outdir/memory.pdf")

    plt = Plots.plot(;
        xlabel = "Size of Hilbert Space",
        ylabel = "Allocations per Time Step",
        framestyle = :origin,
        palette = :roma10,
        legend = false,
    )
    Plots.plot!(plt, N, A.allocs; color=7, lw=3, shape=:star,   label=false)
    Plots.savefig(plt, "$outdir/allocs.pdf")
end