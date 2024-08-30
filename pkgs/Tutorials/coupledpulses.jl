#=

Using a fixed basis of complex harmonics up to P modes
    (for a total of 2Pn parameters),
    we want to measure the natural pulses and optimize with them.

We need two devices, one which uses a DISJOINT mapper,
    and the other which uses a LinearMapper.

We start by using the DISJOINT mapper
    to measure the quantum Fisher information tensor at x=0.
Then we SVD that and reshape the eigenvectors into the A tensor used in the LinearMapper.

Then we'll optimize the LinearMapper device, and collect the actual pulse parameters.
Then we use the DISJOINT device to re-measure the quantum Fisher information tensor
    at this optimized point.

And then we SVD that and reshape the eigenvectors into another A tensor,
    for which I guess we'll want a third device.
Finally, optimize that too,
    re-measure the quantum Fisher information tensor at this optimized point,
    and shape that into yet another A tensor.
Dunno if we actually need a fourth device but might as well.

Adding in three devices for normal pulses and gradient signals,
    that's a total of ten devices in this script.

=#

import CtrlVQE
import CtrlVQE: Parameters, Devices, Modulars
    #= NOTE: Most of the important functions are exported directly by `CtrlVQE`,
        but a few are localized to these modules. =#

import LinearAlgebra

import NPZ
import LineSearches, Optim
import FiniteDifferences
import Plots

Float = Float64

##########################################################################################
#= System specification. =#

# THE CHEMISTRY
H = NPZ.npzread("$(@__DIR__)/matrix/lih30.npy")     # MOLECULAR HAMILTONIAN
ket_prep = [0,0,1,1]                    # HARTREE FOCK REFERENCE (for this mapping)
HΛ, HU = LinearAlgebra.eigen(H)
E0 = HΛ[1]

# # THE CHEMISTRY (H2 for debugging purposes)
# H = NPZ.npzread("$(@__DIR__)/matrix/H2_sto-3g_singlet_1.5_P-m.npy")     # MOLECULAR HAMILTONIAN
# ket_prep = [1,0]                    # HARTREE FOCK REFERENCE (for this mapping)

# THE PULSE
T = 30.0 # ns                           # PULSE DURATION IN NANOSECONDS
W = 10                                  # NUMBER OF HARMONIC MODES TO INCLUDE IN ANSATZ
basis_prep = CtrlVQE.Bases.DRESSED      # WHICH BASIS IS THE HF STATE PREPPED IN?
basis_meas = CtrlVQE.Bases.DRESSED      # WHICH BASIS DO WE MEASURE IN?
frame_meas = CtrlVQE.Operators.STATIC   # WHAT FRAME DO WE MEASURE IN? (STATIC~"dressed")

# # THE DEVICE (borrowed from a device used by LBNL's AQT lab)
# ω = 2π .* [5.230708, 5.297662, 5.459108, 5.633493] # GHz                # QUBIT FREQUENCY
# δ = 2π .* [0.27366, 0.27332, 0.27070, 0.26745] # GHz                    # ANHARMONICITY
#     # NOTE: δ is not actually used in this simulation, which truncates to two levels.
# g = 2π .* [0.0025, 0.00273, 0.00311] # GHz                              # COUPLING
# quples = [CtrlVQE.Quple(1,2), CtrlVQE.Quple(2,3), CtrlVQE.Quple(3,4)]   # COUPLINGS

# THE DEVICE (the systematic (but degenerate) device we've been mostly using)
ω = 2π .* [4.82, 4.84, 4.86, 4.88] # GHz                                # QUBIT FREQUENCY
δ = 2π .* [0.30, 0.30, 0.30, 0.30] # GHz                                # ANHARMONICITY
    # NOTE: δ is not actually used in this simulation, which truncates to two levels.
g = 2π .* [0.02, 0.02, 0.02] # GHz                                      # COUPLING
quples = [CtrlVQE.Quple(1,2), CtrlVQE.Quple(2,3), CtrlVQE.Quple(3,4)]   # COUPLINGS

# # THE DEVICE (the systematic (but degenerate) device we've been mostly using, for H2)
# ω = 2π .* [4.82, 4.84] # GHz                                # QUBIT FREQUENCY
# δ = 2π .* [0.30, 0.30] # GHz                                # ANHARMONICITY
#     # NOTE: δ is not actually used in this simulation, which truncates to two levels.
# g = 2π .* [0.02] # GHz                                      # COUPLING
# quples = [CtrlVQE.Quple(1,2)]   # COUPLINGS

# THE SIMULATION
r = round(Int, 20T)                     # NUMBER OF TIME STEPS
evolution = CtrlVQE.TOGGLE              # HOW DO WE INTEGRATE TIME? (TOGGLE~"Trotterize")

# THE OPTIMIZATION
linesearch = LineSearches.MoreThuente() # NOTE: I have no idea how to pick this...
f_tol = 0.0                             # Terminate if subsequent energies are this close.
g_tol = 1e-6                            # Terminate if the gradient norm is this small.
maxiter = 1000                          # Give up after this many iterations.

# THE PULSE SELECTION
threshold = 0.05                        # ACCEPTABLE RATIO BETWEEN PULSE EIGENVALUES

##########################################################################################
#= Build out preliminary objects. =#

n = length(ket_prep)
grid = CtrlVQE.TemporalLattice(T,r)     # USED TO CONTROL TIME INTEGRATIONS
protopulse = CtrlVQE.CompositeSignal([  # TEMPLATE FOR THE PULSES ON EACH QUBIT
    CtrlVQE.ConstrainedSignal(CtrlVQE.ComplexHarmonic(Float(0), Float(0), w, T), :T)
        for w in 1:W
])

# Device components
A = Modulars.TruncatedBosonicAlgebra{2,n}       # 2 levels per qubit, n qubits
static = Modulars.TransmonHamiltonian{A}(ω, δ, g, quples)
channels = [                    # NOTE: Pulse parameters live in this object!
    Modulars.QubitChannel{A}(
        q,
        deepcopy(protopulse),
        CtrlVQE.ConstrainedSignal(CtrlVQE.Constant(ω[q]), :A),
    ) for q in 1:n
]

# Energy function components
preparer = Modulars.KetPreparation{A}(ket_prep, basis_prep)
measurer = Modulars.BareMeasurement{A}(H, basis_meas, frame_meas)

# Optimizer
optimizer = Optim.BFGS(linesearch=linesearch)
options = Optim.Options(
    show_trace = true, show_every = 1,  # Monitor progress as optimizations proceed.
    store_trace = true,                 # Lets us lazily fetch the loss curve.
    f_tol=f_tol, g_tol=g_tol, iterations = maxiter,
)

##########################################################################################
#= Toolkit to reconstruct a disjoint device with linearly-coupled parameters. =#

function CoupledDevice(template, basis) # [i←j,k]
    n = length(template.channels)
    K = size(basis, 2)
    basis = reshape(basis, (n,:,K))     # [i,j,k]
    A = permutedims(basis, (2,3,1))     # [j,k,i]

    return Modulars.ModularDevice(
        template.static,
        deepcopy(template.channels),
        Modulars.LinearMapper(A),
        Float,
    )
end

function CoupledEnergy(template, coupleddevice)
    return Modulars.ModularEnergy(
        template.evolution,
        template.grid,
        coupleddevice,
        template.preparer,
        template.measurer,
    )
end

function disjointvalues(device)
    return reduce(vcat, Parameters.values(channel) for channel in device.channels)
end

##########################################################################################
#= Toolkit to build out the various pulse bases. =#

function sortingslice(Λ, ε)
    cΛ = abs.(Λ)
    σ = sortperm(cΛ)
    cΛσ = cΛ[σ]

    imin = length(cΛσ)
    while imin > 1
        cΛσ[imin] * ε > cΛσ[imin-1] && break
        imin -= 1
    end
    return σ[imin:end]
end

function state_evolver(fn)
    x0 = Vector{eltype(fn.device)}(undef, Parameters.count(fn.device))
    workbasis = CtrlVQE.workbasis(fn.evolution)
    ψ0 = Modulars.initialstate(fn.preparer, fn.device, workbasis)
    return (x) -> (
        x0 .= Parameters.values(fn.device);
        Parameters.bind!(fn.device, x);
        ψ = CtrlVQE.evolve(fn.evolution, fn.device, workbasis, fn.grid, ψ0);
        Parameters.bind!(fn.device, x0);
        ψ
    )
end

function calculate_jacobian(fn, x)
    @time J = FiniteDifferences.jacobian(
        FiniteDifferences.central_fdm(5,1),
        state_evolver(fn),
        x,
    )[1]

    ∂ψα = @view(J[1:2:end,:])
    ∂ψβ = @view(J[2:2:end,:])

    return Complex.(∂ψα, ∂ψβ)
end

function fetch_jacobian(fn, x, tag)
    file = "dat/$(basename(@__FILE__)[begin:end-3]).jacobian.$tag.npy"
    isfile(file) && return NPZ.npzread(file)
    ∂ψ = calculate_jacobian(fn, x)
    NPZ.npzwrite(file, ∂ψ)
    return ∂ψ
end

function calculate_quantumgeometrictensor(ψ,∂ψ)
    return ∂ψ'*∂ψ .- (∂ψ'*ψ)*(ψ'*∂ψ)
end

function calculate_hessian(fn, x)
    @time Hs = FiniteDifferences.jacobian(
        FiniteDifferences.central_fdm(5,1),
        CtrlVQE.grad_function(fn),
        x,
    )[1]

    return Hs
end

function fetch_hessian(fn, x, tag)
    file = "dat/$(basename(@__FILE__)[begin:end-3]).hessian.$tag.npy"
    isfile(file) && return NPZ.npzread(file)
    Hs = calculate_hessian(fn, x)
    NPZ.npzwrite(file, Hs)

    # REGULARIZE
    return (Hs .+ Hs') ./ 2
end

function fetch_optim(f, g!, x0, optimizer, options, tag)
    xfile = "dat/$(basename(@__FILE__)[begin:end-3]).xf.$tag.npy"
    Efile = "dat/$(basename(@__FILE__)[begin:end-3]).E.$tag.npy"
    isfile(xfile) && isfile(Efile) && return (
        NPZ.npzread(xfile), NPZ.npzread(Efile), nothing
    )
    optimization = Optim.optimize(f, g!, x0, optimizer, options)
    xf = Optim.minimizer(optimization)
    E = Optim.f_trace(optimization)
    NPZ.npzwrite(xfile, xf)
    NPZ.npzwrite(Efile, E)
    return xf, E, optimization
end

##########################################################################################
#= Do the thing. =#


DEVICE = Modulars.ModularDevice(static, channels, Modulars.DISJOINT, Float)
ENERGY = Modulars.ModularEnergy(evolution, grid, DEVICE, preparer, measurer)

f  = CtrlVQE.cost_function(ENERGY)
g! = CtrlVQE.grad_function_inplace(ENERGY)
x0 = zeros(Float, Parameters.count(DEVICE))

xf, E, optimization = fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
Parameters.bind!(DEVICE, xf)
xfp = Main.disjointvalues(DEVICE)       # Identical to `xf` but included for consistency.

Ef = f(xf)

module Normal
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = deepcopy(Main.x0)

    println("Constructing basis matrix for $(@__MODULE__)...")
    Hs = Main.fetch_hessian(ENERGY, x, @__MODULE__)
    Λ, U = LinearAlgebra.eigen(Hs)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end

module InterNormal
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = Main.Normal.xfp

    println("Constructing basis matrix for $(@__MODULE__)...")
    Hs = Main.fetch_hessian(ENERGY, x, @__MODULE__)
    Λ, U = LinearAlgebra.eigen(Hs)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end

module OptiNormal
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = Main.InterNormal.xfp

    println("Constructing basis matrix for $(@__MODULE__)...")
    Hs = Main.fetch_hessian(ENERGY, x, @__MODULE__)
    Λ, U = LinearAlgebra.eigen(Hs)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end


module Natural
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = deepcopy(Main.x0)

    println("Constructing basis matrix for $(@__MODULE__)...")
    ψ = Main.state_evolver(ENERGY)(x)
    ∂ψ = Main.fetch_jacobian(ENERGY, x, @__MODULE__)
    qgt = Main.calculate_quantumgeometrictensor(ψ,∂ψ)
    qfi = real.(qgt)
    Λ, U = LinearAlgebra.eigen(qfi)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end

module InterNatural
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = Main.Natural.xfp

    println("Constructing basis matrix for $(@__MODULE__)...")
    ψ = Main.state_evolver(ENERGY)(x)
    ∂ψ = Main.fetch_jacobian(ENERGY, x, @__MODULE__)
    qgt = Main.calculate_quantumgeometrictensor(ψ,∂ψ)
    qfi = real.(qgt)
    Λ, U = LinearAlgebra.eigen(qfi)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end

module OptiNatural
    import ..CtrlVQE, ..Parameters, ..Optim, ..LinearAlgebra, ..Main
    import ..optimizer, ..options, ..DEVICE, ..ENERGY

    x = Main.InterNatural.xfp

    println("Constructing basis matrix for $(@__MODULE__)...")
    ψ = Main.state_evolver(ENERGY)(x)
    ∂ψ = Main.fetch_jacobian(ENERGY, x, @__MODULE__)
    qgt = Main.calculate_quantumgeometrictensor(ψ,∂ψ)
    qfi = real.(qgt)
    Λ, U = LinearAlgebra.eigen(qfi)

    σ = Main.sortingslice(Λ, Main.threshold)
    pulsebasis = U[:, σ]
    pulseweights = Λ[σ]
    device = Main.CoupledDevice(DEVICE, pulsebasis)
    energy = Main.CoupledEnergy(ENERGY, device)

    f  = CtrlVQE.cost_function(energy)
    g! = CtrlVQE.grad_function_inplace(energy)
    x0 = zeros(Main.Float, Parameters.count(device))

    xf, E, optimization = Main.fetch_optim(f, g!, x0, optimizer, options, @__MODULE__)
    Parameters.bind!(device, xf)
    xfp = Main.disjointvalues(device)

    Ef = f(xf)
end




##########################################################################################
#= Plot pulse bases. =#
##########################################################################################

function matshow!(plt, A)
    return Plots.heatmap!(plt, A;
        xlims=[0,size(A,2)] .+ 0.5,
        ylims=[0,size(A,1)] .+ 0.5,
        ticks=false,
        aspect_ratio=1,
    )
end

matshow(A) = matshow!(Plots.heatmap(), A)

function plot_basis(MODULE)
    A = abs.(MODULE.pulsebasis)
    plt = matshow(A)
    Plots.heatmap!(plt;
        xticks=[size(A,2)],
        yticks=0:2W:2W*n,
    )

    file = "fig/$(basename(@__FILE__)[1:end-3]).basis.$MODULE.pdf"
    Plots.savefig(plt, file)
end

plot_basis(Normal)
plot_basis(InterNormal)
plot_basis(OptiNormal)
plot_basis(Natural)
plot_basis(InterNatural)
plot_basis(OptiNatural)




function plot_cross(MODULE1, MODULE2)
    A = abs.(MODULE1.pulsebasis' * MODULE2.pulsebasis)
    plt = matshow(A)
    Plots.heatmap!(plt;
        xlabel=MODULE2,
        ylabel=MODULE1,
    )

    file = "fig/$(basename(@__FILE__)[1:end-3]).cross.$(MODULE1)_$(MODULE2).pdf"
    Plots.savefig(plt, file)
end



plot_cross(Normal, InterNormal)
plot_cross(OptiNormal, InterNormal)
plot_cross(Natural, InterNatural)
plot_cross(OptiNatural, InterNatural)


##########################################################################################
#= Plot optimal parameters. =#
##########################################################################################

function plot_weights(MODULE)
    λ = MODULE.pulseweights
    λ ./= maximum(abs.(λ))              # NORMALIZE TO [-1,+1]
    λ .*= maximum(abs.(MODULE.xf))      # MATCH SCALE WITH PARAMETER VALUES

    plt = Plots.plot(;
    )

    Plots.bar!(plt, MODULE.xf; label="Optimized Value")
    Plots.plot!(plt, λ; lw=2, label="Normalized Eigenweight")

    file = "fig/$(basename(@__FILE__)[1:end-3]).weight.$MODULE.pdf"
    Plots.savefig(plt, file)
end

plot_weights(Normal)
plot_weights(InterNormal)
plot_weights(OptiNormal)
plot_weights(Natural)
plot_weights(InterNatural)
plot_weights(OptiNatural)



function plot_distribution!(plt, MODULE)
    x = range(0, 1.0, length(MODULE.xf))
    y = sort(abs.(MODULE.xf); rev=true)
    y ./= sum(y) / length(y)
    Plots.plot!(plt, x, y; lw=2, label=string(MODULE))
end

distributionplot = Plots.plot(;)
plot_distribution!(distributionplot, Normal)
plot_distribution!(distributionplot, InterNormal)
plot_distribution!(distributionplot, OptiNormal)
plot_distribution!(distributionplot, Natural)
plot_distribution!(distributionplot, InterNatural)
plot_distribution!(distributionplot, OptiNatural)
Plots.savefig(distributionplot, "fig/$(basename(@__FILE__)[1:end-3]).distribution.pdf")


##########################################################################################
#= Plot loss curves. =#
##########################################################################################

function plot_loss!(plt, MODULE; kwargs...)
    Plots.plot!(
        plt, MODULE.E .- E0;
        lw=3, α=0.6, label=string(MODULE),
        kwargs...
    )
end

lossplot = Plots.plot(;
    ylims=[1e-17, 1e1],
    yticks=10.0 .^ (-16:2:0),
    yscale=:log10,
    legend=:bottomright,
)
plot_loss!(lossplot, Main; color=:black)
plot_loss!(lossplot, Normal)
plot_loss!(lossplot, InterNormal)
plot_loss!(lossplot, OptiNormal)
plot_loss!(lossplot, Natural)
plot_loss!(lossplot, InterNatural)
plot_loss!(lossplot, OptiNatural)
Plots.savefig(lossplot, "fig/$(basename(@__FILE__)[1:end-3]).loss.pdf")