import CtrlVQE
import NPZ
import Optim
import Plots
import LinearAlgebra

pkg_dir = "pkgs/Tutorial"

# READ IN MATRIX
matrixname = "H2_sto-3g_singlet_1.5_P-m"
Hmol = NPZ.npzread("$pkg_dir/$matrixname.npy")

# IDENTIFY NUMBER OF QUBITS AND REFERENCE STATE
n = CtrlVQE.QubitOperators.nqubits(Hmol)
ψ_HF = CtrlVQE.QubitOperators.reference(Hmol)

# EXTRACT ANALYTICAL RESULTS
E_HF = real(ψ_HF'*Hmol*ψ_HF)
Λmol, Umol = LinearAlgebra.eigen(LinearAlgebra.Hermitian(Hmol))
E_FCI = Λmol[1]
ψ_FCI = Umol[:,1]
E_CORR = E_HF - E_FCI

# SELECT PULSE TEMPLATE
W = 50
T = 12.0
pulse = CtrlVQE.WindowedSquarePulse(T,W)

# CONSTRUCT DEVICE
m = 3
device = CtrlVQE.SystematicTransmonDevice(m,n,pulse)
x̄i = identity.(CtrlVQE.Parameters.values(device))
L = CtrlVQE.Parameters.count(device)

# PROJECT PROBLEM HAMILTONIAN ONTO TRANSMON SPACE
O0 = CtrlVQE.QubitOperators.project(Hmol, device)
ψ0 = CtrlVQE.QubitOperators.project(ψ_HF, device)

# CONSTRUCT TIME GRID
r = 1000
τ, τ̄, t̄ = CtrlVQE.Evolutions.trapezoidaltimegrid(T,r)

# CONSTRUCT COST FUNCTIONS
import CtrlVQE.Operators: IDENTITY, UNCOUPLED, STATIC
import CtrlVQE.CostFunctions: ProjectedEnergy, Normalization, BareEnergy, NormalizedEnergy
frame = UNCOUPLED
fE, gE = NormalizedEnergy.functions(O0, ψ0, T, device, r; frame=frame)
fF, gF = Normalization.functions(ψ0, T, device, r)

# VISUALIZE ENERGY AND LEAKAGE THROUGHOUT INITIAL EVOLUTION
algorithm = CtrlVQE.Evolutions.Rotate(r)
Ēi, F̄i = zeros(r+1), zeros(r+1)
callback = (i, t, ψ) -> (
    Ēi[i] = CtrlVQE.CostFunctions.evaluate(fE, ψ, t);
    F̄i[i] = CtrlVQE.CostFunctions.evaluate(fF, ψ, t);
)
ψi = CtrlVQE.evolve(algorithm, device, T, ψ0; callback=callback)

initial_plot = Plots.plot(
    title = "Initial Pulse Parameters",
    xlabel= "Time (ns)",
    ylabel= "Energy (Ha)",
    ylims = [E_FCI, E_FCI+4E_CORR],
    legend= :topleft,
)
Plots.plot!(initial_plot, t̄, Ēi, label="Projected")
Plots.plot!(initial_plot, t̄, Ēi ./ F̄i, label="Normalized")
Plots.plot!(Plots.twinx(initial_plot),
    t̄, 1 .- F̄i,
    legend=:topright, ylabel="Leakage", ylims=[0,1],
    label="Leakage", color=4,
)

# USE HARD BOUNDS FOR AMPLITUDES
ΩMAX = 2π * 0.02 # GHz
δΩ = 2π * 0.01 # GHz
λ̄Ω = vcat(ones(L-n), zeros(n))   # DO NOT APPLY PENALTY ON LAST n PARAMETERS (ν̄)
x̄L = fill(-ΩMAX, L)
x̄R = fill(+ΩMAX, L)
σ̄Ω = fill(δΩ, L)
import CtrlVQE.CostFunctions: HardBounds
fΩ, gΩ = HardBounds.functions(λ̄Ω, x̄L, x̄R, σ̄Ω)

# USE SOFT BOUNDS FOR FREQUENCIES
δΔ = 2π * 1.0 # GHz
λ̄ν = vcat(zeros(L-n), ones(n))   # DO NOT APPLY PENALTY ON FIRST L-n PARAMETERS (Ω̄)
x̄0 = vcat(zeros(L-n), device.ω̄)
σ̄ν = fill(δΔ, L)
import CtrlVQE.CostFunctions: SoftBounds
fν, gν = SoftBounds.functions(λ̄ν, x̄0, σ̄ν)

# CONSTRUCT FULL LOSS FUNCTION
import CtrlVQE.CostFunctions: CompositeCostFunction, CompositeGradientFunction
f = CompositeCostFunction(fE, fΩ, fν)
g = CompositeGradientFunction(gE, gΩ, gν)

# RUN OPTIMIZATION
optimizer = Optim.LBFGS()
# TODO: Play with different linesearches, cause the default sux.
# TODO: Also play with different ν0
options = Optim.Options(
    show_trace=true,
    show_every=1,
    iterations=20,
    store_trace=true,
)
#= TODO (hi): CompositeFunction stores last-evaluated f/|g|,
    so that callback can push each onto array. No need for store_trace I think.
=#
optimization = Optim.optimize(f, g, x̄i, optimizer, options)



# VISUALIZE ENERGY AND LEAKAGE THROUGHOUT FINAL EVOLUTION
x̄f = Optim.minimizer(optimization)
CtrlVQE.Parameters.bind(device, x̄f)

Ēf, F̄f = zeros(r+1), zeros(r+1)
callback = (i, t, ψ) -> (
    Ēf[i] = CtrlVQE.CostFunctions.evaluate(fE, ψ, t);
    F̄f[i] = CtrlVQE.CostFunctions.evaluate(fF, ψ, t);
)
ψf = CtrlVQE.evolve(algorithm, device, T, ψ0; callback=callback)

final_plot = Plots.plot(
    title = "Final Pulse Parameters",
    xlabel= "Time (ns)",
    ylabel= "Energy (Ha)",
    ylims = [E_FCI, E_FCI+4E_CORR],
    legend= :topleft,
)
Plots.plot!(final_plot, t̄, Ēf, label="Projected")
Plots.plot!(final_plot, t̄, Ēf ./ F̄f, label="Normalized")
Plots.plot!(Plots.twinx(final_plot),
    t̄, 1 .- F̄f,
    legend=:topright, ylabel="Leakage", ylims=[0,1],
    label="Leakage", color=4,
)

# VISUALIZE PATH OF OPTIMIZER
Ē_optim = Optim.f_trace(optimization)
ḡ_optim = Optim.g_norm_trace(optimization)
optim_plot = Plots.plot(
    title = "Optimizer Path",
    xlabel= "Iteration",
    ylabel= "",
    yscale= :log,
    ylims = [1e-16, 1e0],
    yticks= 10.0 .^ (-16:2:0),
    legend= :bottomleft,
)

Plots.plot!(optim_plot, (Ē_optim .- E_FCI)./E_CORR, label="Energy Error (normalized)")
Plots.plot!(optim_plot, ḡ_optim, label="Gradient Norm")

# TODO (hi): save plots to pdf