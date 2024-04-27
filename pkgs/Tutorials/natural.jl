#= This script is meant to be a self-contained tutorial
    on how to optimize "natural" pulses,
    those eigenpulses of the Jacobian which presumably tell you
    the optimal ways to drive in each direction of Hilbert space.

The script will run through two distinct phases:

Given a pulse parameterization (which the user should designate),
        calculate the Jacobian at the reference state,
        and run an SVD to extract eigenpulses.
    Neglect any pulses with a singular value near zero.
    Optimize an ansatz constructed as a linear combination of natural pulses.

The reference state is most certainly singular,
    meaning there are not going to be enough eigenpulses left
    to parameterize the full Hilbert space.
So the second phase is to repeat the whole process,
    now using the optimized state as the new reference.

The metrics to check at the end are:

1. Energy error.

    Obviously, this checks if the method worked.

2. Overlap of the optimized pulses with each natural pulse from phase two.
    (I.e. the parameters at the end of phase two...)

    Eigenpulses are ordered in descending order of singular value.
    Thus, if we find that our parameters at the end of phase two
        tend to be larger at the beginning of the vector than at the end,
        it suggests a robust method to reduce parameter counts,
        by leaving off the less important eigenpulses a priori.

3. Overlap of the optimized pulses with each natural pulse from phase one.

    This is probably the most important metric,
        as they can be compared directly for different molecules,
        thus revealing precisely those delocalized pulses
        which are most useful for chemistry.
    If we can find a consistent set,
        we have discovered a method to prepare better reference states than HF
        on a given device.

Note that these results will depend on the initial pulse parameterization,
    and on the total pulse duration.
A full-fledged experiment (beyond the scope of this tutorial) would use
    a "fully Trotterized" ansatz,
    a host of equally-sized molecular systems
        (ideally using JW mapping sans tapering,
        with a standard solution to the gradient problem),
    and scanning over several reasonable pulse durations.
For simplicity, this tutorial uses
    uniformly spaced square windows,
    the same parity-mapped and tapered LiH from our previous papers,
    and a single reasonable pulse duration.

=#

import CtrlVQE

##########################################################################################
###                              PHASE 0: INITIAL SETUP                                ###
##########################################################################################


##########################################################################################
#= Load a molecular system. =#

# At present, the code relies on the matrix representation of a molecular Hamiltonian,
#   most conveniently provided as a .npy file.
# Note that the matrix form is obtained *after* the fermion → qubit mapping.
import NPZ
matrixfile = "matrix/lih30.npy"
H = NPZ.npzread("$(@__DIR__)/$matrixfile")      # MOLECULAR HAMILTONIAN MATRIX
n = CtrlVQE.QubitOperators.nqubits(H)           # NUMBER OF QUBITS
ψ_REF = CtrlVQE.QubitOperators.reference(H)     # REFERENCE BASIS VECTOR
    #= NOTE: This `reference` method can select a state in the wrong Fock sector,
        so double-check it if you change the matrix. =#

# Perform an exact diagonalization to obtain the true ground state.
import LinearAlgebra
Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H))
ψ_FCI = U[:,1]                                  # GROUND STATE VECTOR

# Calculate the characteristic energies of the system.
REF = real(ψ_REF' * H * ψ_REF)                  # REFERENCE STATE ENERGY
FCI = Λ[1]                                      # GROUND STATE ENERGY
COR = REF - FCI                                 # CORRELATION ENERGY
FES = Λ[2]                                      # FIRST EXCITED STATE ENERGY, just for fun
GAP = FES - FCI                                 # ENERGY GAP FOR THIS HAMILTONIAN
println("""
       Reference energy: $REF
    Ground-state energy: $FCI
     Correlation energy: $COR
      1st-excited state: $FES
             Gap energy: $GAP
""")
println()

##########################################################################################
#= Design the ansatz, device and simulation parameters. =#

T = 30.0 # ns       # PULSE DURATION
W = 4               # NUMBER OF FOURIER MODES TO OPTIMIZE
m = 2               # NUMBER OF LEVELS PER TRANSMON
r = round(Int, 20T)                     # NUMBER OF TROTTER STEPS

evolution = CtrlVQE.TOGGLE      # Time-evolve using efficient-ish Trotterization.
basis = CtrlVQE.DRESSED         # Measure in the device basis.
frame = CtrlVQE.STATIC          # Measure in the rotated frame.

grid = CtrlVQE.TemporalLattice(T, r)

protopulse = CtrlVQE.UniformWindowed(CtrlVQE.ComplexConstant(0.0,0.0), T, W)
    #= NOTE: This is a pulse on a single qubit;
        it is automatically duplicated when we construct the device. =#

protodevice = CtrlVQE.Systematic(CtrlVQE.FixedFrequencyTransmonDevice, n, protopulse; m=m)
    #= NOTE: My devices have the pulse ansatz "baked in";
        we'll only use this one to get a Jacobian,
        then duplicate and modify it for successive stages. =#
#

# A few more miscellaneous derived objects...
nD = CtrlVQE.ndrives(protodevice)                       # Total number of drive signals.
x0 = CtrlVQE.Parameters.values(protodevice)             # Initial protopulse parameters.
O0 = CtrlVQE.QubitOperators.project(H, protodevice)     # Hamiltonian in transmon space.
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, protodevice) # Reference in transmon space.

##########################################################################################
#= Define how we measure the Jacobian. =#

# Define the multivariate function we want the Jacobian of.
function protoevolve(x)
    x0 = CtrlVQE.Parameters.values(device);
    CtrlVQE.Parameters.bind(device, x);
    ψ = CtrlVQE.evolve(evolution, device, basis, grid, ψ0);
    CtrlVQE.Parameters.bind(device, x0);
    return ψ
end
    #= NOTE: This is a little clunky.

    In order to evolve with a particular set of parameters,
        my code literally mutates the device
        (specifically, it mutates the parametric signals generating the drive terms).
    So we have to un-mutate at the end,
        to ensure the finite difference uses the same reference each time. =#

# Specify a finite difference method.
import FiniteDifferences
cfd = FiniteDifferences.central_fdm(5, 1)

# Method-agnostic function for the Jacobian.
jacobian(fn, xi) = FiniteDifferences.jacobian(cfd, fn, xi)[1]

##########################################################################################
###                      PHASE 1: NATURAL PULSES FROM HARTREE-FOCK                     ###
##########################################################################################

##########################################################################################
#= Measure the Jacobian using finite difference. =#
println("Phase 1: calculating the Jacobian...")
@time J1 = jacobian(protoevolve, x0)

##########################################################################################
#= Extract eigenpulses from the Jacobian. =#

# Perform singular value decomposition.
import LinearAlgebra
USV1 = LinearAlgebra.svd(J1)        # The SVD factorization.
nS1 = count(x -> x > 1e-8, USV1.S)  # The number of non-trivial singular values.

# Use the eigen-pulse parameters to make a list of natural pulses.
pulsesets_1 = Vector{typeof(protopulse)}[]
    #= NOTE: This is a vector of vectors!

    Each element of the outer vector is a delocalized natural pulse,
        which is itself a vector of signals on each qubit.

    The way we are going to extract each vector of signals is a little fancy.
    We're going to bind the eigen-parameters to our protodevice,
        then duplicate its list of signals, which have those parameters baked in.
        =#
for l in 1:nS
    CtrlVQE.bind(protodevice, USV.V[:,l])
    pulseset = [deepcopy(CtrlVQE.drivesignal(protodevice, i)) for i in 1:nD]
    push!(pulsesets_1, pulseset)
end
CtrlVQE.bind(protodevice, x0)







#= TODO: D'oh. I'm an idiot.

Linear combinations of pulses are great, but that's not what the problem was.

Parameters delocalized over multiple channels. That's the problem.
Obviously the parameterization for any single channel won't help with that.

Here's what we need:
Each channel has three signals:
    1. Amplitude        \/- alternative device has real/imag amplitudes.
    2. Phase            /
    3. Frequency
Each device has a list of channels, and a list of parameters.
Each parameter has ... what? A defined impact on each channel, somehow.

Um. Each signal parameter is a function of the device parameters;
    the device will need to store both the function itself (for parameter updates)
    and its analytical gradient (for the chain rule in the gradient).

=#











##########################################################################################
#= Specify the evolution algorithm and measurement protocol. =#


# H and ψ_REF live in a 2-level qubit space. Project them into the physical m-level space.
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN

# Select the energy function.
energyfn = CtrlVQE.ProjectedEnergy(evolution, device, basis, frame, grid, ψ0, O0)

# Combine it with a constraint on the amplitude.
lossfn = CtrlVQE.ConstrainedEnergyFunction(energyfn,
    CtrlVQE.GlobalAmplitudeBound(device, grid, ΩMAX, 1.0, ΩMAX),
)

##########################################################################################
#= Configure the optimizer. =#

# We will use the `Optim` package in this demo. It is not by any means the only choice.
import LineSearches, Optim
linesearch = LineSearches.MoreThuente()
optimizer = Optim.BFGS(linesearch=linesearch)
options = Optim.Options(
    show_trace = true, show_every = 1,  # Monitor progress as optimizations proceed.
    f_tol = 0,                          # Terminate if subsequent energies are this close.
    g_tol = 1e-6,                       # Terminate if the gradient norm is this small.
    iterations = 1000,                  # Give up after this many iterations.
)

# Construct the actual callable functions from our `lossfn` object.
f  = CtrlVQE.cost_function(lossfn)              # CALLABLE f(x)
g! = CtrlVQE.grad_function_inplace(lossfn)      # CALLABLE g!(∇f, x)

##########################################################################################
#= Run the optimization. =#

optimization = Optim.optimize(f, g!, xi, optimizer, options)
Lf = Optim.minimum(optimization)        # FINAL LOSS FUNCTION
xf = Optim.minimizer(optimization)      # FINAL PARAMETERS

Ef = energyfn(xf)                       # CURRENT ENERGY
λf = Lf - Ef                            # PENALTY CONTRIBUTION
εE = Ef - FCI                           # ENERGY ERROR
cE = 1 - εE/COR                         # CORRELATION ENERGY RECOVERED
gE = 1 - εE/GAP                         # "GAP" ENERGY RECOVERED

println("""

    Optimization Results
    --------------------
        Energy (Ha): $Ef
        Energy Error: $εE
    % Corr Energy: $(cE*100)
    %  Gap Energy: $(gE*100)

    Loss Function: $Lf
        from  Energy: $Ef
        from Penalty: $λf

""")

display(["$name: $value GHz" for (name, value) in zip(names, xf./2π)])
println()

##########################################################################################
#= Plot results. =#

# Prepare the time grid, from 0 to T with r steps (including T, this means r+1 points).
t = CtrlVQE.lattice(grid)

# Extract the timeseries values of the optimized pulse.
CtrlVQE.Parameters.bind(device, xf)             # Lock in the optimized pulse.
Ωt = Array{ComplexF64}(undef, r+1, n)           # Allocate space to hold pulse values...
for q in 1:n; Ωt[:,q] .= device.Ω̄[q](t); end    #   ...and fill that space.

# SET UP PLOT OBJECT
import Plots
yMAX = (ΩMAX / 2π) * 1.2        # Divide by 2π to convert angular frequency to frequency.
                                # Multiply by 1.2 to add a little buffer to the plot.
plot = Plots.plot(;
    xlabel= "Time (ns)",
    ylabel= "Amplitude (GHz)",
    ylims = [-yMAX, yMAX],
    legend= :topright,
)

# DUMMY PLOT OBJECTS TO SETUP LEGEND THE WAY WE WANT IT
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")   # Real part.
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot,   color=:black, label="β")   # Imag part.

# PLOT AMPLITUDES
for i in 1:n
    Plots.plot!(plot, t, real.(Ωt[:,i])./2π, lw=3, ls=:solid, color=i, label="Drive $i")
    Plots.plot!(plot, t, imag.(Ωt[:,i])./2π, lw=3, ls=:dot,   color=i, label=false)
end

# SHOW PLOT
Plots.gui()
