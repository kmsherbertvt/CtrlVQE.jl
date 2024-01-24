#= Optimize a pulse using the trigonometric ansatz Hisham came up with.

Generate distinct pulses for the real and imaginary components,
    decomposed as a Fourier series with period T as the pulse duration.

    ∑[ a[i] cos(πt/T) + b[i] sin(πt/T) ]

The coefficients a[i] and b[i] are your parameters.
In practice we probably like the signal to start and end on 0,
    so all a[i] can be constrained to zero.
Simply include as many modes as parameters you want to have to optimize.
Easily "adapted" by adding on higher modes iteratively,
    perhaps freezing the lower modes in place.

This script will also manually construct a transmon device,
    so that it is easier to tailor to a particular real device.

=#

import CtrlVQE

##########################################################################################
#= Load a molecular system. =#

# At present, the code relies on the matrix representation of a molecular Hamiltonian,
#   most conveniently provided as a .npy file.
# Note that the matrix form is obtained *after* the fermion → qubit mapping.
import NPZ
matrixfile = "matrix/H2_sto-3g_singlet_1.5_P-m.npy"
H = NPZ.npzread("$(@__DIR__)/$matrixfile")      # MOLECULAR HAMILTONIAN MATRIX
    #= NOTE: @__DIR__ gives the directory of the actual script itself,
            rather than the present working directory. =#

# Infer the number of qubits from the size of the matrix.
n = CtrlVQE.QubitOperators.nqubits(H)
println("Number of qubits: $n")
println()

# Infer the reference state from whichever basis state gives the smallest energy.
ψ_REF = CtrlVQE.QubitOperators.reference(H)     # REFERENCE BASIS VECTOR
    #= NOTE: Sometimes this will select a basis state in the wrong Fock sector
            (ie. the wrong charge or the wrong spin),
            so you should check the reference energy against, eg. pyscf. =#

# Perform an exact diagonalization to obtain the true ground state.
import LinearAlgebra
Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H))
    #= NOTE: `Hermitian(H)` creates a special "view" into `H`
            (meaning we have a new object reference, but the actual data is shared)
            which simply *ignores* the lower triangle part of `H`.
        Whenever you ask for an element in the lower triangle of `Hermitian(H)`,
            it just gives you the complex conjugate
            of the corresponding element in the upper triangle of `H`.
        Similarly, when you ask for a diagonal element,
            it just gives you the real part of the diagonal element in `H`.

        Doing this ensures that the `eigen` function uses a specialized diagonalization
            which is more efficient for Hermitian matrices.
    =#
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
#= Design the ansatz. =#

T = 14.3 # ns       # PULSE DURATION
W = 4               # NUMBER OF FOURIER MODES TO OPTIMIZE

# Convenience constructor for the n-th Fourier sine mode.
sinemode(n,T) = CtrlVQE.ConstrainedSignal(
    CtrlVQE.Sine(0.0, π*n/T, 0.0), # The 3rd arg is phase. Change it to π/2 for a cosine.
    :ν, :ϕ,                         # Flag frequency and phase as non-variational.
)

# The following is a bit obscure...
#   - AnalyticSignal scales our real-valued sine functions by a complex number,
#       either 1 for the real part or i for the imaginary part.
#   - reduce(vcat, ...) assembles a list of the real and imaginary sines for each mode.
#   - CompositeSignal adds all those sines into a single pulse.
protopulse = CtrlVQE.CompositeSignal(reduce(vcat, [
    CtrlVQE.AnalyticSignal(sinemode(n,T),  1),      # nth Fourier mode for the real part
    CtrlVQE.AnalyticSignal(sinemode(n,T), im),      #        "         for the imag part
] for n in 1:W))


##########################################################################################
#= Construct the device. =#

# These numbers are taken from the first two qubits of IBMQ's algiers device.
device = CtrlVQE.FixedFrequencyTransmonDevice(  # Vary frequenices with `TransmonDevice`.
    2π .* [4.9462424483428675, 4.8360305756841],        # RESONANCE FREQUENCIES
    2π .* [0.3441379643980549, 0.34870027897863465],    # ANHARMONICITIES
    2π .* [0.0018554933031236817],                      # LIST OF COUPLING STRENGTHS
    [CtrlVQE.Quple(1,2)],                               # LIST OF COUPLINGS
    [1, 2],                                             # LIST OF QUBITS BEING DRIVEN
    2π .* [4.9462424483428675, 4.8360305756841],        # LIST OF DRIVE FREQUENCIES
    [deepcopy(protopulse) for _ in 1:2],                # LIST OF DRIVE SIGNALS
    2,                                                  # NUMBER OF TRANSMONS PER LEVEL
)
#= The actual amplitude caps: 2π .* [0.1276456365439478, 0.13370758447967016]
    So, we'll be sure to set Ω0 to, say, 2π*0.1275.
=#
ΩMAX = 2π * 0.1275 # GHz    # MAXIMUM PULSE AMPLITUDE

# There should be 2nW parameters: a real and imag amplitude for each mode, for each qubit.
L = CtrlVQE.Parameters.count(device)
println("Total # of parameters: $L")
println()

# All parameters should start at 0.0, since in the `sinemode` function
#   we used 0.0 as the sine's amplitude, and all other parameters were constrained.
names = CtrlVQE.Parameters.names(device)
xi = CtrlVQE.Parameters.values(device)
display(["$name: $value GHz" for (name, value) in zip(names, xi./2π)])
println()

##########################################################################################
#= Specify the evolution algorithm and measurement protocol. =#

# Encapsulate temporal quantities into an `Integration` object.
r = round(Int, 20T)
grid = CtrlVQE.TemporalLattice(T, r)

# Construct the evolution algorithm.
evolution = CtrlVQE.TOGGLE

# H and ψ_REF live in a 2-level qubit space. Project them into the physical m-level space.
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)          # REFERENCE STATE

# Select the measurement basis and frame.
basis = CtrlVQE.OCCUPATION
frame = CtrlVQE.STATIC

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
