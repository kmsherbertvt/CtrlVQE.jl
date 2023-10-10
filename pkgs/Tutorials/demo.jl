#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

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
    Ground-state energy:
     Correlation energy:
      1st-excited state: $FES
             Gap energy: $GAP
""")
println()

##########################################################################################
#= Construct the device. =#

T = 10.0 # ns       # PULSE DURATION
W = 4               # NUMBER OF WINDOWS PER PULSE
m = 2               # NUMBER OF LEVELS PER TRANSMON

# In my formalism, the "device" includes the microwave pulse generator,
#   so we need to start by defining our pulse signal.
protopulse = CtrlVQE.ComplexConstant(0.0, 0.0)
println("Parametric form of each window: $(string(protopulse))")
println()

# Just for the sake of demonstration,
#   here is how you would freeze a parameter:
frozen_imag = CtrlVQE.ConstrainedSignal(CtrlVQE.ComplexConstant(0.0, 0.5), :B)
println("Parametric form of a constrained window: $(string(frozen_imag))")
println()
# But let's not actually use that.

# Break the pulse up into windows with independently optimizable parameters.
windowedpulse = CtrlVQE.UniformWindowed(protopulse, T, W)
println("Parametric form of the windowed pulse: $(string(windowedpulse))")
println()

# Create the device, using some standardized settings which roughly emulate real devices.
device = CtrlVQE.Systematic(CtrlVQE.TransmonDevice, n, windowedpulse; m=m)
    #= NOTE:
        To manually specify all the settings (eg. resonance frequency, coupling, etc.),
            you should call a constructor for `CtrlVQE.TransmonDevice`.
        But it's a little ugly, so you'll have to find out how from the documentation. ;)

        The "normal" use of the `Systematic` function
            is to specify the signals for each drive term,
            by passing a Vector of signals as the third parameter.
        But, as a convenience, if just one signal is passed, like here,
            it gets copied so that there is one drive term for each qubit.
    =#

# How many parameters are there?
# It's the number of parameters per window, times the number of windows per pulse,
#   times the number of pulses... *plus* the number of drive frequencies!
# ...here's a shortcut:
L = CtrlVQE.Parameters.count(device)
println("Total # of parameters: $L")
println()

# Okay but which parameters are what?
# That is, admittedly, confusing. It's probably easiest to just show you...
names = CtrlVQE.Parameters.names(device)
display(names)
println()

# What do the parameters start as?
xi = CtrlVQE.Parameters.values(device)
display(["$name: $value" for (name, value) in zip(names, xi)])
println()
    #= NOTE:
        All the amplitudes are zero,
            because our protopulse was constructed with 0.0 for both A and B.
        (That one protopulse just got copied a whole bunch.)

        Pulse frequencies initialize to the qubit resonance frequencies,
            when using the `Systematic` constructor.
    =#

# Which indices correspond to amplitudes and which to frequencies?
#   Alas, this is something you have to work out manually.
Ω = 1:L-n               # VECTOR OF INDICES DESIGNATING AMPLITUDES
ν = 1+L-n:L             # VECTOR OF INDICES DESIGNATING FREQUENCIES


##########################################################################################
#= Specify the evolution algorithm and measurement protocol. =#

r = round(Int, 20T)
    #= NOTE: Error scales with the step-size T÷r,
        so it's generally prudent to select r based on T. =#

# Encapsulate temporal quantities into an `Integration` object.
grid = CtrlVQE.TemporalLattice(T, r)
    #= NOTE: The `Integration` object is this arcane thing some parts of the code use
            to integrate certain functions (eg. gradient signals) from t=a to t=b.
        The `TemporalLattice` recipe is short-hand for `TrapezoidalIntegration(0.0, T, r),
            which calculates integrals using a trapezoidal Riemann sum from t=0 to t=T.

        I named the recipe `TemporalLattice`,
            since the more visible role of the object is that it defines the time lattice.
    =#

# Construct the evolution algorithm.
evolution = CtrlVQE.TOGGLE
    #= NOTE: The `Toggle` algorithm is probably the always the best choice
            when the device Hamiltonian is represented internally as a dense matrix
            and when the time-dependent part is factorizable (ie. qubit-local drives).

        I think it's probably still the best given only the second condition,
            but don't hold me to that.
    =#

# H and ψ_REF live in a 2-level qubit space. Project them into the physical m-level space.
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)          # REFERENCE STATE
    #= NOTE: We have been implicitly assuming
            that it is easy to prepare arbitrary basis vectors.
        But this involves a layer of X gates, typically, which is not a big deal
            but it *may* be a long time relative to T.
        It's probably worth asking how much longer T needs to be
            to allow us to start from |0⟩.
    =#

# Select the measurement basis and frame.
basis = CtrlVQE.OCCUPATION
    #= NOTE: Choosing CtrlVQE.OCCUPATION takes measurement as
            with respect to the logical qubit basis
            (which corresponds to the
        This means that, after projective measurement,
            we presume the computer is in a logical basis state.

        Quite frankly, it's almost certainly actually in
            an *eigenstate of the static device Hamiltonian.
        Model this by using `CtrlVQE.DRESSED` instead.
        Note that doing so also reinterprets O0 as ψ0 in this basis
            (which is probably correct:
                resetting to the |0⟩ state is probably resetting
                to the lowest-energy state of the device).
        I'm using CtrlVQE.OCCUPATION for the demo because,
            well, it's what I've been using for most of a year.
            I'm comfortable with it, and it is more thoroughly tested.
    =#

frame = CtrlVQE.STATIC
    #= NOTE: I think this (and its interaction with the projective measurement)
        is the most confusing part of this whole ctrl-VQE thing.

    Consider: we expect that NOT applying any microwave pulse to our reference state
        should NOT change the state.
    But, we know that the reference state evolves under the device's static Hamiltonian.
    The solution is to work in a *time-dependent basis*
        for which the reference state always has the exact same form.
    We call this a frame rotation,
        and we achieve it by considering the ouput state |ψ̃(t)⟩ = exp(i H0 t) |ψ(t)⟩

    The choice `CtrlVQE.STATIC` selects this frame.

    The trouble is, it seems like measurement in the lab
        should happen in the so-called "lab frame" - we measure |ψ(t)⟩, not |ψ̃(t)⟩.
    Thus, the frame rotation seems to properly occur in the *classical* post-processing
        (practically, we rotate the observable instead,
            ie. H → H̃ = exp(-i H0 t) H exp(i H0 t) ),
        which is not necessarily tractable experimentally.
    One alternative is to *approximate* the static frame
        by omitting any entangling terms in the exponential.
    You can, in principle, achieve this by selecting `frame = CtrlVQE.UNCOUPLED`,
        but it isn't tested thoroughly.

    You can omit any frame rotation at all by selecting `frame = CtrlVQE.IDENTITY`.

    Another alternative is to manually rotate your state prior to measurement,
        rather than trying to do it classically.
    Then you need to ask questions like "how do I unevolve by the device Hamiltonian"
        and "how does projective measurement interact with this final basis rotation",
        which make my head hurt.
    And don't get me started on where the basis rotation
        to measure non-diagonal Pauli terms comes in!
    Needless to say, there is still a ways to go to model a practical ctrl-VQE experiment.

    =#

# Select the energy function.
energyfn = CtrlVQE.ProjectedEnergy(evolution, device, basis, frame, grid, ψ0, O0)
    #= NOTE: `ProjectedEnergy`

    Let Π be the projector onto the logical two-level space,
            and Õ be the frame-shifted observable as discussed above.

    The `ProjectedEnergy` calculates ⟨ψ|ΠÕΠ|ψ⟩,
        where Π is the projector onto the logical two-level space,
        and Õ is the frame-shifted observable as discussed above.

    Its key features are that frame rotation occurs *after* projective measurement,
        meaning it models a protocol where the frame shift must be applied classically,
    and any leakage in |ψ⟩ is annihilated,
        meaning the expectation ⟨Õ⟩ is effectively with respect to an unnormalized state.
    As long as the ground state energy is negative,
        the leakage certainly raises the energy,
        and an optimization to minimize ⟨Õ⟩ will naturally eliminate leakage.

    Alternatively use the `NormalizedEnergy` ⟨ψ|ΠÕΠ|ψ⟩ / ⟨ψ|Π|ψ⟩,
        which renders leakage invisible to the optimizer.
    Which choice is truer to experiment
        depends on the "discriminator" in the measurement protocol.

    Another alternative is the `BareEnergy` ⟨ψ|Õ|ψ⟩, which omits explicit projection,
        but actually there is implicit projection when writing the molecular Hamiltonian
        as an operator on the full physical space, so really the `BareEnergy` models when
        the frame rotation is applied before measurement.
    There should probably also be a normalized version of the latter, ⟨ψ|Õ|ψ⟩ / ⟨ψ|Π̃|ψ⟩,
        but this is not currently implemented.

    =#

##########################################################################################
#= Configure the optimizer. =#

ΩMAX = 2π * 0.02 # GHz      # MAXIMUM PULSE AMPLITUDE
ΔMAX = 2π * 1.0 # GHz       # MAXIMUM PULSE DETUNING (Δ ≡ ω - ν)
    #= NOTE: Ω and Δ are implicitly in units of ANGULAR frequency, hence the 2π. =#

# HACK: Keep complex pulses truly under the hardware amplitude bound.
ΩMAX /= √2
    #= NOTE: "Complex pulses" are, experimentally,
        just *real* pulses with a relative phase between the control and drive signals.

    The amplitude bound applies to this *real* pulse, whose magnitude is
        the modulus of the real and imaginary components A and B of our complex pulse.
    So really, we should be enforcing the constraint A² + B² ≤ ΩMAX.
    In other words, we have stay within the bounds of a circle with radius ΩMAX.

    It is however much easier numerically to enforce independent bounds on each parameter.
    In other words, we stay within the bounds of a *square*, with sidelength 2ΩMAX.
    By dividing off a factor of √2 from ΩMAX, we ensure
        the square used in simulation *inscribes* the circle relevant in experiment.
    =#

# We will use the `Optim` package in this demo. It is not by any means the only choice.
import LineSearches, Optim
linesearch = LineSearches.MoreThuente()
    #= NOTE: I don't understand the linesearches very well.
        Empirically I have found MoreThuente does marginally better
            than the default HagerZhang,
            and Guo Xuan has gotten better results with BackTracking,
            but this is likely system dependent and probably not all that significant.
        =#
optimizer = Optim.BFGS(linesearch=linesearch)
    #= NOTE: BFGS is always a good fallback if you have reliable gradients.
        There is a more memory-efficient approximation, `LBFGS`,
            which may be preferred if you are working with many many parameters.
        Note, however, that we do NOT expect to have reliable gradients
            in an experimental protocol.
    =#
options = Optim.Options(
    show_trace = true, show_every = 1,  # Monitor progress as optimizations proceed.
    f_tol = 0,                          # Terminate if subsequent energies are this close.
    g_tol = 1e-6,                       # Terminate if the gradient norm is this small.
    iterations = 1000,                  # Give up after this many iterations.
)

# To enforce bounds, we need to modify our cost function to include penalty terms.
#   The code includes a framework for doing so:
λ = zeros(L); λ[Ω] .= 1.0; λ[ν] .= 1.0          # STRENGTH OF THE PENALTY TERMS
σ = zeros(L); σ[Ω] .= ΩMAX; σ[ν] .= ΔMAX        # STEEPNESS OF THE PENALTY TERMS
uB = copy(xi); uB[Ω] .+= ΩMAX; uB[ν] .+= ΔMAX   # UPPER BOUNDS FOR EACH PARAMETER
lB = copy(xi); lB[Ω] .-= ΩMAX; lB[ν] .-= ΔMAX   # LOWER BOUNDS FOR EACH PARAMETER
boundsfn = CtrlVQE.SmoothBound(λ, uB, lB, σ)
    #= NOTE: The smooth bound function is so named
        because its functional form is "smooth", ie. infinitely differentiable,
        even at the bounds (within which the penalty is defined to be zero).

    As a consequence, the penalty is very shallow at the bounds,
        and the optimizer may "cheat" by cutting into the bounds a small bit.
    Using the parameters here, it seems to cheat by about 5% or so,
        though this should in principle depend on the exact shape of the energy surface.
    If this is an issue, the simplest solution is to just lower the bounds a small bit.

    I think there is a more sophisticated approach to optimization
        which constrains the penalty terms to exactly zero,
        but I haven't thought about it very much.
    =#

# Combine our energy function with the penalty terms.
costfn = CtrlVQE.CompositeCostFunction(energyfn, boundsfn)
f  = CtrlVQE.cost_function(costfn)              # CALLABLE f(x)
g! = CtrlVQE.grad_function_inplace(costfn)      # CALLABLE g!(∇f, x)
    #= NOTE: Note the `!` in the variable name `g!` is Julia-code for signaling that
            `g!` is a function which mutates at least one of its arguments.
        The inplace version of the gradient function includes a parameter ∇f,
            to which the result is written. Its starting value is irrelevant.
        This is the form of the gradient which Optim prefers,
            since it allows a more memory-efficient optimization.

        Alternatively, you could use `g = CtrlVQE.grad_function(costfn)`
            for a callable g(x) which returns the vector ∇f.
    =#

##########################################################################################
#= (Optional) Perturb the initial parameter vector by a random amount. =#

seed = 0000                 # RANDOM SEED
kick_Ω = ΩMAX               # MAXIMUM KICK FOR AMPLITUDES
kick_Δ = ΔMAX               # MAXIMUM KICK FOR FREQUENCIES

# Perturb the initial parameters by a random amount.
import Random; Random.seed!(seed)
xi[Ω] .+= ΩMAX .* (2 .* rand(length(Ω)) .- 1)
xi[ν] .+= ΔMAX .* (2 .* rand(length(ν)) .- 1)

println("Parameters after random kicks:")
display(["$name: $value" for (name, value) in zip(names, xi)])

##########################################################################################
#= Run the optimization. =#

# Run the optimization.
optimization = Optim.optimize(f, g!, xi, optimizer, options)
Lf = Optim.minimum(optimization)        # FINAL LOSS FUNCTION
xf = Optim.minimizer(optimization)      # FINAL PARAMETERS

Ef = energyfn(xf)                       # CURRENT ENERGY
λf = boundsfn(xf)                       # PENALTY CONTRIBUTION
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

# Calculate detunings.
Δ̄ = device.ω̄ .- device.ν̄
    #= NOTE: This method of calculating the detunings is not standardized;
        other devices may not work like this, but the systematic TransmonDevice does. =#

println("Detunings for each qubit (GHz):")
display(Δ̄ ./ 2π)        # Divide by 2π to convert angular frequency to frequency.


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