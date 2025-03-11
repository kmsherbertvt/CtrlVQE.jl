# import CtrlVQE

# import CtrlVQE.Bases: DRESSED
# import CtrlVQE.Operators: STATIC
# import CtrlVQE.QUBIT_FRAME

using CtrlVQE
using CtrlVQE.ModularFramework

import Optim
import Plots

import LinearAlgebra: eigen

const Float = Float64

##########################################################################################
#= SET UP THE PROBLEM =#

# DEFINE THE PROBLEM HAMILTONIAN
measurement = PauliMeasurement(DRESSED, STATIC;
    II=-0.6568598870801926,
    IZ=0.004188958260028297,
    ZI=0.1291013128871109,
    ZZ=0.12910131288711094,
    XZ=0.22953593605970196,
) #=
The goal of any VQE is to find the lowest eigenvalue of a Hermitian observable.

The `ModularFramework.PauliMeasurement` allows us to define the observable
    as a linear combination of Pauli words, which is helpful
    since in typical problems (e.g. electronic structure) this is a sparse representation.
Using this form of the constructor, each Pauli word is a new keyword argument,
    with its coefficient as the value;
    all Pauli words must have only "I", "X", "Y", or "Z",
    and they must all have the same number of letters.
Alternatively, if you have a dense matrix representation of your observable,
    you may pass it in as a positional argument.

The first two arguments relate to how measurements will be made experimentally.
- The first argument determines whether measuring in the computational basis
    projects qubit states onto the `BARE` basis of each qubit,
    or onto the `DRESSED` basis of the device.
- The second argument determines the measurement frame.
  Think of this argument as identifying the operator
    under which our quantum state evolves when considered "constant".
  If you want to work in the lab frame,
    where the state evolves under the static Hamiltonian of the device
    even in the absence of control fields, use `IDENTITY`.
  If you prefer to define "no control fields" as synonymous with "no evolution",
    use `STATIC`, representing the static Hamiltonian of the device.

Note that these objects (`BARE`, `DRESSED`, `STATIC`, `IDENTITY`) are *singleton* objects,
    meaning they are the only object of their respective type.
These types don't "do" anything; they serve only to act as a label for separate categories
    like the choice of computational basis or reference frame.
=#

##########################################################################################
#= SPECIFY THE ANSATZ =#

# DEFINE THE REFERENCE STATE
reference = KetReference(DRESSED, [1,0])
#=
VQEs typically try to take as a starting point
    (i.e. the state before applying any parameterized quantum circuit)
    the best product state they can,
    so that the circuit being optimized is essentially finding the optimal entanglement.
One also typically chooses to represent the problem Hamiltonian in a basis
    where the best product state is a computational basis state,
    which in principle is very easy to prepare.

The `ModularFramework.KetReference` allows us to define this basis state.
The first argument specifies which *basis*.
The second argument is a vector of qubit values (0 or 1) representing the ket.
This example says we are preparing the state |10⟩ in the dressed basis.
=#

# PULSE DURATION
T = 10.0 # ns

# NUMBER OF WINDOWS
W = 4

# DEFINE THE PULSE SHAPE(s)
ansatz_Ω = Windowed(Constant(0.0, 0.0), T, W)
ansatz_Δ = Constrained(Constant(0.0), :A)
#=
Control signals in a typical transmon architecture enter as time-dependent functions
    for the amplitude A(t), the frequency ν(t), and the phase ϕ(t) for each drive.
Practically speaking, it's easier to work
    with complex amplitudes Ω(t)≡A(t)⋅cis(ϕ(t)) and the detuning Δ(t)≡ν(t)-ω,
    where ω is the resonance frequency of the target qubit.

Each of these time-dependent functions
    should depend on some finite number of variational parameters.

Here, We are showcasing a few ways of constructing parameterized time-dependent functions.

The `Constant` constructor represents a constant time signal
    where the constant value is allowed to vary.
The argument determines the initial value and the type of number.
Passing a single real number (e.g. `0.0`) means there is just one real-valued parameter.
Passing two real numbers, or alternatively a single complex number,
    produces a constant signal where the first parameter is the real part,
    and the second is the imaginary part.

The `Windowed` constructor takes an existing signal
    and breaks it apart into equal-sized chunks.
The second argument (`T` in this example) specifies the total duration,
    and the third argument specifies how many chunks to divide that duration into.
In this example, each chunk has two parametes (a real and imaginary part)
    so our pulse shape for Ω(t) will have four parameters for each qubit.
=#
println("Parameters for each Ω(t)")
display(Parameters.names(ansatz_Ω))
#=
The `Constrained` constructor takes an existing signal
    and freezes some parameters, identified by name in subsequent arguments.
A real `Constant` signal has just one parameter labeled `A`,
    and our example here freezes that parameter to its initial value `0.0`,
    so our pulse shape for Δ(t) will be fixed on resonance throughout the pulse.

=#
println("Parameters for each Δ(t)")
display(Parameters.names(ansatz_Δ))


##########################################################################################
#= SPECIFY THE SIMULATION COST =#

# BOSONIC TRUNCATION LEVEL
m = 3

# NUMBER OF TROTTER STEPS
r = 200

# INITIALIZE THE TIME GRID
grid = TemporalLattice(T, r)
#=
We'll be integratin Schrödinger's equation over time.
The `grid` object defines how this integration is conducted,
    including the discretization (from `r`) and the integration bounds (from t=0 to `T`).
=#

##########################################################################################
#= CONSTRUCT THE DEVICE =#

#=
Here we are going to construct the object representing the quantum computer to simulate.
The basic types provided by this package offer a number of ways to do so,
    but the most flexible and intuitive is provided by the `ModularFramework` module,
    so let's focus on that for this tutorial.

The `ModularFramework` entails constructing a few intermediate objects,
    each representing a different piece of the whole device Hamiltonian.
Alternatively, our choice of ansatz,
    resonant pulses with complex amplitudes divided into equal-length windows,
    could be equally well constructed with the somewhat lighter-weight type `CWRTDevice`.

=#

# DEFINE THE ALGEBRA
n = length(reference.ket)
algebra = TruncatedBosonicAlgebra{m,n}()
#=
The `algebra` specifies the Hilbert space that a device Hamiltonian acts on.
Transmon devices should use the `ModularFramework.TruncatedBosonicAlgebra` constructor,
    representing a joint space of distinguishable bosonic modes
    truncated to a finite occupation number.
The constructor takes no parameters, but the type itself has two *type parameters*,
    which are `m` - the number of levels retained for each transmon,
    and `n` - the number of transmons.
=#

A = algebratype(algebra)
#=
The later objects are parameterized by the type of the algebra, so we fetch it here.
The `ModularFramework.algebratype` function lets you fetch the algebra type
    associated with any of these objects.
For the `algebra` object itself, it's identical to the built-in `typeof` function.
=#

# DEFINE THE STATIC HAMILTONIAN
drift = TransmonDrift{A}(
    2π.*[4.82, 4.84],               # QUBIT RESONANCE
    2π.*[0.30, 0.30],               # ANHARMONICITIES
    2π.*[0.02],                     # COUPLING STRENGTH
    [Quple(1,2)],                   # COUPLING MAP
)
#=
We are using a fixed-coupling transmon Hamiltonian,
    which can be modeled as a set of linearly-coupled anharmonic oscillators.
We use `ModularFramework.TransmonDrift` to define the static Hamiltonian, or *drift* term.

An anharmonic oscillator is defined by a resonance frequency,
    which determines the effective width of the potential well,
    and the *anharmonicity* which determines the degree to which
    the potential well is not quadratic like a harmonic oscillator.
The first two arguments list the resonance frequencies and anharmonicities for each qubit.

The third argument is a list of coupling strengths,
    one for each distinct pair of transmons that are coupled.
Typically, transmon devices have relatively limited connectivity.
The coupling map is specified in the fourth argument,
    which identifies which two qubits are involved in each pair.
Think of the `Quple` object as a regular tuple except that order doesn't matter
    (since `(2,1)` is not a *distinct* pair from `(1,2)`).

=#

# DEFINE THE DRIVES
drives = [
    DipoleDrive{A}(q, drift.ω[q], deepcopy(ansatz_Ω), deepcopy(ansatz_Δ))
        for q in 1:n
]
#=
The drive terms are modeled by a dipole interaction of each transmon
    with the electric field induced by our control signals.
In transmon architectures, control signals can target specific qubits,
    so there is typically one drive term for each qubit.
We are creating all drives at once here.

The `ModularFramework.DipoleDrive` type approximates the dipole interaction
    with the rotating wave approximation,
    which requires knowing the resonance frequency of the transmon.
The first argument simply indicates which qubit is being targeted,
    and the second simply gives its resonance frequency.
The third and fourth arguments are where we actually define
    the full time-dependent control signals for complex amplitude and detuning,
    but here we just copy the `ansatz` objects we created earlier.
Note that we have to *copy* them,
    or else all drives would share the same variational parameters.

=#

device = LocalDevice(Float, algebra, drift, drives, DISJOINT)
#=
Finally, we combine our `algebra`, `drift`, and `drives` into one `device` object,
    representing the complete Hamiltonian that we will integrate.

The first parameter, `Float`, simply specifies the float precision (typically 64 bits).

The last parameter, `ModularFramework.DISJOINT`,
    indicates that the variational parameters for each drive are independent of one another.
Alternatively, you could pass a `ModularFramework.Linear` object,
    which sets the variational parameters of the device
    to be the coefficients in front of fixed couplings of all drive parameters.

=#
println("Parameters for the device")
display(Parameters.names(device))

##########################################################################################
#= DEFINE THE LOSS FUNCTION =#

# CONSTRUCT THE ENERGY FUNCTION
energy = Energy(QUBIT_FRAME, device, grid, reference, measurement)
#=
Measuring an energy entails solving Schrödinger's equation,
    i.e. integrating the device hamiltonian over time,
    and then measuring the resulting state.
The initial conditions of the integration are provided by the `reference`.

For the most part, this is just encapsulating objects we've already talked about.
The only new thing is the first argument,
    which specifies the *algorithm* for performing time evolution
    (which is distinct from the mathematical definiton of the integral given by `grid`).
The original implementation of ctrl-VQE, accessed by `ROTATING_FRAME`,
    constructs the interaction picture Hamiltonian at each time step
    and exponentiates it.

Alternatively, the `QUBIT_FRAME` algorithm alternately evolves
    under the drift and drive terms at each time step.
Since the drift term never changes,
    only the exponentiation of the drive terms needs to be constructed at each time step.
When drive terms in the Hamiltonian are represented compactly in the bare qubit basis,
    this exponentiation is easy and the `QUBIT_FRAME` is significantly more efficient.

=#

# CONSTRUCT THE PENALTY FUNCTIONS
ΩMAX = 2π * 0.02 # GHz
penalties = [
    SignalStrengthPenalty(grid, drives[q].Ω; A=ΩMAX)
        for q in 1:n
]
#=
We can only put so much energy into our device at a time without burning it out.
This typically manifests as a maximum amplitude we can apply in A(t), equivalently |Ω(t)|.
The `SignalStrengthPenalty` object represents a penalty which integrates over A(t)
    and penalizes any area under the curve that appears over the maximum amplitude.

In general, different drives may have a different maximum amplitude,
    so we construct a distinct penalty term for each drive.
=#

# CONSTRUCT THE LOSS FUNCTION
lossfn = CompositeCostFunction(energy, DrivePenalty(device, penalties))
#=
The `ModularFramework.DrivePenalty` object simply wraps together
    a penalty for each drive in the device,
    managing the mapping from device parameters to drive parameters.
Note that, since we are using a completely constrained Δ(t),
    the drive parameters are equivalent to the signal parameters for Ω(t).
In general, an additional wrapper
    between the drive and the `SignalStrengthPenalty` might be required.

The `CompositeCostFunction` object simply takes multiple existing cost functions
    and adds them together.
As a convenience, it also remembers the last-computed values for each component,
    which can help avoid repeating
    as we shall see.
=#

##########################################################################################
#= RUN THE OPTIMIZATION =#

# CONSTRUCT THE ACTUAL FUNCTIONS TO OPTIMIZE
f  = cost_function(lossfn)
g! = grad!function(lossfn)
#=
In spite of its name, `lossfn` is an object *specifying* a cost function;
    it is not itself a callable function.
We construct the callable function with the `cost_function` function.
This function accepts a parameter vector and returns a real scalar.

Similarly, we construct the callable gradient function with the `grad_function` function,
    which accepts a parameter vector and returns a vector,
    of partial derivatives of the cost function with respect to each parameter.
This information enables gradient descent without resorting to finite differences.

Most optimization packages prefer a two-argument form of the gradient function,
    where the first argument is an empty vector to which the gradient will be written.
This helps avoid allocating a new vector in every iteration,
    minimizing overhead from memory management and garbage allocation.
We construct this two-argument form with the `grad!function` function.

=#

# INITIAL PARAMETER VECTOR
xi = collect(Parameters.values(device))
#=
Local optimization algorithms generally require an initial guess for the parameters.
Zero is usually a sensible choice.
We've constructed our ansatz using zeros as the initial parameters,
    so here we can just fetch the current values of those parameters.
=#

# RUN THE OPTIMIZATION
optimizer = Optim.BFGS()
options = Optim.Options(
    show_trace = true,
    show_every = 1,
    f_tol = 0.0,
    g_tol = 1e-6,
    iterations = 1000,
)
optimization = Optim.optimize(f, g!, xi, optimizer, options)
xf = Optim.minimizer(optimization)      # FINAL PARAMETERS
#=
We're using the `Optim` package in this tutorial,
    but any optimization package works just as well.
The details are out of scope of this tutorial,
    but I'm always happy to answer any questions
    if you'd like to understand how the optimization works!
=#

##########################################################################################
#= REPORT RESULTS =#

# CALCULATE ENERGY ERROR AND CORRELATION ENERGY
H = observables(measurement, device)[:,:,1]
FCI = eigen(H).values[1]            # FULL CONFIGURATION INTERACTION
#=
To get the energy error, we need the exact energy.
`ModularFramework.observables` lets us convert our `measurement` into a dense observable.
Certain measurement types (e.g. a normalized energy measurement)
    have multiple distinct observables associated with them
    (e.g. the energy of the projection onto the two-level space,
    and the leakage outside of it)
    but `ModularFramework.PauliMeasurement` combines all terms into one matrix,
    so we use `[:,:,1]` to restrict the output of `ModularFramework.observables`
    to only the first (and last) matrix.

We then use Julia's built-in eigensolver to extract the lowest eigenvalue.
=#

ψ0 = prepare(reference, device)
REF = real(ψ0' * H * ψ0)            # HARTREE-FOCK ENERGY
#=
We can use `ModularFramework.prepare` to convert our `reference` into a dense vector.
In conjunction with the dense observable, we can compute the reference energy.

Alternatively, since we are operating in the `STATIC` frame,
    we can just compute the loss function with zero parameters.
=#
println("""
    Exact energy: $FCI
Reference energy: $REF
      Zero-pulse: $(f(zero(xf)))
""")

# OPTIMIZED LOSS FUNCTION
Lf = f(xf)
Ef = first(lossfn.values)
Λf = last(lossfn.values)
#=
The optimization has given us an optimal choice of parameters `xf`.
Call the function one last time to get the final energy.
Doing so also writes the optimized values
    of the energy and penalty components to `lossfn.values`.
=#
println("""
Final loss function: $Lf
      due to energy: $Ef
     due to penalty: $Λf
""")

# SUCCESS METRICS
εE = Ef - FCI           # ENERGY ERROR
cE = 1-εE/(REF-FCI)     # CORRELATION ENERGY
#=
Obviously, we want a small energy error.
Chemists also like to use *correlation energy*,
    the increase in energy magnitude over the reference energy.
The maximum correlation energy is the difference between the reference and exact.
Here I'll report the fraction of that maximum correlation energy.
=#
println("""
     Energy Error: $εE
    % Corr Energy: $(cE*100)
""")

# PLOT PULSES
plts = []
t = collect(grid)
for (i, drive) in enumerate(device.drives)
    Ω = valueat(drive.Ω, grid)

    plt = Plots.plot(;
        xlabel = "Time (ns)",
        ylabel = "Qubit $i Quadratures (GHz)",
        framestyle = :origin,
        ylims = [-ΩMAX, +ΩMAX] ./ 2π,
    )
    Plots.plot!(
        plt, t, zero(t); fillrange = real.(Ω) ./ 2π,
        color=:lightblue, fillalpha=0.5, linealpha=0.0, label=false,
    )
    Plots.plot!(
        plt, t, zero(t); fillrange = imag.(Ω) ./ 2π,
        color=:darkblue, fillalpha=0.5, linealpha=0.0, label=false,
    )
    push!(plts, plt)
end
Plots.plot(plts...)
Plots.gui()