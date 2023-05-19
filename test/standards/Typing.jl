#= We want to work out a systematic way of testing new devices. =#

import CtrlVQE: Parameters, Signals, Devices
import CtrlVQE.Devices: LocallyDrivenDevices
import CtrlVQE.Bases: DRESSED, OCCUPATION
import CtrlVQE.Operators: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE.Operators: Qubit, Channel, Drive, Hamiltonian, Gradient

const t = 0.0
const r = 10
const τ = t / r

const τ̄ = fill(τ, r+1); τ̄[[1,end]] ./=2
const t̄ = range(0.0, t, r+1)
const ϕ̄ = ones(r+1)

function check_types(device::Devices.Device)
    @code_warntype Devices.nlevels(device)
    @code_warntype Devices.nqubits(device); q = Devices.nqubits(device)
    @code_warntype Devices.nstates(device); N = Devices.nstates(device)
    @code_warntype Devices.ndrives(device); i = Devices.ndrives(device)
    @code_warntype Devices.ngrades(device); j = Devices.ngrades(device)

    println("Parameters and Gradient")

    @code_warntype Parameters.count(device)
    @code_warntype Parameters.names(device)
    @code_warntype Parameters.values(device); x̄ = Parameters.values(device)
    @code_warntype Parameters.bind(device, x̄)

    ϕ̄_ = repeat(ϕ̄, 1, j)
    @code_warntype Devices.gradient(device, τ̄, t̄, ϕ̄_)
    grad = Devices.gradient(device, τ̄, t̄, ϕ̄_)
    @code_warntype Devices.gradient(device, τ̄, t̄, ϕ̄_; result=grad)


    println("Hamiltonian Terms")
    @code_warntype Devices.algebra(device); ā = Devices.algebra(device)

    @code_warntype Devices.eltype_localloweringoperator(device)
    @code_warntype Devices.localloweringoperator(device)
    a = Devices.localloweringoperator(device)
    @code_warntype Devices.localloweringoperator(device; result=a)

    @code_warntype Devices.eltype_qubithamiltonian(device)
    @code_warntype Devices.qubithamiltonian(device, ā, q)
    B = Devices.qubithamiltonian(device, ā, q)
    @code_warntype Devices.qubithamiltonian(device, ā, q; result=B)

    @code_warntype Devices.eltype_staticcoupling(device)
    @code_warntype Devices.staticcoupling(device, ā)
    C = Devices.staticcoupling(device, ā)
    @code_warntype Devices.staticcoupling(device, ā; result=C)

    @code_warntype Devices.eltype_driveoperator(device)
    @code_warntype Devices.driveoperator(device, ā, i, t)
    D = Devices.driveoperator(device, ā, i, t)
    @code_warntype Devices.driveoperator(device, ā, i, t; result=D)

    @code_warntype Devices.eltype_gradeoperator(device)
    @code_warntype Devices.gradeoperator(device, ā, j, t)
    E = Devices.gradeoperator(device, ā, j, t)
    @code_warntype Devices.gradeoperator(device, ā, j, t; result=E)


    println("Basis Control")

    @code_warntype Devices.globalize(device, a, q)
    F = Devices.globalize(device, a, q)
    @code_warntype Devices.globalize(device, a, q; result=F)

    @code_warntype Devices.diagonalize(DRESSED, device)
    @code_warntype Devices.diagonalize(OCCUPATION, device)
    @code_warntype Devices.diagonalize(OCCUPATION, device, q)

    @code_warntype Devices.basisrotation(DRESSED, OCCUPATION, device)
    @code_warntype Devices.basisrotation(OCCUPATION, OCCUPATION, device)
    @code_warntype Devices.basisrotation(OCCUPATION, OCCUPATION, device, q)
    @code_warntype Devices.localbasisrotations(OCCUPATION, OCCUPATION, device)

    @code_warntype Devices.eltype_algebra(device)
    @code_warntype Devices.eltype_algebra(device, DRESSED)
    @code_warntype Devices.algebra(device, DRESSED)
    @code_warntype Devices.localalgebra(device)

    ψ = zeros(ComplexF64, N); ψ[N] = 1
    ops = [
        IDENTITY,
        COUPLING,
        STATIC,
        Qubit(q),
        Channel(i,t),
        Drive(t),
        Hamiltonian(t),
        Gradient(j,t),
    ]

    println("Operator Methods")
    for op in ops
        println("> $op")

        @code_warntype Devices.eltype(op, device)
        @code_warntype Devices.eltype(op, device, DRESSED)

        @code_warntype Devices.operator(op, device)
        G = Devices.operator(op, device)
        @code_warntype Devices.operator(op, device; result=G)
        @code_warntype Devices.operator(op, device, DRESSED)
        H = Devices.operator(op, device, DRESSED)
        @code_warntype Devices.operator(op, device, DRESSED; result=H)

        @code_warntype Devices.propagator(op, device, τ)
        I = Devices.propagator(op, device, τ)
        @code_warntype Devices.propagator(op, device, τ; result=I)
        @code_warntype Devices.propagator(op, device, DRESSED, τ)
        J = Devices.propagator(op, device, DRESSED, τ)
        @code_warntype Devices.propagator(op, device, DRESSED, τ; result=J)

        @code_warntype Devices.propagate!(op, device, τ, ψ)
        @code_warntype Devices.propagate!(op, device, DRESSED, τ, ψ)

        if op isa StaticOperator
            @code_warntype Devices.evolver(op, device, τ)
            M = Devices.evolver(op, device, τ)
            @code_warntype Devices.evolver(op, device, τ; result=M)
            @code_warntype Devices.evolver(op, device, DRESSED, τ)
            N = Devices.evolver(op, device, DRESSED, τ)
            @code_warntype Devices.evolver(op, device, DRESSED, τ; result=N)

            @code_warntype Devices.evolve!(op, device, τ, ψ)
            @code_warntype Devices.evolve!(op, device, DRESSED, τ, ψ)
        end

        @code_warntype Devices.expectation(op, device, ψ)
        @code_warntype Devices.expectation(op, device, DRESSED, ψ)

        @code_warntype Devices.braket(op, device, ψ, ψ)
        @code_warntype Devices.braket(op, device, DRESSED, ψ, ψ)
    end


    println("Local Methods")
    @code_warntype Devices.localqubitoperators(device)
    Q = Devices.localqubitoperators(device)
    @code_warntype Devices.localqubitoperators(device; result=Q)
    @code_warntype Devices.localqubitoperators(device, OCCUPATION)
    R = Devices.localqubitoperators(device, OCCUPATION)
    @code_warntype Devices.localqubitoperators(device, OCCUPATION; result=R)

    @code_warntype Devices.localqubitpropagators(device, τ)
    S = Devices.localqubitpropagators(device, τ)
    @code_warntype Devices.localqubitpropagators(device, τ; result=S)
    @code_warntype Devices.localqubitpropagators(device, OCCUPATION, τ)
    T = Devices.localqubitpropagators(device, OCCUPATION, τ)
    @code_warntype Devices.localqubitpropagators(device, OCCUPATION, τ; result=T)

    @code_warntype Devices.localqubitevolvers(device, τ)
    U = Devices.localqubitevolvers(device, τ)
    @code_warntype Devices.localqubitevolvers(device, τ; result=U)
    @code_warntype Devices.localqubitevolvers(device, OCCUPATION, τ)
    V = Devices.localqubitevolvers(device, OCCUPATION, τ)
    @code_warntype Devices.localqubitevolvers(device, OCCUPATION, τ; result=V)

    return nothing
end

function check_types(device::LocallyDrivenDevices.LocallyDrivenDevice)
    invoke(check_types, Tuple{Devices.Device}, device)
    i = Devices.ndrives(device)
    j = Devices.ngrades(device)

    @code_warntype LocallyDrivenDevices.drivequbit(device, i)
    @code_warntype LocallyDrivenDevices.gradequbit(device, j)

    println("Local Drive Methods")
    @code_warntype LocallyDrivenDevices.localdriveoperators(device, t)
    W = LocallyDrivenDevices.localdriveoperators(device, t)
    @code_warntype LocallyDrivenDevices.localdriveoperators(device, t; result=W)
    @code_warntype LocallyDrivenDevices.localdriveoperators(device, OCCUPATION, t)
    X = LocallyDrivenDevices.localdriveoperators(device, OCCUPATION, t)
    @code_warntype LocallyDrivenDevices.localdriveoperators(device, OCCUPATION, t;
            result=X)

    @code_warntype LocallyDrivenDevices.localdrivepropagators(device, τ, t)
    Y = LocallyDrivenDevices.localdrivepropagators(device, τ, t)
    @code_warntype LocallyDrivenDevices.localdrivepropagators(device, τ, t; result=Y)
    @code_warntype LocallyDrivenDevices.localdrivepropagators(device, OCCUPATION, τ, t)
    Z = LocallyDrivenDevices.localdrivepropagators(device, OCCUPATION, τ, t)
    @code_warntype LocallyDrivenDevices.localdrivepropagators(device, OCCUPATION, τ, t;
            result=Z)

    return nothing
end





function check_types(signal::Signals.AbstractSignal{P,R}) where {P,R}

    # TEST PARAMETERS

    @code_warntype Parameters.count(signal); L = Parameters.count(signal)
    @code_warntype Parameters.values(signal); x̄ = Parameters.values(signal)
    @code_warntype Parameters.names(signal); names = Parameters.names(signal)
    @code_warntype Parameters.bind(signal, x̄)

    # TEST FUNCTION CONSISTENCY

    @code_warntype signal(t)
    @code_warntype signal(t̄); ft̄ = signal(t̄)
    @code_warntype signal(t̄; result=ft̄)

    # TEST GRADIENT CONSISTENCY

    @code_warntype Signals.partial(L, signal, t)
    @code_warntype Signals.partial(L, signal, t̄); gt̄ = Signals.partial(L, signal, t̄)
    @code_warntype Signals.partial(L, signal, t̄; result=gt̄)

    # CONVENIENCE FUNCTIONS

    @code_warntype string(signal, names)
    @code_warntype string(signal)

    @code_warntype Signals.integrate_partials(signal, τ̄, t̄, ϕ̄)
    Ip = Signals.integrate_partials(signal, τ̄, t̄, ϕ̄)
    @code_warntype Signals.integrate_partials(signal, τ̄, t̄, ϕ̄; result=Ip)

    @code_warntype Signals.integrate_signal(signal, τ̄, t̄, ϕ̄)

end
