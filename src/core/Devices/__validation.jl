import ..CtrlVQE: Validation
import ..CtrlVQE.Validation: @withresult

import ..CtrlVQE: Parameters

import ..CtrlVQE.LinearAlgebraTools as LAT

import ..CtrlVQE.Bases: BARE, DRESSED
import CtrlVQE.Operators: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE.Operators: Qubit, Channel, Drive, Hamiltonian, Gradient

import LinearAlgebra: Diagonal

function Validation.validate(
    device::DeviceType{F};
    grid=nothing,
    t=zero(F),
) where {F}
    Parameters.validate(device)

    # CHECK TYPING FUNCTIONS
    F_ = eltype(device);                        @assert F == F_

    # CHECK NUMBER FUNCTIONS
    nD = ndrives(device);                       @assert nD isa Int
    nG = ngrades(device);                       @assert nG isa Int
    nO = noperators(device);                    @assert nO isa Int
    m = nlevels(device);                        @assert m isa Int
    n = nqubits(device);                        @assert n isa Int
    N = nstates(device);                        @assert N isa Int

    # CHECK BASIS AND ROTATIONS
    I = LAT.basisvectors(N)
    Λ, U = dress(device);                           @assert U*U' ≈ I
    U_BB = basisrotation(BARE, BARE, device);       @assert U_BB ≈ I
    U_DD = basisrotation(DRESSED, DRESSED, device); @assert U_DD ≈ I
    U_BD = basisrotation(BARE, DRESSED, device);    @assert U_BD ≈ U
    U_DB = basisrotation(DRESSED, BARE, device);    @assert U_DB ≈ U'

    ######################################################################################
    #= CHECK ABSTRACT INTERFACE =#

    # ALGEBRA METHODS
    ā = @withresult globalalgebra(device);      @assert size(ā) == (N,N,nO,n)
    ā0 = @withresult localalgebra(device);      @assert size(ā0) == (m,m,nO,n)
    for o in 1:nO
        for q in 1:n
            @assert ā[:,:,o,q] ≈ @withresult globalize(device, @view(ā0[:,:,o,q]), q)
        end
    end

    # MODEL METHODS
    h̄ = [@withresult qubithamiltonian(device, ā, q) for q in 1:n];
        @assert all(h ≈ h' for h in h̄)
    h̄0 = [qubithamiltonian(device, ā0, q) for q in 1:n]
    for q in 1:n
        @assert h̄[q] ≈ globalize(device, h̄0[q], q)
    end

    G = @withresult staticcoupling(device, ā);
        @assert G ≈ G'
    v̄ = [@withresult driveoperator(device, ā, i, t) for i in 1:nD]
        @assert all(v ≈ v' for v in v̄)
    Ā = [@withresult gradeoperator(device, ā, j, t) for j in 1:nG]
        @assert all(A ≈ A' for A in Ā)

    # CHECK DRESSED BASIS
    H0 = sum(h̄) .+ G
        @assert U * Diagonal(Λ) * U' ≈ H0

    ######################################################################################
    #= CHECK OPERATOR FUNCTIONS =#

    # SET DEFAULT EVOLVABLES
    ψ = convert(Vector{Complex{F}}, LAT.basisvector(N,1))
    ρ = ψ * ψ'
    τ = one(F)

    # BASIC FUNCTIONALITY
    for op in [
        IDENTITY, COUPLING, STATIC,
        Qubit(n), Channel(nD,t), Gradient(nG,t),
        Drive(t), Hamiltonian(t),
    ]
        H = @withresult operator(op, device)
        U = @withresult propagator(op, device, τ)
            @assert U ≈ cis((-τ) .* H)
        ψ_ = deepcopy(ψ); propagate!(op, device, τ, ψ_)
            @assert ψ_ ≈ U * ψ
        ρ_ = deepcopy(ρ); propagate!(op, device, τ, ρ_)
            @assert ρ_ ≈ U * ρ * U'
        E = expectation(op, device, ψ)
            @assert abs(E - (ψ'*H*ψ)) < 1e-10
        M = braket(op, device, ψ, ψ)
            @assert abs(M - (ψ'*H*ψ)) < 1e-10

        if op isa StaticOperator
            Uτ = @withresult evolver(op, device, τ)
                @assert U ≈ Uτ
            ψ_ = deepcopy(ψ); evolve!(op, device, τ, ψ_)
                @assert ψ_ ≈ U * ψ
            ρ_ = deepcopy(ρ); evolve!(op, device, τ, ρ_)
                @assert ρ_ ≈ U * ρ * U'
        end
    end

    # OPERATOR CONSISTENCY
    @assert operator(IDENTITY, device) ≈ I
    @assert operator(COUPLING, device) ≈ G
    @assert operator(STATIC, device) ≈ H0
    @assert operator(Qubit(n), device) ≈ h̄[n]
    @assert operator(Channel(nD,t), device) ≈ v̄[nD]
    @assert operator(Gradient(nG,t), device) ≈ Ā[nG]
    @assert operator(Drive(t), device) ≈ sum(v̄)
    @assert operator(Hamiltonian(t), device) ≈ sum(v̄) .+ H0

    # CHECK ONE ALTERNATIVE BASIS FOR GOOD MEASURE
    @assert operator(STATIC, device, DRESSED) ≈ Diagonal(Λ)

    # LOCAL OPERATORS
    h̄L = @withresult localqubitoperators(device)
        @assert all(h̄L[:,:,q] ≈ h̄0[q] for q in 1:n)
    ūL = @withresult localqubitpropagators(device, τ)
        @assert ūL ≈ @withresult localqubitevolvers(device, τ)
        @assert all(ūL[:,:,q] ≈ cis((-τ).*h̄L[:,:,q]) for q in 1:n)

    if !isnothing(grid)
        ϕ = ones(F, length(grid), nG)
        @withresult gradient(device, grid, ϕ)
        # Just make sure it can be called. Accuracy must be checked with a cost function.
    end
end