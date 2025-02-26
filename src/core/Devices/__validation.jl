using .Devices: DeviceType

import ..CtrlVQE: Validation

function Validation.validate(device::DeviceType{F}; grid=nothing) where {F}
    # CHECK TYPING FUNCTIONS
    F_ = eltype(device);                        @assert F == F_

    # `Parameters` INTERFACE
    L = Parameters.count(device);               @assert L isa Int
    x = Parameters.values(device);              @assert x isa Vector{F}
    names = Parameters.names(device);           @assert names isa Vector{String}
    Parameters.bind!(device, x);    # No check. Just make sure it doesn't error.

    # CHECK NUMBER FUNCTIONS
    nD = ndrives(device);                       @assert N isa Int
    nG = ngrades(device);                       @assert N isa Int
    nO = noperators(device);                    @assert N isa Int
    m = nlevels(device);                        @assert N isa Int
    n = nqubits(device);                        @assert N isa Int
    N = nstates(device);                        @assert N isa Int

    # CHECK ABSTRACT INTERFACE
    ā = localalgebra(device);                   @assert size(ā) == (m,m,nO,n)
    ā_ = similar(ā)
    ā__ = localalgebra(device; result=ā_)
        @assert ā_ == ā
        @assert ā_ === ā__

    # TODO: Sorry, we need the global algebra here first...
    # TODO: We really need to fetch the suite of checks from the old code.

    h = qubithamiltonian(device, ā, 1)          @assert size(h) == (m,m)
    h_ = similar(h)
    h__ = qubithamiltonian(device, ā, 1; result=h_)
        @assert h_ == h
        @assert h_ === h__

    G = qubithamiltonian(device, ā, 1)          @assert size(G) == (N,N)
    G_ = similar(G)
    G__ = qubithamiltonian(device, ā, 1; result=G_)
        @assert G_ == G
        @assert G_ === G__

    #=
        export localalgebra
        export qubithamiltonian, staticcoupling, driveoperator, gradeoperator

        export globalalgebra
        export globalize, dress, basisrotation

        export operator, localqubitoperators
        export propagator, localqubitpropagators, propagate!
        export evolver, localqubitevolvers, evolve!
        export expectation, braket
    =#

    if !isnothing(grid)
        # gradient
    end
end