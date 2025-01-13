"""
    Integrations

Everything you need to know how to integrate over time.

"""
module Integrations
    include("Integrations/__abstractinterface.jl")
        export IntegrationType
        export nsteps
        export timeat, stepat
        export Prototype
    include("Integrations/__concreteinterface.jl")
        export starttime, endtime, duration, stepsize
        export lattice
        export integrate
        # Also implements AbstractVector interface
    include("Integrations/__validation.jl")
end