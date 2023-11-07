export CostFunctionType, CompositeCostFunction, EnergyFunction
export cost_function, grad_function, grad_function_inplace
export trajectory_callback

import LinearAlgebra

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    CostFunctionType{F}

Super-type for "cost functions", to be plugged directly into optimization algorithms.

# Implementation

Any concrete sub-type `CF` must implement the following methods:
- `Base.length(::CF)`:
        the number of parameters this cost function takes
- `cost_function(::CF)`:
        returns a `Function` which takes a parameter vector
            and returns the output of the cost function
        ie. a callable expression (x::Vector) -> f(x)
- `grad_function_inplace(::CF)`:
        returns a mutating `Function` which takes a gradient vector (to be mutated)
            and a parameter vector, and writes the gradient vector to the first argument.
        As a matter of habit, the resulting gradient vector should also be returned.

If your cost function involves calculating the expectation value of a time-evolved state,
    you should implement an `EnergyFunction` (even if it isn't strictly an energy).
This type has a couple extra requirements to allow energy trajectories over the evolution.

"""
abstract type CostFunctionType{F} end

"""
    Base.length(fn::CostFunctionType)

Gives the number of parameters for this cost function.
"""
function Base.length(fn::CostFunctionType)
    error("Not Implemented")
    return 0
end

"""
    cost_function(fn::CostFunctionType)

Converts the struct `fn` into a literal `Function` of a parameter vector.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the value of `fn` at the point `x̄`.

"""
function cost_function(fn::CostFunctionType{F}) where {F}
    error("Not Implemented")
    return (x̄) -> zero(F)
end

"""
    grad_function_inplace(fn::CostFunctionType)

Constructs a (mutating) `Function` to calculate the gradient of `fn` at a particular point.

The function accepts a gradient vector `∇f̄` (to be mutated) and a parameter vector `x̄`.
Both should have the type and length given by `eltype(fn)` and `length(fn)`.
After the function is called, `∇f̄` contains the gradient of `fn` at the point `x̄`.

"""
function grad_function_inplace(fn::CostFunctionType{F}) where {F}
    error("Not Implemented")
    return (∇f̄, x̄) -> Vector{F}(undef, length(fn))
end



# IMPLEMENTED INTERFACE

"""
    Base.eltype(fn::CostFunctionType)

Gives the number type for parameters of this cost function.
"""
function Base.eltype(::CostFunctionType{F}) where {F}
    return F
end

"""
    (fn::CostFunctionType)(x̄::AbstractVector)

Evaluate the value of `fn` at the point `x̄`.

This is syntactic sugar for constructing a dedicated `cost_function` and calling it.
It will not normally take advantage of cached work variables,
    so avoid using it in high-performance code.

"""
function (fn::CostFunctionType)(x̄::AbstractVector)
    return cost_function(fn)(x̄)
end

"""
    grad_function(fn::CostFunctionType)

Constructs a `Function` to calculate the gradient of `fn` at a particular point.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the gradient (a vector) of `fn` at the point `x̄`.

"""
function grad_function(fn::CostFunctionType{F}; kwargs...) where {F}
    g! = grad_function_inplace(fn; kwargs...)
    ∇f̄ = Vector{F}(undef, length(fn))
    return (x̄) -> (g!(∇f̄, x̄); ∇f̄)
end





"""
    CompositeCostFunction(components::AbstractVector{CostFunctionType})

The sum of several cost-functions.

Use this eg. to combine an energy function with one or more penalty functions.

Note that all components must have matching `eltype` and `length`.

As a practical convenience, this struct maintains some dynamic attributes
    so that results for each component function can be inspected
    in the middle of an optimization.
I don't really consider them an official part of the interface,
    but feel free to use them. Although...they aren't technically "thread-safe". ^_^

"""
struct CompositeCostFunction{F} <: CostFunctionType{F}
    components::Vector{CostFunctionType{F}}

    L::Int                  # NUMBER OF PARAMETERS

    f_counter::Ref{Int}     # NUMBER OF TIMES ANY COST FUNCTION IS CALLED
    g_counter::Ref{Int}     # NUMBER OF TIMES ANY GRAD FUNCTION IS CALLED
    values::Vector{F}       # STORES VALUES OF EACH COMPONENT FROM THE LAST CALL
    gnorms::Vector{F}       # STORES NORMS OF EACH GRADIENT FROM THE LAST CALL

    function CompositeCostFunction(
        components::AbstractVector{CostFunctionType{F}},
    ) where {F}
        if length(components) == 0
            return new{F}(CostFunctionType{F}[], 0, Ref(0), Ref(0), F[], F[])
        end

        L = length(first(components))
        for component in components; @assert length(component) == L; end

        return new{F}(
            convert(Vector{CostFunctionType{F}}, components),
            L, Ref(0), Ref(0),
            Vector{F}(undef, L),
            Vector{F}(undef, L),
        )
    end
end

"""
    CompositeCostFunction(f̄::CostFunctionType{F}...)

Alternate constructor, letting each function be passed as its own argument.

"""
function CompositeCostFunction(components::CostFunctionType{F}...) where {F}
    return CompositeCostFunction(CostFunctionType{F}[component for component in components])
end

Base.length(fn::CompositeCostFunction) = fn.L

function cost_function(fn::CompositeCostFunction)
    fs = [cost_function(component) for component in fn.components]
    return (x̄) -> (
        fn.f_counter[] += 1;
        total = 0;
        for (i, f) in enumerate(fs);
            fx = f(x̄);          # EVALUATE EACH COMPONENT
            fn.values[i] = fx;
            total += fx;        # TALLY RESULTS FOR THE COMPONENT
        end;
        total
    )
end

function grad_function_inplace(fn::CompositeCostFunction{F}) where {F}
    g!s = [grad_function_inplace(component) for component in fn.components]
    ∇f̄_ = zeros(F, fn.L)        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
    return (∇f̄, x̄) -> (
        fn.g_counter[] += 1;
        ∇f̄ .= 0;
        for (i, g!) in enumerate(g!s);
            g!(∇f̄_, x̄);         # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∇f̄_
            fn.gnorms[i] = LinearAlgebra.norm(∇f̄_);
            ∇f̄ .+= ∇f̄_;         # TALLY RESULTS FOR THE COMPONENT
        end;
        ∇f̄;
    )
end


"""
    EnergyFunction{F}

Super-type for cost functions calculating the expectation value of a time-evolved state.

# Implementation

Any concrete sub-type `CF` must implement
    *everything* required in the `CostFunctionType` interface,
    so consult the documentation for `CostFunctionType` carefully.

In additon, the following method must be implemented:
- `trajectory_callback(::CF, E::AbstractVector; callback=nothing)`
        returns a `Function` compatible with Evolutions.evolve callback
            ie. a callable expression (i::Int, t::Real, ψ::Vector) -> Nothing
        which sets E[i] to the energy of a partially evolved wavefunction ψ.
        If `callback` is provided, the function calls that `callback` afterwards.

Finally, the following methods must now accept a keyword argument:
- `cost_function(::CF; callback=nothing)`:
        When `callback` is provided, the time evolution must call it at each timestep.

- `grad_function_inplace(::CF; ϕ=nothing)`:
        When `ϕ` is provided, write the gradient signals to it.

"""
abstract type EnergyFunction{F} <: CostFunctionType{F} end

function trajectory_callback(fn::EnergyFunction, E::AbstractVector; callback=nothing)
    error("Not Implemented")
    return (i, t, ψ) -> nothing
end

"""
    cost_function(fn::CostFunctionType; callback=nothing)

Same as for `CostFunctionType` except that whenever the function is called,
    the time evolution calls `callback` (if provided) in each time step.

"""
function cost_function(fn::EnergyFunction{F}; callback=nothing) where {F}
    error("Not Implemented")
    return (x̄) -> zero(F)
end

"""
    grad_function_inplace(fn::CostFunctionType; ϕ=nothing)

Same as for `CostFunctionType` except that whenever the function is called,
    ϕ (if provided) is updated to contain the gradient signals.

"""
function grad_function_inplace(fn::EnergyFunction{F}; ϕ=nothing) where {F}
    error("Not Implemented")
    return (∇f̄, x̄) -> Vector{F}(undef, length(fn))
end




"""
    ConstrainedEnergyFunction(E, Λ̄, λ̄)

An energy function modified by any number of penalty terms.

# Parameters
- `E`: the base `EnergyFunction`
- `Λ̄`: a vector of cost functions, each matching the `eltype` and `length` of `E`
- `λ̄`: a vector of Lagrange multipliers, matching the length of `Λ̄`

NOTE: If your penalties include an energy function,
    (eg. calculating (⟨N⟩-N0)², which calculates an "energy" of the `N̂` operator),
    this function is usable, but will repeat the time-evolution unnecessarily.

TODO (lo): Write yet another cost function,
        which has a list of all the observables to estimate,
        and a (generic?) function of those observables to report as the loss.
    I think this waits 'til we have symbolic observables (ie. not dense matrices).

As a practical convenience, this struct maintains some dynamic attributes
    so that results for each component function can be inspected
    in the middle of an optimization.
I don't really consider them an official part of the interface,
    but feel free to use them. Although...they aren't technically "thread-safe". ^_^

"""
struct ConstrainedEnergyFunction{F} <: EnergyFunction{F}
    energyfn::EnergyFunction{F}
    penaltyfns::Vector{CostFunctionType{F}}
    weights::Vector{F}

    L::Int                  # NUMBER OF PARAMETERS
    nΛ::Int                 # NUMBER OF PENALTY FUNCTIONS

    f_counter::Ref{Int}     # NUMBER OF TIMES ANY COST FUNCTION IS CALLED
    g_counter::Ref{Int}     # NUMBER OF TIMES ANY GRAD FUNCTION IS CALLED
    energy::Ref{F}          # STORES VALUE OF ENERGY FROM THE LAST CALL
    energygd::Ref{F}        # STORES NORM OF ENERGY GRADIENT FROM THE LAST CALL
    penalties::Vector{F}    # STORES VALUE OF EACH PENALTY FROM THE LAST CALL
    penaltygds::Vector{F}   # STORES NORMS OF EACH PENALTY GRADIENT FROM THE LAST CALL

    function ConstrainedEnergyFunction(
        energyfn::EnergyFunction{F},
        penaltyfns::AbstractVector{CostFunctionType{F}},
        weights::AbstractVector{<:Real},
    ) where {F}
        # NUMBER OF PARAMETERS
        L = length(energyfn)
        for penaltyfn in penaltyfns; @assert length(penaltyfn) == L; end

        # NUMBER OF PENALTIES
        nΛ = length(penaltyfns)
        @assert length(weights) == nΛ

        return new{F}(
            energyfn,
            convert(Vector{CostFunctionType{F}}, penaltyfns),
            convert(Vector{F}, weights),
            L, nΛ,
            Ref(0), Ref(0),
            Ref(zero(F)), Ref(zero(F)),
            Vector{F}(undef, nΛ),
            Vector{F}(undef, nΛ),
        )
    end
end

"""
    ConstrainedEnergyFunction(E::EnergyFunction, Λ̄::CostFunctionType{F}...)

Alternate constructor, letting each penalty function be passed as its own argument.

Using this constructor, weights always default to λ=1 for each penalty function.

"""
function ConstrainedEnergyFunction(
    energyfn::EnergyFunction,
    penaltyfns::CostFunctionType{F}...
) where {F}
    return ConstrainedEnergyFunction(
        energyfn,
        CostFunctionType{F}[penaltyfn for penaltyfn in penaltyfns],
        ones(F, length(penaltyfns)),
    )
end

Base.length(fn::ConstrainedEnergyFunction) = fn.L

function cost_function(fn::ConstrainedEnergyFunction; callback=nothing)
    ef = cost_function(fn.energyfn; callback=callback)
    fs = [cost_function(penaltyfn) for penaltyfn in fn.penaltyfns]
    return (x̄) -> (
        fn.f_counter[] += 1;
        fn.energy[] = ef(x̄);
        total = fn.energy[];
        for (i, f) in enumerate(fs);
            fx = f(x̄);          # EVALUATE EACH COMPONENT
            fn.penalties[i] = fx;
            total += fn.weights[i] * fx;    # TALLY RESULTS FOR THE COMPONENT
        end;
        total
    )
end

function grad_function_inplace(fn::ConstrainedEnergyFunction{F}; ϕ=nothing) where {F}
    eg! = grad_function_inplace(fn.energyfn; ϕ=ϕ)
    g!s = [grad_function_inplace(penaltyfn) for penaltyfn in fn.penaltyfns]
    ∇f̄_ = zeros(F, fn.L)        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
    return (∇f̄, x̄) -> (
        fn.g_counter[] += 1;
        eg!(∇f̄_, x̄);
        fn.energygd[] = LinearAlgebra.norm(∇f̄_);
        ∇f̄ .= ∇f̄_;
        for (i, g!) in enumerate(g!s);
            g!(∇f̄_, x̄);         # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∇f̄_
            fn.penaltygds[i] = LinearAlgebra.norm(∇f̄_);
            ∇f̄ .+= fn.weights[i] .* ∇f̄_;    # TALLY RESULTS FOR THE COMPONENT
        end;
        ∇f̄;
    )
end

function trajectory_callback(
    fn::ConstrainedEnergyFunction,
    E::AbstractVector;
    callback=nothing,
)
    return trajectory_callback(fn.energyfn, E; callback=callback)
end