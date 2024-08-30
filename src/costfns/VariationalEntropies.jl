#= Making up words is fun. -_-

Define a probability distribution ρ[i] = |x[i]|² / ∑|x[j]|² from variational parameters x.
Define the variational entropy as the Shannon entropy of ρ.

In some sense, this describes the amount of information needed to ...
    ... write down these parameters, I guess.
Not really sure how to formalize that.

But, a lower entropy is associated with a sparser parameter set,
    so including the variational entropy as a penalty function
    is a way to encourage optimization to find sparser solutions.

=#

import ..CostFunctions
export VariationalEntropy

import ..Parameters, ..Devices

safelog(p) = p == 0 ? zero(p) : log(p)  # Use ONLY when multiplying log(p) BY p.
entropyunit(p) = -p * safelog(p)        # A single term of Shannon entropy.


squared(x) = x^2
partition(x̄) = sum(squared, x̄)


"""
    VariationalEntropy(device)

Compute the variational entropy of the input parameters.

Note that the calculation does not depend on `device` -
    it only serves to fix the type and number of parameters
    (but in a dynamic way, if parameters are added to the device).
As usual, if parameters are added to the device,
    the functions will need to be regenerated.

"""
struct VariationalEntropy{F} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F}
end

Base.length(fn::VariationalEntropy) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::VariationalEntropy{F}) where {F}
    ρ̄ = Array{F}(undef, length(fn))
    return (x̄) -> (
        S = zero(F);
        Z = partition(x̄);
        if Z > 0;
            ρ̄ .= x̄ .^ 2 ./ Z;
            for i in 1:length(fn);
                S += -ρ̄[i] * safelog(ρ̄[i]);
            end;
        end;
        S
    )
end

function CostFunctions.grad_function_inplace(fn::VariationalEntropy{F}) where {F}
    entropy = CostFunctions.cost_function(fn)
    ρ̄ = Array{F}(undef, length(fn))
    return (∇f̄, x̄) -> (
        ∇f̄ .= 0;
        Z = partition(x̄);
        if Z > 0;
            ρ̄ .= x̄ .^ 2 ./ Z;
            S = entropy(x̄);
            for i in 1:length(fn);
                ∇f̄[i] = -2 * x̄[i] / Z * (S + safelog(ρ̄[i]))
            end;
        end;
        ∇f̄
    )
end




#= Solving the singularity with a smooth transition function... =#

exparg(u) = (2u-1) / (u^2 - u)
tran(u) = u == 0 ? zero(u) : (1 + exp(exparg(u))^(-1))
gradtran(u) = u == 0 ? zero(u) : (1 + cosh(exparg(u)))^(-1) * (2*(u^2-u)-1)/(u^2-u)^2 / 2

struct SmoothVariationalEntropy{F} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F}
    σ::F
end

Base.length(fn::SmoothVariationalEntropy) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::SmoothVariationalEntropy{F}) where {F}
    ζ = length(fn) * fn.σ^2
    vanilla = VariationalEntropy(fn.device)
    entropy = CostFunctions.cost_function(vanilla)
    return (x̄) -> (
        S = entropy(x̄);
        Z = partition(x̄);
        T = tran(ζ * Z);
        S*T
    )
end

function CostFunctions.grad_function_inplace(fn::SmoothVariationalEntropy{F}) where {F}
    ζ = length(fn) * fn.σ^2
    vanilla = VariationalEntropy(fn.device)
    entropy = CostFunctions.cost_function(vanilla)
    gradentropy! = CostFunctions.grad_function_inplace(vanilla)
    return (∇f̄, x̄) -> (
        gradentropy!(∇f̄, x̄);
        S = entropy(x̄);
        Z = partition(x̄);
        T = tran(ζ * Z);
        ∇f̄ .*= T;
        ∇f̄ .+= x̄ .* (2 * S * ζ * gradtran(ζ*Z));
        ∇f̄
    )
end

#= TODO: Normalize by parameter count..?
Don't love this 'cause the entropy properly defined does have upper bound log(L)...
=#