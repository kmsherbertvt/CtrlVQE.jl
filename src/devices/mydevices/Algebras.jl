abstract type AlgebraType{F} end






struct Bosonic{F} <: AlgebraType{F}
    m::Int      # Truncation level, ie. # of levels per qubit.
end

Devices.nlevels(algebra::Bosonic) = algebra.m

Devices.eltype_localloweringoperator(::Bosonic{F}) where {F} = F

@memoize Dict function _cachedloweringoperator(algebra::Bosonic{F}) where {F}
    result = Matrix{F}(undef, algebra.m, algebra.m)
    return Devices.localloweringoperator(algebra; result=result)
end

function Devices.localloweringoperator(algebra::Bosonic; result=nothing)
    isnothing(result) && return _cachedloweringoperator(algebra)
    result .= 0

    for i in 1:algebra.m-1
        result[i,i+1] = √i
    end
    return result
end

function Devices.localalgebra(algebra::Bosonic{F}) where {F}
    ā = reshape(_cachedloweringoperator(algebra), (algebra.m, algebra.m, 1))
end







#= TODO:

Spinor, which has localloweringoperator (X+iY),
    but more importantly algebra and localalgebra giving Pauli basis
    rather than each a[q].

=#