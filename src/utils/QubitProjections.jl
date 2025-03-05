"""
    QubitProjections

Implement some handy tools for working with specifically two-level systems.

"""
module QubitProjections
    export qubitprojector, localqubitprojectors
    export qubitisometry, localqubitisometries
    export nqubits

    import ..CtrlVQE: LAT

    import TemporaryArrays: @temparray

    """
        nqubits(N, m=2)

    Infer the number of qubits from the number of states and levels per qubit.

    """
    function nqubits(N::Int, m::Int=2)
        return round(Int, log(m, N))
    end

    """
        nlevels(N, n)

    Infer the number of levels per qubit from the number of states and qubits.

    """
    function nlevels(N::Int, n::Int)
        return round(Int, N ^ (1/n))
    end

    """
        nstates(n, m=2)

    Infer the number of states from the number of qubits and levels per qubit.
    """
    function nstates(n::Int, m::Int=2)
        return m^n
    end

    """
        projector(n, m; m0=2, result)

    A projector from a system with `m` levels per qubit onto one with `m0` levels.

    The result is an (`m^n`⨯`m^n`) matrix,
        acting on a vector with `n` qubits of `m` levels each,
        producing a vector of the same size
        but where each qubit has support on only the first `m0` levels.

    If `result` is provided, results are written to `result` without further allocation.

    """
    function projector(n::Int, m::Int; m0::Int=2, result=nothing)
        N = nstates(n, m)
        isnothing(result) && (result = Array{Bool}(undef, N, N))

        π̄ = @temparray(Bool, (m,m,n), :projector)
        return LAT.kron(localprojectors(n, m; m0=m0, result=π̄); result=result)
    end

    """
        localprojectors(n, m; m0=2, result)

    A matrix list (aka a 3d array) of `n` local qubit projectors.

    Each projector is an (`m`⨯`m`) matrix,
        acting on a vector with `m` levels and producing a vector with the same size,
        but support on only `m0` levels.

    If `result` is provided, results are written to `result` without further allocation.

    """
    function localprojectors(n::Int, m::Int; m0::Int=2, result=nothing)
        isnothing(result) && (result = Array{Bool}(undef, m, m, n))
        result .= 0
        for q in 1:n
            for z in 1:m0
                result[z,z,q] = 1
            end
        end
        return result
    end

    """
        isometry(n, m; m0=2, result)

    An isometry lifting or reducing a system over `m0` levels to `m` levels.

    The result is an (`m^n`⨯`m0^n`) matrix,
        acting on a vector with `n` qubits of `m0` levels each,
        and producing a vector with `n` qubits of `m` levels each.

    If `result` is provided, results are written to `result` without further allocation.

    """
    function isometry(n::Int, m::Int; m0::Int=2, result=nothing)
        N  = nstates(n, m)
        N0 = nstates(n, m0)
        isnothing(result) && (result = Array{Bool}(undef, N, N0))

        π̄ = @temparray(Bool, (m,m0,n), :isometry)
        return LAT.kron(localisometries(n, m; m0=m0, result=π̄); result=result)
    end

    """
        localisometries(n, m; m0=2, result)

    A matrix list (aka a 3d array) of `n` local qubit isometries.

    Each isometry is an (`m`⨯`m0`) matrix,
        acting on a vector with `m0` levels and producing a vector with `m` levels.

    If `result` is provided, results are written to `result` without further allocation.

    """
    function localisometries(n::Int, m::Int; m0::Int=2, result=nothing)
        isnothing(result) && (result = Array{Bool}(undef, m, m0, n))
        result .= 0
        for q in 1:n
            for z in 1:min(m0,m)
                result[z,z,q] = 1
            end
        end
        return result
    end

    """
        isometrize(A, n, m; result)

    Lift or reduce a statevector or matrix for `n` qubits
        onto a physical Hilbert space with just `m` levels per qubit.

    If `result` is provided, results are written to `result` without further allocation.

    """
    function isometrize(A::AbstractVector{F}, n::Int, m::Int; result=nothing) where {F}
        N  = nstates(n, m)
        N0 = size(A,1)
        m0 = nlevels(N0, n)
        isnothing(result) && (result = zeros(F, N))

        msmall = min(m, m0)
        mlarge = max(m, m0)
        for i in 1:min(N,N0)
            I = mapindex(i, n, mlarge, msmall)
            m0 < m ? (result[I] = A[i]) : (result[i] = A[I])
        end
        return result
    end

    function isometrize(A::AbstractMatrix{F}, n::Int, m::Int; result=nothing) where {F}
        N  = nstates(n, m)
        N0 = size(A,1)
        m0 = nlevels(N0, n)
        isnothing(result) && (result = zeros(F, N))

        msmall = min(m, m0)
        mlarge = max(m, m0)
        for i in 1:min(N,N0)
            I = mapindex(i, n, mlarge, msmall)
            for j in 1:min(N,N)
                J = mapindex(j, n, mlarge, msmall)
                m0 < m ? (result[I,J] = A[i,j]) : (result[i,j] = A[I,J])
            end
        end
        return result
    end

    """
        mapindex(i0, n, m, m0=2)

    Given an index `i0` defined for vectors of `n` qubits with `m0` levels each,
        compute the corresponding index in a space with `m` levels each.

    This function assumes `m > m0`. Otherwise, there may not *be* a corresponding index!

    """
    function mapindex(i0::Int, n::Int, m::Int, m0::Int=2)
        p = @temparray(Int, n, :mapindex)
        digits!(p, i0-1; base=m0)
        return 1 + evalpoly(m, p)
    end

end
