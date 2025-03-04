"""
    QubitOperations

Implement some handy tools for working with specifically two-level systems.

TODO: Optional type arg (defaults to Bool).
TODO: kwargs for m0, defaults to 2.
TODO: result kwarg

"""
module QubitOperations
    export qubitprojector, localqubitprojectors
    export qubitisometry, localqubitisometries
    export nqubits

    import ..CtrlVQE: LAT

    import TemporaryArrays: @temparray

    """
        qubitprojector(m::Int, n::Int)

    A projector from the physical Hilbert space of a device onto a logical two-level space.

    The projector acts on a vector with m-level qubits,
        produces a vector with m-level qubits but support only on the lowest two levels.

    # Parameters
    - `m`: Number of levels to project each qubit from.
    - `n`: Number of qubits.

    """
    function qubitprojector(m::Int, n::Int)
        return LAT.kron(localqubitprojectors(m,n))
    end


    """
        localqubitprojectors(m::Int, n::Int)

    A matrix list of local qubit projectors for each individual qubit in the device.

    The projector acts on a vector with m-level qubits,
        produces a vector with m-level qubits but support only on the lowest two levels.

    # Parameters
    - `m`: Number of levels to project each qubit from.
    - `n`: Number of qubits.

    """
    function localqubitprojectors(m::Int, n::Int)
        π = @temparray(Bool, (m,m), :localqubitisometries)
        LAT.basisvectors!(π)
        for l in 3:m
            π[l,l] = 0
        end
        π̄ = Array{Bool}(undef, m, m, n)
        for q in 1:n
            π̄[:,:,q] .= π
        end
        return π̄
    end

    """
        qubitisometry(m::Int, n::Int)

    An isometry from the physical Hilbert space of a device onto a logical two-level space.

    The isometry acts on a vector with two-level qubits,
        and produces a vector with m-level qubits.

    # Parameters
    - `m`: Number of levels to project each qubit onto.
    - `n`: Number of qubits.

    """
    function qubitisometry(m::Int, n::Int)
        # NOTE: Acts on qubit space, projects up to device space.
        return LAT.kron(localqubitisometries(m,n))
    end

    """
        localqubitisometries(m::Int, n::Int)

    A matrix list of local qubit isometries for each individual qubit in the device.

    The isometry acts on a vector with two-level qubits,
        and produces a vector with m-level qubits.

    # Parameters
    - `m`: Number of levels to project each qubit onto.
    - `n`: Number of qubits.

    """
    function localqubitisometries(m::Int, n::Int)
        ϕ = @temparray(Bool, (m,2), :localqubitisometries)
        LAT.basisvectors!(ϕ)

        ϕ̄ = Array{Bool}(undef, m, 2, n)
        for q in 1:n
            ϕ̄[:,:,q] .= ϕ
        end
        return ϕ̄
    end

    """
        nqubits(N)

    Infer the number of qubits from the number of states, assuming a two-level system..

    """
    nqubits(N::Int) = round(Int, log2(N))

    """
        project(A, m::Int, n::Int)

    Extend a statevector or matrix living in a two-level space
        onto a physical Hilbert space with `m` levels per qudit on `n` qudits.

    """
    function project(ψ::AbstractVector{F}, m::Int, n::Int) where {F}
        N0 = length(ψ)
        N = m^n

        m̄0 = fill(2, n)
        m̄  = fill(m, n)

        ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
        result = zeros(F, N)
        for i in 1:N0
            result[ix_map[i]] = ψ[i]
        end
        return result
    end

    function project(H::AbstractMatrix{F}, m::Int, n::Int) where {F}
        N0 = size(H, 1)
        N = m^n

        m̄0 = fill(2, n)
        m̄  = fill(m, n)

        ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
        result = zeros(F, N, N)
        for i in 1:N0
            for j in 1:N0
                result[ix_map[i],ix_map[j]] = H[i,j]
            end
        end
        return result
    end



    function _cd_from_ix(i::Int, m̄::AbstractVector{<:Integer})
        i = i - 1       # SWITCH TO INDEXING FROM 0
        ī = Vector{Int}(undef, length(m̄))
        for q in eachindex(m̄)
            i, ī[q] = divrem(i, m̄[q])
        end
        return ī
    end

    function _ix_from_cd(ī::AbstractVector{<:Integer}, m̄::AbstractVector{<:Integer})
        i = 0
        offset = 1
        for q in eachindex(m̄)
            i += offset * ī[q]
            offset *= m̄[q]
        end
        return i + 1    # SWITCH TO INDEXING FROM 1
    end

end