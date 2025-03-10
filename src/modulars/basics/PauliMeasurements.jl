module PauliMeasurements
    import CtrlVQE.ModularFramework as Modular
    import CtrlVQE.ModularFramework: AlgebraType, MeasurementType
    import CtrlVQE.ModularFramework: PauliAlgebra

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: QubitProjections
    import CtrlVQE: Bases, Operators
    import CtrlVQE: Integrations, Devices, CostFunctions

    import TemporaryArrays: @temparray

    import LinearAlgebra

    #= TODO:
    For both reference and measurements,
        I think perhaps we can relegate `basis` and `frame` to type parameters,
        and all the basis rotations happen by default.
    Similar to evolutions, users then need only implement a version where you assume that's already happened.

    Lol they already ARE type parameters in each implementation, so yeah.
    The only problem would be if we think there is any chance ever
        that it is useful to make a non-singleton basis or frame type.
    Since, the basis rotations in the default would necessarily pass `B()` or `O()`.
    I suppose we can require implementations for such types to override the defaults...

    If t is omitted, maybe omit the frame rotation entirely?

    Also btw constructors should probably be unchanged?
    Easier to do PauliMeasurement(..., BARE, STATIC) than PauliMeasurement{Bare,Static}(...)
    But I suppose it's okay to flip the order.

    =#

    """
        PauliMeasurement(X, Z, c, basis, frame)

    Represents a bare measurement of a linear combination of Paulis,
        without any logical projection prior to frame rotation.

    Paulis are encoded symplectically, as two length-`n` bitstrings labeled `x` and `z`.
    1s in `x` indicate a Pauli X appears on that bit; 1s in `z` indicate the same for Z.
    When a bit has both an X and a Z, it is interpreted as having a Y.
    (You could think of it as Z*X = iY, except the implementation here ignores phase.)

    # Parameters
    - `X`: a vector of integers whose bitstrings give each `x`.
    - `Z`: a vector of integers whose bitstrings give each `z`.
    - `c`: a vector of floats giving the coefficients for each pauli word.
    - `basis`: the `BasisType` identifying the basis `observable` is written in.
    - `frame`: the `OperatorType` identifying the frame where measurements are conducted.

    """
    struct PauliMeasurement{
        F,  # MAY BE ANY NUMBER TYPE, REAL OR COMPLEX
        B <: Bases.BasisType,
        O <: Operators.StaticOperator,
    } <: MeasurementType
        X::Vector{Int}
        Z::Vector{Int}
        c::Vector{F}
        basis::B
        frame::O

        function PauliMeasurement(
            X::AbstractVector{Int},
            Z::AbstractVector{Int},
            c::AbstractVector{F},
            basis::B,
            frame::O,
        ) where {F,B,O}
            n = length(c)
            n == length(X) == length(Z) || error("Vector lengths must match")
            return new{F,B,O}(
                convert(Array, X),
                convert(Array, Z),
                convert(Array, c),
                basis,
                frame,
            )
        end
    end

    """
        PauliMeasurement(basis, frame; paulis...)

    A constructor accepting coefficients for each Pauli word.

    # Parameters
    - `basis`: the `BasisType` identifying the basis `observable` is written in.
    - `frame`: the `OperatorType` identifying the frame where measurements are conducted.

    # Keyword Arguments
    - Each key is a consistent-length string consisting only of "I", "X", "Y", "Z".
    - Each value is the float coefficient for that Pauli word.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> I = [1 0; 0 1]; X = [0 1; 1 0]; Z = [1 0; 0 -1];

    julia> observable = (0.5 .* kron(X,Z)) .+ (1.0 .* kron(I, Z))
    4×4 Matrix{Float64}:
     1.0   0.0  0.5   0.0
     0.0  -1.0  0.0  -0.5
     0.5   0.0  1.0   0.0
     0.0  -0.5  0.0  -1.0

    julia> measurement = PauliMeasurement(Bases.BARE, Operators.STATIC; XZ=0.5, IZ=1.0);

    julia> device = Prototype(LocalDevice{Float64}; n=2);

    julia> validate(measurement; device=device);

    julia> Ō = observables(measurement, device, Bases.BARE, 0.0);

    julia> size(Ō)
    (4, 4, 1)

    julia> observable ≈ reshape(Ō, (4,4))
    true
    ```

    """
    function PauliMeasurement(
        basis::Bases.BasisType,
        frame::Operators.StaticOperator;
        kwargs...
    )
        K = length(kwargs)
        X = zeros(Int, K)
        Z = zeros(Int, K)
        labels = [String(key) for key in keys(kwargs)]
        c = [value for value in values(kwargs)]

        n = K > 0 ? length(first(labels)) : 0
        for (k, label) in enumerate(labels)
            length(label) == n || error("All Paulis must have same length.")
            for q in 1:n
                mask = 1 << (n-q)
                if     label[q] == 'I'
                    # NOTHING TO DO
                elseif label[q] == 'X'
                    X[k] += mask
                elseif label[q] == 'Y'
                    X[k] += mask
                    Z[k] += mask
                elseif label[q] == 'Z'
                    Z[k] += mask
                else
                    error("Paulis can only have characters {I,X,Y,Z}.")
                end
            end
        end
        return PauliMeasurement(X, Z, c, basis, frame)
    end

    """
        PauliMeasurement(observable, basis, frame)

    A constructor accepting a dense matrix observable.

    The Pauli coefficients are computed once from (a not very efficient implementation of)
        the Hilbert-Schmidt inner product.

    # Parameters
    - `observable`: the dense matrix observable.
    - `basis`: the `BasisType` identifying the basis `observable` is written in.
    - `frame`: the `OperatorType` identifying the frame where measurements are conducted.

    """
    function PauliMeasurement(
        observable::AbstractMatrix{F},  # NOTE: Assumes Hermitian qubit operator.
        basis::Bases.BasisType,
        frame::Operators.StaticOperator;
        eps=1e-10,
    ) where {F}
        N = size(observable, 1)
        n = QubitProjections.nqubits(N, 2)
        σ = reshape(Devices.localalgebra(PauliAlgebra{1}()), 2, 2, 3)

        X = Int[]
        Z = Int[]
        c = real(F)[]

        Pq = Array{eltype(σ)}(undef, 2, 2, n)
        P  = Array{eltype(σ)}(undef, N, N)
        for x in 0:N-1
        for z in 0:N-1
            for q in 1:n
                mask = 1 << (n-q)
                hasx = !iszero(x & mask)
                hasz = !iszero(z & mask)
                if hasx && hasz
                    Pq[:,:,q] .= σ[:,:,2]
                elseif hasx
                    Pq[:,:,q] .= σ[:,:,1]
                elseif hasz
                    Pq[:,:,q] .= σ[:,:,3]
                else
                    Pq[:,:,q] .= one(Pq[:,:,q])
                end
            end
            LAT.kron(Pq; result=P)
            cP = real(LinearAlgebra.tr(P*observable)) / N
            abs(cP) < eps && continue
            push!(X, x)
            push!(Z, z)
            push!(c, cP)
        end; end

        return PauliMeasurement(X, Z, c, basis, frame)
    end

    function Modular.measure(
        measurement::PauliMeasurement{F},
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        ψ::AbstractVector,
        t::Real,
    ) where {F}
        # COPY THE STATE SO WE CAN ROTATE IT
        ψ_ = @temparray(eltype(ψ), size(ψ), :measure)
        ψ_ .= ψ

        # ROTATE THE STATE INTO THE MEASUREMENT BASIS
        U = Devices.basisrotation(measurement.basis, basis, device)
        LAT.rotate!(U, ψ_)

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurement.frame, device, measurement.basis, -t, ψ_)

        # TAKE THE EXPECTATION VALUE
        m = Devices.nlevels(device)
        n = Devices.nqubits(device)
        N = Devices.nstates(device)
        E = zero(Complex{F})
        for k in eachindex(measurement.c)
            nY = count_ones(measurement.X[k] & measurement.Z[k])
            for z in 0:N-1
                z_ = z ⊻ measurement.X[k]
                i  = QubitProjections.mapindex(z +1, n, m)
                i_ = QubitProjections.mapindex(z_+1, n, m)

                nZ = count_ones(z & measurement.Z[k])
                E += (im)^nY * (-1)^nZ * measurement.c[k] * ψ_[i] * ψ_[i_]'
            end
        end
        return real(E)
    end

    CostFunctions.nobservables(::Type{<:PauliMeasurement}) = 1

    function Modular.observables(
        measurement::PauliMeasurement,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        t::Real;
        result=nothing
    )
        m = Devices.nlevels(device)
        n = Devices.nqubits(device)
        N = Devices.nstates(device)
        isnothing(result) && (result = Array{Complex{eltype(device)}}(undef, N, N, 1))
        O = @view(result[:,:,1])

        # REPRESENT O IN THE MEASUREMENT BASIS
        O .= 0
        for k in eachindex(measurement.c)
            nY = count_ones(measurement.X[k] & measurement.Z[k])
            for z in 0:N-1
                z_ = z ⊻ measurement.X[k]
                nZ = count_ones(z & measurement.Z[k])

                i  = QubitProjections.mapindex(z +1, n, m)
                i_ = QubitProjections.mapindex(z_+1, n, m)

                O[i_, i] += (im)^nY * (-1)^nZ * measurement.c[k]
            end
        end

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, measurement.basis, device)
        LAT.rotate!(U, O)

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurement.frame, device, basis, t, O)

        return result
    end

    function Devices.gradient(
        measurement::PauliMeasurement,
        device::Devices.DeviceType,
        grid::Integrations.IntegrationType,
        ϕ::AbstractArray,
        ψ::AbstractVector;
        result=nothing
    )
        return Devices.gradient(device, grid, @view(ϕ[:,:,1]); result=result)
    end
end