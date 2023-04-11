using Test
import CtrlVQE: LinearAlgebraTools

@testset "LinearAlgebraTools" begin
    i = im

    # PREPARE SOME BASIC TEST ARRAYS
    e0 = [1, 0]             # |0⟩
    e1 = [0, 1]             # |1⟩
    e⁺ = [1, 1] / √2        # |+⟩
    e⁻ = [1,-1] / √2        # |-⟩
    I  = [1 0; 0 1]         # Identity
    X  = [0  1; 1  0]       # Pauli X
    Y  = [0 -1; 1  0]i      # Pauli Y
    Z  = [1  0; 0 -1]       # Pauli Z
    RX = [1 i; i 1] / √2    # Rotate about X axis by -π/2 radians

    # TEST `kron`
    @test LinearAlgebraTools.kron([e0, e0, e1]) == [0,1,0,0,0,0,0,0]
        # ORDER MATTERS: |001⟩≃|1⟩, not |5⟩
    @test LinearAlgebraTools.kron([X, I]) == [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
        # ORDER MATTERS: X⊗I = [0 I; I 0], not [X 0; 0 X]

    # TEST `cis!`
    res = convert(Array{ComplexF64}, X)
    res_ = LinearAlgebraTools.cis!(res, π/4)
    @test res === res_
    @test res ≈ RX

    # TEST `rotate!`
    res = convert(Array{ComplexF64}, e⁻)
    res_ = LinearAlgebraTools.rotate!(RX, res)
    @test res === res_
    @test res ≈ RX * e⁻

    res = convert(Array{ComplexF64}, Y)
    res_ = LinearAlgebraTools.rotate!(RX', res)
    @test res === res_
    @test res ≈ Z

    # TEST `rotate!` WITH TENSOR CONTRACTION
    r̄ = [I, RX, I]
    R = LinearAlgebraTools.kron(r̄)
    ē = [e0, e⁻, e⁺]
    x = LinearAlgebraTools.kron(ē)

    res = convert(Array{ComplexF64}, x)
    res_ = LinearAlgebraTools.rotate!(r̄, res)
    @test res === res_
    @test res == R * x

    r̄ = [I, convert(Matrix, RX'), I]    # NOTE: kron args must be same type
    R = LinearAlgebraTools.kron(r̄)
    ā = [X, Y, Z]
    A = LinearAlgebraTools.kron(ā)

    res = convert(Array{ComplexF64}, A)
    res_ = LinearAlgebraTools.rotate!(r̄, res)
    @test res === res_
    @test res == R * A * R'

    # TEST `expectation` AND `braket`
    @test LinearAlgebraTools.expectation(X, e⁺) ≈ 1.0
    @test LinearAlgebraTools.braket(e⁻, X, e⁺) ≈ 0.0

end
