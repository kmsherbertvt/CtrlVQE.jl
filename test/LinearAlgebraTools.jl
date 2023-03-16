using Test
using CtrlVQE: LinearAlgebraTools


# DEFINE SOME EASY TEST ARRAYS

v1 = [1]
v2 = [1, 2]
v3 = [1, 2, 3]

M21 = [1
       2]
M22 = [1 3
       2 4]
M23 = [1 3 5
       2 4 6]



@testset "LinearAlgebraTools" begin
    @test LinearAlgebraTools.kron(v1, v1) == [1]
    @test LinearAlgebraTools.kron(v1, v2) == [1,2]
    @test LinearAlgebraTools.kron(v2, v2) == [1,2,2,4]
    @test LinearAlgebraTools.kron(v2, v3) == [1,2,3,2,4,6]
end
