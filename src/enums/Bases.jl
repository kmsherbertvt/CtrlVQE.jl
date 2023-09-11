export BasisType, LocalBasis
export DRESSED, OCCUPATION, COORDINATE, MOMENTUM

abstract type BasisType end

"""
    Dressed(), aka DRESSED

The eigenbasis of the static Hamiltonian associated with a `Device`.
Eigenvectors are ordered to maximize similarity with an identity matrix.
Phases are fixed so that the diagonal is real.

"""
struct Dressed <: BasisType end
const DRESSED = Dressed()




abstract type LocalBasis <: BasisType end

"""
    Occupation(), aka OCCUPATION

The eigenbasis of local number operators ``n̂ ≡ a'a``.
Generally equivalent to what is called the "Z basis", or the "computational basis".

"""
struct Occupation <: LocalBasis end
const OCCUPATION = Occupation()

"""
    Coordinate(), aka COORDINATE

The eigenbasis of local quadrature operators ``Q ≡ (a + a')/√2``.

"""
struct Coordinate <: LocalBasis end
const COORDINATE = Coordinate()

"""
    Momentum(), aka MOMENTUM

The eigenbasis of local quadrature operators ``P ≡ i(a - a')/√2``.

"""
struct Momentum <: LocalBasis end
const MOMENTUM   = Momentum()