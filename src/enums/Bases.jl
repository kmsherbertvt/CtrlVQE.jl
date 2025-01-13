"""
    Bases

Enumerates various linear-algebraic bases for representing statevectors and matrices.

"""
module Bases
    export BasisType
    export DRESSED, BARE

    abstract type BasisType end

    """
        Dressed(), aka DRESSED

    The eigenbasis of the static Hamiltonian associated with a `Device`.
    Eigenvectors are ordered to maximize similarity with an identity matrix.
    Phases are fixed so that the diagonal is real.

    """
    struct Dressed <: BasisType end
    const DRESSED = Dressed()

    """
        Bare(), aka BARE

    The "default" representation, defined by the `localalgebra` a `Device` implements.

    For transmons, it is the eigenbasis of local number operators ``n̂ ≡ a'a``,
        and generally, it is what would be called the "computational basis".

    """
    struct Bare <: BasisType end
    const BARE = Bare()

end