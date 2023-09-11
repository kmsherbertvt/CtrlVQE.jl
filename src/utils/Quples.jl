"""
    Quple(q1,q2)

A (symmetric) coupling between qubits indexed by `q1` and `q2`.

Note that the order is irrelevant: `Quple(q1,q2) == Quple(q2,q1)`.

"""
struct Quple
    q1::Int
    q2::Int
    # INNER CONSTRUCTOR: Constrain order so that `Quple(q1,q2) == Quple(q2,q1)`.
    Quple(q1, q2) = q1 > q2 ? new(q2, q1) : new(q1, q2)
end

# IMPLEMENT ITERATION, FOR CONVENIENT UNPACKING
Base.iterate(quple::Quple) = quple.q1, true
Base.iterate(quple::Quple, state) = state ? (quple.q2, false) : nothing