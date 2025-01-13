# Algebraic Framework

The types in `CtrlVQE` are designed to be as flexible as possible, while remaining *performant*.
We did this by designing the `DeviceType` to represent the complete Hamiltonian,

$$\hat H(\vec x,t) = \hat H_0(\vec x) + \sum_i \hat V_i (\vec x,t)$$

There is an interface to follow drift and drive terms are handled correctly and efficiently in time evolution,
    but it is completely and entirely your choice how to implement them.
But prioritizing *flexibility* and *performance* has a tendency to sacrifice *readability* and *reusability*.

This package attempts to find a middle ground,
    by splitting each bit of the Hamiltonian equation into chunks.
That is, there is one type (`AlgebraType`) which defines the *space* on which each operator $\hat H$, $\hat H_0$, $\hat V_i$ acts.
There is another (`DriftType`) defining $\hat H_0$.
There is another (`DriveType`) defining $\hat V_i$.

And to help keep the drives as modular as possible,
    there is a ``MapperType`` to map "device" parameters to individual drift and drive.
A more complete equation - which is itself less readable,
    but it makes the code *so* much easier to work with - would be

$$\hat H(\vec x,t) = \hat H_0(h(\vec x)) + \sum_i \hat V_i(f_i(\vec x), t) $$

For example, let's say you're curious how a single three-qubit interaction term
    in the static Hamiltonian of a transmon device would alter the minimal evolution time.
This is a tiny change, but going from Basics,
    it would seem to require implementing a whole new device type, and to do that correctly,
    you'd have to be sure to implement the dozen or so methods required for new device types.
You'll presumably just copy and paste from an existing implementation,
    which is generally considered bad practice and has a tendency to produce unexpected bugs
    (hence the desire for code reusability).
And you'll generate a file with hundreds of lines of code,
    a real pain to organize, document, and debug
    (hence the desire for code readability).

Going from here, you'd simply implement a new drift type, with two methods, and be done.
