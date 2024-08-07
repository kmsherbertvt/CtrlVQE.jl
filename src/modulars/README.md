This folder is my sandbox for drafting a new, more modular way of building up devices.
Actually, it sort-of hearkens back to an earlier iteration of my code,
    which I discarded as "too complicated and spaghettified".
However, I think having it as a SUB-type of the generic DeviceType lets me have it both ways.

In particular, this new design should make easy the following use-cases:
1. Coupling multiple signals to single set of parameters.
2. Allowing drive frequencies to be time-dependent if desired.

The overall strategy is to separate the static Hamiltonian and the drive terms into independent pieces
    (the idea being that the static Hamiltonian is particular to the actual underlying architecture,
    but the drives are what we do to it).
We'll do as control theorists normally do, and associate a
