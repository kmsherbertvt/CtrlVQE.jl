From the start of February (when frantically trying to come up with a way to get pulses for Hisham) to my BNL trip, I was frenetically pursuing some lines of thought which I think, when meditated upon more carefully, may give a plausible direction for finding the "right" coupling pulses. I kinda hit a road block, though, and I *have* to set this aside for a couple other far more pressing projects. So these notes are to catch you up on what I was thinking, so that you can stew over them for awhile.

Each section below is titled according to the somewhat arbitrarily named directory I used to develop the idea at that stage.

# Adiabatic Evolution

Observation: optimizing infidelity, the optimizer quickly finds a point somewhat better than the reference, but then makes no progress, even after increasing pulse duration.

Hypothesis: Intuitively, infidelity should decrease as you progress along the geodesic connecting reference and target states. But perhaps a nearby local minimum results in an even steeper descent in a different direction?

Idea: construct a *path*, and optimize *progress* along the path, while penalizing *deviation* from the path.

Intuitively, we expect:
- nevertheless, this should force the optimizer to stay on the path
- the path we select may not necessarily be the time-optimal path
- nevertheless, increasing pulse duration should always let you progress further along the path

For example, we can select the following path:
- imagine the reference state and target state as points on a Bloch sphere in the two-level subspace they span
- the great circle connecting the two points is the path
- the poles of the Bloch sphere are an arbitrary orthogonal pair of vectors $|\ket{0}$ and $\ket{1}$ defined in terms of the reference and target states; the particular choice serves to rotate the Bloch sphere but doesn't change the trajectory any.

Hindsight notes:
- At this stage I went to some effort to define |0> and |1> so that the reference and target states were treated as symmetrically as possible (i.e. their coordinates on the Bloch sphere are $\theta = \pi/2 \pm \theta_0$, $\phi = \pm \phi_0$).
- In all later stages, I realized it's much simpler and quite frankly more intuitive to put the reference at |1>, construct |0> with basic Gram-Schmidt orthogonalization, and then the great circle is just a meridian with $\phi$ given by that of the target state.
- Hisham has since pointed out to me that the Bloch sphere is not actually unique, since the global phase of the Bloch sphere is not global when embedded in the total Hilbert space! That means there is a free parameter $\alpha$ which could be tuned to improve performance. I haven't tried this yet, though I'm pessimistic it would result in improved performance.

Optimization
- Given an ansatz, we can define a path parameter $\Gamma$ (e.g. which falls from 1 (at the reference) to 0 (at the target)) by projecting onto the Bloch sphere and identifying the angle traversed over the great circle connecting reference and target.
- Deviation from the path $\Delta$ is measured by the quality of that projection (i.e. if most of your probability mass is not in the Bloch sphere, you are deviating far from the path).
- I guess there must be additional deviation from being in the Bloch sphere but not on the great circle. I can't seem to remember handling that anywhere in what I implemented, but it's perfectly doable (especially easy if the great circle is a meridian).
- For the purposes of this, I adopted an "adaptive concatenation" scheme, optimizing a pulse then giving it another 10ns to play with, the idea being that the additional progress along the path after each round of optimization is due mostly to the last window.  Roughly speaking this is supposed to justify the name of this section.
- This idea lends itself really well to freezing previous windows' parameters. It makes for delightfully fast optimizations, since I can save statevectors from previous iterations, so each new iteration need only simulate 10ns more. Roughly, I think this strat took about twice as long total pulse duration to get to the same point as when you allow parameters to relax, which is probably interesting but a bit off topic, so I won't bother plotting the comparison here.
  - I also experimented with optimizing not infidelity or progress along path but the *integral* of infidelity over pulse duration, involving the same sort of mathematics we'd discussed for penalizing leakage throughout pulse duration. You might expect it to achieve the same desired effect. But it is also a bit off topic, and the results aren't especially interesting, so I won't bother plotting it here.

Results
- I can make a really great-looking parametric plot of $\Gamma$ and $\Delta$, where a marker for each Trotter step tells you where your trjaectory is with respect to the path. This should be a sensible 2D way of plotting optimization trajectories, even when not using the path coordinates for the optimization.
- Unfortunately, it doesn't work as intended, at all. It happily zips right off the path, straight to a point that has the same infidelity as the results from optimizing infidelity!
- I can scale up the penalty for deviating from the path. It follows the same trajectory for a short distance but then simply refuses to go further.

![adiabaticpath.pdf](Update/adiabaticpath.pdf)

# Tangent Spaces

Question: how well do our parameters allow us travel down the meridian?
- Pulse ansatz: single short square window, abstracting an instantaneous push of the pulses
- Build a (non-orthogonal) basis from partial derivatives of each parameter (aka the Jacobian)
- Construct tangent vector along great circle of Bloch sphere connecting reference and target state
- Project the tangent vector onto the basis, giving a measure of how well the drives can instantaneously push along Bloch sphere
- Do this for each point on the great circle

Tried this for a bunch of numerical things, e.g.
- more accurate finite difference
- more Trotter steps
- with/out RWA
- square/smooth transition window edge
None of these made any difference to the plot

What did make a difference:
- pulse duration
  - because *frame rotation* gives different states!
  - this is probably very important but I haven't thought about it at all
- bosonic truncation
  - a bit worse, easily explained from trying to project a vector onto an even less complete basis
  - 4 levels identical to 3

![tangentspaces.pdf](Update/tangentspaces.pdf)

# Linearity

To what extent are dynamics linear in pulse parameters?
That is, to what extent is a given ansatz (with each drive driven by some amount) described as a linear combination of the directions given above?

- Write down an ansatz, ie. a combination of pulse parameters (each one is a square window on a different drive).
- Compute the linear combination of the basis vectors defined above (aka the Jacobian) corresponding to the ansatz. That gives you an "expected" direction, if one presumes linearity.
- Just for fun, the ansatz we write down is the one whose linear combination of basis vectors would give maximal overlap with the tangent vector of the great circle defined above, at the reference state. But any arbitrary ansatz would serve the purpose.
- Compute the actual direction in Hilbert space that the ansatz sends you in, and compare.

Results:
- dynamics are NOT linear in pulse parameters, at ALL...
- Below I show three vectors
  1. the tangent vector of the great circle
  2. the linear combination of basis vectors giving the best overlap with (1)
  3. the actual direction when plugging in the parameters that would ostensibly have produced (2) if dynamics were linear in pulse parameters
- I also show a weird error matrix
  - The bottom triangle is overlap $|\braket{\psi,\phi}|^2$.
  - The upper triangle is the error norm $|\psi - \phi|$.
  - The important bit is, the overlaps are all small and the errors are all large.

```
The Vectors: ∂θψ | Projection under linearity | Actual reconstruction
16×3 Matrix{ComplexF64}:
   -0.0815931+0.0im           6.85783e-18-1.18055e-18im  -9.56487e-18-1.81533e-18im
  2.93738e-17+0.0im            -0.0175087-0.00194062im      0.0424736+0.00470724im
    -0.289759+0.0im             -0.291884-0.00503447im       0.707919+0.0122081im
  6.12323e-17-1.96282e-16im           0.0+0.0im                   0.0+0.0im
    -0.448306+0.0im           -0.00247227+0.00298774im     0.00599636-0.00724641im
  4.31646e-22+0.0im                   0.0+0.0im                   0.0+0.0im
     0.551394+0.0im                   0.0+0.0im                   0.0+0.0im
             ⋮
  1.04869e-16+0.0im                   0.0+0.0im          -4.62542e-13+0.0im
   1.9925e-15+0.0im                   0.0+0.0im                   0.0+0.0im
  2.07115e-15+0.0im            -0.0175997-0.00183043im      0.0426976+0.0044386im
     0.335818+0.0im           3.16879e-14+3.12568e-14im   1.73453e-13+5.78178e-14im
 -4.19474e-16+0.0im          -0.000104599+0.000143955im    0.00025374-0.000349295im
    -0.448306+0.0im          -0.000613793-6.51929e-5im     0.00148916+0.000158016im
   -0.0815931-0.0im           7.16835e-18+3.28149e-19im  -1.73857e-17-7.95825e-19im
Overlap \ Error
3×3 Matrix{Float64}:
 0.0        0.911033  1.68067
 0.0289062  0.0       1.41233
 0.170018   0.170018  0.0
```

# Optimization

Dynamics aren't linear in pulse parameters.
So, what IS the best ansatz our drives can do?

Operation:
- Take a trial set of pulse parameters.
- Construct a "coupled" device with a single parameter x, such that the actual drive is x * each pulse parameter.
- Compute the Jacobian for this single-parameter device, centered on x=0
  - i.e. what direction do you go in when you apply an infinitesimal drive where each pulse has a relative strength defined by the trial set of pulse parameters?
- Compute the overlap of this direction vector with the tangent of the great circle we want to traverse.
- Optimize the trial set of pulse parameters to maximize that overlap.

Results:
- The optimized overlap here is much worse than the projection onto the Jacobian basis would suggest should be possible.
- I feel like this explains why the first experiment failed so spectacularly.
- Recall my previous note that the Bloch sphere I am using is not unique; optimizing the Bloch sphere's global phase could in principle result in somewhat better results than presented here (but I haven't tried).

![optimized.pdf](Update/optimized.pdf)

# Momentum

A given set of coupled pulse parameters, centered on x=0, pushes you instantaneously in a direction.
How useful is this information..?
E.g. if you apply these pulse parameters with a finite x, how large can x be before the direction you actually go is totally different?

Process:
- Write down an ansatz, i.e. a fixed set of pulse parameter ratios. For the sake of picking something, we'll use the one that maximizes the overlap with the tangent vector of the great circle.
- Sample x on a linspace from 0 up to its maximum value (as set by an amplitude constraint).
- For each x, compute the direction in Hilbert space that evolving with the drive actually imparts.
- Plot the overlap of each direction with that of the tangent vector of the great circle.

Expectations:
- At x=0, overlap should be that which we get on the previous plot.
- I'd expect it to decay as x increases, since the direction actually traversed would probably become increasingly unrelated to the initial direction.
- The interesting question ought to be, does that decay happen within the range of values that x can take or not?

Results:
- The plot is exactly the opposite of expectations. At small x, overlap is zero. As x increases, it asymptotically approaches the expected value.
- I have NO IDEA what is going on here!!! I dug deep into the FiniteDifferences code to be sure I was definitely using a suitably small step for the purposes of computing the instantaneous direction, and I really do seem to be, but this seems to be wholly contradictory. It surely must be a dumb bug, but I felt I was going a bit insane so I stopped looking for it. Well that and I had to travel. :)
- Also of note is the fact that, at small x, the overlap with the target tangent vector is somewhat *better* than the asymptotic value corresponding to the supposedly optimal parameters, presented in the previous plot. I mean, it's *supposed* to be the small x overlaps that give the best overlap with the tangent vector. But, the optimizer's loss function measures the overlap of the vector apparently reproduced when using *large* x. So this is all very confusing.

![momentum.pdf](Update/momentum.pdf)