# Time Evolution: Single Qubit Solutions
##### 11/29/2022
##### Updated 4/6/2023
##### Updated 12/11/2024

Given an input state $|ψ_0⟩$ subject to a Hamiltonian $\hat H=\hat H_0+\hat V(t)$ , what is the state $|ψ(T)⟩$?
In this notebook, we will use a single transmon qubit:

$$\hat H_0 = ω a^† a - \frac{δ}{2} a^† a^† a a $$

with a time-dependent EM field:

$$\hat V(t) = Ω (e^{iνt} a + e^{-iνt} a^†) $$

The amplitude $Ω$ and frequency $ν$ may in general be time-dependent, although the first (and currently only) examples will take them as constant for the duration of the evolution (ie. a basic square pulse).

The above formulation assumes $ν$ and $Ω$ are real.
We are stuck with $ν$, but we *could* include a complex phase in $Ω$.
In that case, we should write:

$$\hat V(t) = Ω e^{iνt} a + \bar Ω e^{-iνt} a^†) $$

Throughout the derivation, we will use the real version of $Ω$, but at the end I'll point out what happens if $Ω$ is complex.

The operator $a$ is the bosonic annihilation operator, defined by the following action:

$$ a|n⟩ = \sqrt{n} |n-1⟩ $$

where $|n⟩$ is the state with occupation number $n$.
We will use this occupation number basis throughout, but we will typically truncate it to a finite number of $m$ modes for each problem.

The solutions in this notebook will be solutions to Schrödinger's equation in the interaction picture:

$$ i \frac{∂}{∂t} |ψ_I⟩ = \hat V_I |ψ_I⟩ $$

where the interaction Hamiltonian is:

$$ \hat V_I(t) = e^{it\hat H_0} \hat V(t) e^{-it\hat H_0} $$

We will commit 100% to the interaction picture, so we will start and end with the interaction statevector $|ψ_I⟩=e^{it \hat H_0} |ψ⟩$.
To calculate the "true" state, remember to "rewind" the results by $e^{-it\hat H_0}$.

Each solution will go through the following basic steps:
1. Represent $|ψ_I⟩$ and $\hat V_I(t)$ as a finite vector and matrix, respectively.
2. Interpret Schrödinger's equation as a system of coupled linear first-order differential equations.
3. Attempt to solve the system analytically, giving a closed form solution of $|ψ_I(t)⟩$ in the finite representation.

## Basic Square Pulse: $m=2$

For this first problem, we will take $Ω$ and $ν$ as constant.

### Step One: Representation in a Finite Space
We may expand our wavefunction at any arbitrary time as:

$$ |ψ_I⟩=\sum_{i=0}^{m-1} c_i |i⟩ $$

where the coefficients $c_i≡⟨i|ψ_I⟩$ are time-dependent.
The vector representation of $|ψ_I⟩$ is simply a column vector of these $c_i$ coefficients.

Now let's focus on constructing $\hat V_I$ as a matrix.

Truncated to $m=2$, the bosonic annihilation operator $a$ has the following form:

$$ a = \left[\begin{array}{cc}
    0 & 1 \\
    0 & 0
\end{array}\right]$$

The static Hamiltonian $\hat H_0$ is:

$$\begin{aligned} \hat H_0
&= ω a^† a - \frac{δ}{2} a^† a^† a a \\
&= ω \left[\begin{array}{cc}
	0 & 0 \\
	1 & 0
\end{array}\right] · \left[\begin{array}{cc}
	0 & 1 \\
	0 & 0
\end{array}\right] - \frac{δ}{2} a^† a^† \left[\begin{array}{cc}
	0 & 1 \\
	0 & 0
\end{array}\right] · \left[\begin{array}{cc}
	0 & 1 \\
	0 & 0
\end{array}\right] \\
&= ω \left[\begin{array}{cc}
	0 & 0 \\
	0 & 1
\end{array}\right] -  \frac{δ}{2} a^† a^† \left[\begin{array}{cc}
	0 & 0 \\
	0 & 0
\end{array}\right] \\
&= \left[\begin{array}{cc}
	0 & 0 \\
	0 & ω
\end{array}\right]
\end{aligned}$$

Note that the quadratic operator $a^† a^† a a$ is actually zero for $m=2$, rendering the anharmonicity $δ$ irrelevant in this problem.

Since this is a diagonal matrix, the matrix exponential $e^{it\hat H_0}$ is easily calculated by applying the exponential function to each diagonal entry:

$$ e^{it\hat H_0} = \left[\begin{array}{cc}
	1 & 0 \\
	0 & e^{iωt}
\end{array}\right]$$

The drive operator $\hat V(t)$ is:

$$\begin{aligned} \hat V(t)
&= Ω\left(e^{iνt} \left[\begin{array}{cc}
    0 & 1 \\
    0 & 0
\end{array}\right] + e^{-iνt} \left[\begin{array}{cc}
    0 & 0 \\
    1 & 0
\end{array}\right]\right) \\
&= \left[\begin{array}{cc}
    0 & Ω e^{iνt} \\
    Ω e^{-iνt} & 0
\end{array}\right]
\end{aligned}$$

Conjugation with the drive operator yields:

$$\begin{aligned} \hat V_I(t)
&= e^{it\hat H_0} \hat V(t) e^{-it\hat H_0} \\
&= \left[\begin{array}{cc}
	1 & 0 \\
	0 & e^{iωt}
\end{array}\right] · \left[\begin{array}{cc}
    0 & Ω e^{iνt} \\
    Ω e^{-iνt} & 0
\end{array}\right] · \left[\begin{array}{cc}
	1 & 0 \\
	0 & e^{-iωt}
\end{array}\right] \\
&= \left[\begin{array}{cc}
    0 & Ω e^{iνt} \\
    Ω e^{-iΔt} & 0
\end{array}\right] · \left[\begin{array}{cc}
	1 & 0 \\
	0 & e^{-iωt}
\end{array}\right] \\
&= \left[\begin{array}{cc}
    0 & Ω e^{iΔt} \\
    Ω e^{-iΔt} & 0
\end{array}\right]
\end{aligned}$$

where we have defined the detuning $Δ≡ν-ω$.

### Step Two: Defining the System of Equations
Schrödinger's equation $i \frac{∂}{∂t} |ψ_I⟩ = \hat V_I |ψ_I⟩$ can now be rewritten as a matrix equation:

$$ i \frac{∂}{∂t} \left[\begin{array}{c}
	c_0 \\
	c_1
\end{array}\right] = \left[\begin{array}{cc}
    0 & Ω e^{iΔt} \\
    Ω e^{-iΔt} & 0
\end{array}\right] · \left[\begin{array}{c}
	c_0 \\
	c_1
\end{array}\right] $$

This is really just two scalar equations:

$$\begin{aligned}
i \dot c_0 &= Ω e^{+iΔt} c_1 \\
i \dot c_1 &= Ω e^{-iΔt} c_0
\end{aligned}$$

Dividing by $i$ and throwing everything to one side puts the differential equations in standard form:

$$ \begin{aligned}
\dot c_0 + iΩ e^{+iΔt} c_1 &= 0 \\
\dot c_1 + iΩ e^{-iΔt} c_0 &= 0
\end{aligned} $$

### Step Three: Solving the System of Equations
We'll focus on solving $c_0(t)$ first, then use a symmetry argument to write down $c_1(t)$.

Take our first equation:

$$ \dot c_0 + iΩ e^{+iΔt} c_1 = 0 $$

Now take an extra derivative:

$$ \ddot c_0 + iΔ (iΩ e^{iΔt} c_1)  + iΩ e^{iΔt} \dot c_1 = 0 $$

We can eliminate $c_1$ using the original first equation, and $\dot c_1$ using the second equation:

$$ \ddot c_0 + (-iΔ) \dot c_0 + Ω^2 c_0 = 0 $$

This is a homogenous second-order linear differential equation with constant coefficients.
You can look up the standard solution in a textbook, but I'll run through a quick way to pseudo-re-derive it, which I have always found invaluable.

First off, we know that there should be two linearly independent solutions: no more, no less.
Second off, this is a physics problem, so we might as well try $c_0(t) = e^{rt}$.
(:P)
Then we have:

$$ r^2 e^{rt} - iΔ r e^{rt} + Ω^2 e^{rt} = 0 $$

We can divide off the $e^{rt}$ factor (so long as we don't later discover $r=-\infty$) to generate the so-called "auxiliary equation":

$$ r^2 - iΔr + Ω^2 = 0 $$

This is a degree-2 polynomial equation, which has two roots - so as long as they're unique, that's our two solutions right there.
Using the quadratic equation:

$$ r = \frac{1}{2}(iΔ \pm \sqrt{-Δ^2 - 4Ω^2}) $$

Note immediately that the discriminant _cannot_ be zero for any $Ω \ne 0$, so we have indeed found both solutions.

Now I'm going to use the greatest weapon in the theoretical physicist's arsenal: a large alphabet.
Let me define two new constants:

$$ \begin{aligned}
χ &≡ 2Ω/Δ \\
η &≡ \sqrt{1 + χ^2}
\end{aligned} $$

Now let me simplify $r$:

$$ \begin{aligned} r_\pm
&= \frac{1}{2}(iΔ \pm \sqrt{-Δ^2 - 4Ω^2}) \\
&= \frac{1}{2}(iΔ \mp iΔ \sqrt{1 + 4Ω^2/Δ^2}) \\
&= iΔ · \frac{1}{2}(1 \mp \sqrt{1 + χ^2}) \\
&= iΔ \frac{1 \mp η}{2}
\end{aligned} $$

This is the _simplest_ form for $r$, but since $η$ has 1 as a lower bound, it's more _elegant_ to use this form:

$$ r_\pm = \pm iΔ \frac{η \pm 1}{2} $$

Now we can write out a _general_ solution as the linear combination of our two particular solutions:

$$ \begin{aligned} c_0(t)
&= A_+ e^{r_+t} + A_- e^{r_-t} \\
&= A_+ e^{-iΔ \frac{η - 1}{2}t} + A_- e^{iΔ \frac{η + 1}{2}t}
\end{aligned} $$

We will soon need an explicit formula for $\dot c_0$ also:

$$ \dot c_0 = -iΔ \frac{η - 1}{2} A_+ e^{-iΔ \frac{η - 1}{2}t}
	+  iΔ \frac{η + 1}{2} A_- e^{iΔ \frac{η + 1}{2}t} $$

We will solve for $A\pm$ by constraining them to our initial conditions:

$$ \begin{aligned}
c_0(0) &= ⟨0|ψ_0⟩ \\
c_1(0) &= ⟨1|ψ_0⟩
\end{aligned} $$

The second constraint actually gives us $\dot c_0(0)$, from our first differential equation:

$$ \begin{aligned}
\dot c_0 + iΩ e^{iΔt} c_1 &= 0 \\
\dot c_0(0) &= -iΩ c_1(0) \\
&= -iΩ ⟨1|ψ_0⟩
\end{aligned} $$

Now we can put in $c_0(0)$ and $\dot c_0(0)$ to create a system of two equations and solve for our two unknowns $A_\pm$:

$$ \begin{aligned}
	⟨0|ψ_0⟩ &= A_+ + A_- \\
-iΩ ⟨1|ψ_0⟩ &= -iΔ \frac{η - 1}{2} A_+ + iΔ \frac{η + 1}{2} A_-
\end{aligned} $$

I will let the reader confirm the following solution:

$$ \begin{aligned}
A_\pm = ⟨0|ψ_0⟩ \frac{η \pm 1}{2η} \mp ⟨1|ψ_0⟩ \frac{χ}{2η}
\end{aligned} $$

The final solution can be written out as:

$$ \begin{aligned} c_0(t)
   =&  \left(⟨0|ψ_0⟩ \frac{η + 1}{2η} + ⟨1|ψ_0⟩ \frac{χ}{2η}\right)
	   e^{-iΔ \frac{η - 1}{2}t} \\
	&+ \left(⟨0|ψ_0⟩ \frac{η - 1}{2η} - ⟨1|ψ_0⟩ \frac{χ}{2η}\right)
		e^{iΔ \frac{η + 1}{2}t}
\end{aligned} $$


To solve $c_1(t)$, note that the two differential equations are identical except that they permute $|0⟩ \leftrightarrow |1⟩$ and $Δ\leftrightarrow-Δ$.
Therefore, the solution is:

$$ \begin{aligned} c_1(t)
   =&  \left(⟨1|ψ_0⟩ \frac{η + 1}{2η} - ⟨0|ψ_0⟩ \frac{χ}{2η}\right)
	   e^{iΔ \frac{η - 1}{2}t} \\
	&+ \left(⟨1|ψ_0⟩ \frac{η - 1}{2η} + ⟨0|ψ_0⟩ \frac{χ}{2η}\right)
		e^{-iΔ \frac{η + 1}{2}t}
\end{aligned} $$

If $Ω$ is complex, $η$ is redefined sensibly as $\sqrt{1+|χ|^2}$.
The only other change we need to note is that the two differential equations now include a switch between $Ω \leftrightarrow \bar Ω$.
Thus, we simply replace each $χ$ in $c_1(t)$ with $\bar χ$.

## Basic Square Pulse: $m=3$
This problem will be considered a direct sequel to the $m=2$ problem, so I won't re-state things I worked out in detail there.
Instead, I will write down the results for Steps One and Two (as defined in the $m=2$ problem), an intermediate result for Step Three, then describe how to finish the problem with robust and easily-understood numerical calculations ('cause I ran out of patience trying to do them by hand...).

### Step One: Representation in a Finite Space
As in the $m=2$ problem, $|ψ_I⟩$ will just be a vector of time-dependent coefficients $c_i$ in the occupation number basis.
Now there will be three of them.

To construct the matrix representation of $\hat V_I$, note that the annihilation operator $a$ now becomes:

$$ a = \left[\begin{array}{cc}
    0 & 1 & 0 \\
    0 & 0 & \sqrt{2} \\
    0 & 0 & 0
\end{array}\right] $$

With patience, you should be able to derive:

$$ \hat V_I(t) = \left[\begin{array}{cc}
    0 & Ω e^{iΔt} & 0 \\
    Ω e^{-iΔt} & 0 & \sqrt{2} Ω e^{i(Δ-δ)t} \\
    0 & \sqrt{2} Ω e^{-i(Δ-δ)t} & 0
\end{array}\right] $$

Using that greatest of tools, I'm just going to define the constants:

$$ \begin{aligned}
V_1 &≡ Ω e^{-iΔt} \\
V_2 &≡ \sqrt{2} Ω e^{-i(Δ-δ)t} \\
\end{aligned} $$

Using bars to denote complex conjugates, I can now write:

$$ \hat V_I(t) = \left[\begin{array}{cc}
    0 & \overline V_1 & 0 \\
    V_1 & 0 & \overline V_2 \\
    0 & V_2 & 0
\end{array}\right] $$

Note that the complex-ness of $Ω$ is just folded into the complex-ness of $V_i$,
	so actually the results below apply perfectly well to complex amplitudes.

### Step Two: Defining the System of Equations
Expanding out Schrödinger's equation:

$$ \begin{aligned}
\dot c_0 + i\overline V_1 c_1 &= 0 \\
\dot c_1 + i V_1 c_0 + i\overline V_2 c_2 &= 0 \\
\dot c_2 + i V_2 c_1 &= 0
\end{aligned} $$

### Step Three: Solving the System of Equations
I found it immensely helpful for the algebra to define just a couple more constants:

$$ \begin{aligned}
C_i &≡ |V_i|^2 \\
D_i &≡ (∂_t V_i) / V_i
\end{aligned} $$

The next step is to de-couple the system of differential equations, obtaining a third-order differential equation for each $c_i$.
I can't pretend it's easy...
Here's what I got:

$$ \begin{aligned}
\partial_t^3 \mathbf{c_0}
	- (2 D_1 + D_2) \mathbf{\ddot c_0}
	+ [C_1^2 + C_2^2 + D_1 ( D_1 + D_2)] \mathbf{\dot c_0}
	- C_1^2 (D_1 + D_2) \mathbf{c_0}
	&= 0 \\
\partial_t^3 \mathbf{c_1}
	+ (D_1 - D_2) \mathbf{\ddot c_1}
	+ (C_1^2 + C_2^2 - D_1 D_2) \mathbf{\dot c_1}
	+ (C_2^2 D_1 - C_1^2 D_2) \mathbf{c_1}
	&= 0 \\
\partial_t^3 \mathbf{c_2}
	+ (D_1 + 2 D_2) \mathbf{\ddot c_2}
	+ [C_1^2 + C_2^2 + D_2 (D_1 + D_2)] \mathbf{\dot c_2}
	+ C_2^2 (D_1 + D_2) \mathbf{c_2}
	&= 0
\end{aligned} $$

Here ends the analytical side of these derivations.
To finish the problem, we need to do the following:
1. Each of these three third-order differential equations has a third-order auxiliary polynomial, which has three roots $r_j$, which can be solved numerically by standard root-finding techniques.
2. Each of these three roots (assuming they are each unique) contributes one of three particular solutions $e^{r_j t}$.
3. Each of these three particular solutions contributes one of three unknowns $A_j$ to a general solution $c_i(t) = \sum_j A_j e^{r_j t}$.
4. Each of these three unknowns will appear in three linearly-independent equations $∂_t^k c_i(t)$.
5. Each of these three equations has one of three definite solutions at $t=0$, obtained from the initial wavevector $|ψ_0⟩$ and the original system of three differential equations.
6. All of these three solutions form a system of equations, which can be solved numerically with standard algebraic techniques.

> _Three shall be the number thou shalt count, and the number of the counting shall be three._

As a matter of fact, once you've decoupled the differential equations, everything can be done essentially the same way for $m>3$ also.

For $m=3$ (or even $m=4$), the roots can be calculated analytically.
They are a horror to try and write down (I tried!), but they are easily computed.
For $m>4$, there is a proof that the roots _can't_ be calculated analytically.
(:o)
Nevertheless, finding roots of a polynomial has kind-of been the main job of algebra for, um, ever.
I'm embarrassed to admit that I don't actually know any of the systematic algorithms to _do_ it, but they exist and you can certainly find them yourself, either to learn or just to use in your code.

The system of equations that you need to solve can be written as the matrix equation $\mathbf{M} · \vec{A} = \vec{b}$, where $\vec{A}$ is a vector of each $A_j$, $\mathbf{M}$ is a matrix with all the coefficients for each $A_j$ after taking each derivative $∂_t^k|_0$, and $\vec{b}$ is a vector of the three definite solutions to those derivatives obtained from boundary conditions.
The solution is just $\vec{A}=\mathbf{M}^{-1} · \vec{b}$.
As long as the three roots are unique, the matrix $\mathbf{M}$ turns out to be a _Vandermonde_ matrix, a special matrix whose inverse certainly exists and can be easily computed (though not easily written!).
Thus, once you have each set of roots (as long as they are unique), solving for each $c_i(t)$ is essentially trivial.

I spent an embarrassing amount of time trying to determine conditions when the roots are _not_ unique.
I wasn't able to prove it (the analytical roots of a cubic polynomial are _not_ fun to work with!), but my intuition is that it never happens for realistic values of $Ω$, $Δ$, etc..
_If_ it were to, however, it's no big problem - you just have to write out a modified general solution.
I think that just involves replacing one of the degenerate $e^{rt}$ with $t e^{rt}$?
If somehow all _three_ roots are degenerate, the third particular solution would be $t^2 e^{rt}$, and so on for more degeneracies in larger $m$.
That's from memory, though; I don't remember if it's correct or how to derive it.
Assuming it's correct, the system of equations will lose its Vandermonde property so I can't guarantee off the top of my head that a solution is guaranteed, but even then it's just a matter of asking the computer to invert the matrix "the long way".
If it turns out to be singular...well, you figured out how to make a black hole with a quantum computer...