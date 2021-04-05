# QuadEig

The QuadEig package implements the `quadeig` algorithm to deflate zero and infinite
eigenvalues of quadratic pencils. The algorithm is published in:

"An Algorithm for the Complete Solution of Quadratic Eigenvalue Problems"
S. Hammarling, C. J. Munro, and F. Tisseur. [ACM Trans. Math. Softw. 39, (2013)](https://dl.acm.org/doi/10.1145/2450153.2450156)

Given a quadratic pencil `Q(λ) = A₀ + λ A₁ + λ² A₂`, where `Aᵢ` are square matrices of size
`N`, we want to solve the quadratic right-eigenvalue problem `Q(λ)φ = 0`. A powerful approach is
to linearize `Q(λ)` into an equivalent linear pencil `L(λ) = A - λB` of size `2N`, and solve
the generalized eigenvalue problem `L(λ)φ´ = (A - λ B)φ´ = 0` instead. There are many
possible linearizations. The eigenvectors `φ´` of `L` will be related to the original `φ` in
a way that depends on the chosen linearization.

The `quadeig` algorithm helps to more efficiently solve the `L(λ)φ´ = 0` problem by
transforming `L(λ)` into a smaller `L₋(λ) = Q L(λ) V` with orthogonal `Q` and `V` operators.
`L₋(λ)` shares the same finite eigenvalues as `L`, but has less (or no) `λ = 0` and `λ = ∞`
eigenvalues which one wants to discard. This process is called "deflation".

The algorithm relies on the specific structure of the so-called second companion
linearization, defined by matrices `A = [A₁ -I; A₀ 0], B = [A₂ 0; 0 -I]` of size `2N`. The
right-eigenvectors of the original problem `Q` are obtained from those of `L` (deflated or
not) by `φ = V * φ´[1:N]`, where `V` is the deflation transformation on the right. For
undeflated linearizations, `V` is not the identity, because a non-deflating transformation
of the second companion linearization is performed for performance reasons.

The QuadEig package exports a `linearize` function to build `L`, and a `deflate` function to
transform an `L` into a deflated `L₋`. The `A`, `B` and `V` matrices of a linearization `l`
can be accessed by `l.A`, `l.B` and `l.V`, or through destructuring `A, B, V = l`.

Example

```julia
julia> using QuadEig, LinearAlgebra

julia> A₀ = rand(6,6); A₁ = rand(6, 6); A₂ = rand(6, 6);

julia> A₀[:, 1:3] .= A₀[:, 4:6];  # This creates 3 λ = 0 eigenvalues

julia> A₂[:, 2:3] .= A₂[:, 4:5];  # This creates 2 λ = ∞ eigenvalues

julia> l = linearize(A₀, A₁, A₂)
Linearization{T}: second companion linearization of quadratic pencil
  Matrix size    : 20 × 20
  Matrix type    : Matrix{ComplexF64}
  Scalings γ, δ  : (1.0, 1.0)
  Deflated       : false

julia> eigvals(l.A, l.B)  # Note the 3 zero (within machine precision) and 2 infinite (NaN) eigenvalues
12-element Vector{ComplexF64}:
     -10.86932670379268 - 7.26632452718938e-15im
    -0.9585605368704543 - 1.3660371264720934im
    -0.9585605368704528 + 1.3660371264720919im
    -0.4087545396926745 + 0.8719854788559378im
    -0.4087545396926742 - 0.8719854788559386im
 -8.591313021709173e-16 + 0.0im
 -7.180754871717689e-17 + 0.0im
 2.0338957932144944e-17 - 0.0im
    0.32412827713495224 - 4.9149302039555125e-17im
      1.281810969835058 + 7.879489919456524e-16im
                    NaN + NaN*im
                    NaN + NaN*im

julia> d = deflate(l)  # or deflate(A₀, A₁, A₂)
Linearization{T}: second companion linearization of quadratic pencil
  Matrix size    : 7 × 7
  Matrix type    : Matrix{ComplexF64}
  Scalings γ, δ  : (1.0, 1.0)
  Deflated       : true (12 -> 7)

julia> eigvals(d.A, d.B)  # The finite eigenvalues are the same, within machine precision
7-element Vector{ComplexF64}:
  -10.869326703792948 - 4.237043003816544e-20im
  -0.9585605368704537 + 1.366037126472092im
  -0.9585605368704532 - 1.366037126472091im
 -0.40875453969267517 + 0.8719854788559379im
 -0.40875453969267417 - 0.8719854788559379im
  0.32412827713495357 - 0.0im
    1.281810969835057 + 1.0657465373859974e-16im
```

The `deflate` function admits an `atol` keyword argument to specify a threshold for
eigenvalues to deflate (`|λ| < atol` for zeros and `|λ| > atol⁻¹` for infinities).
