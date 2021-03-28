module QuadEig

using LinearAlgebra, SparseArrays, SuiteSparse

export linearize, deflate, eigbasis

### PencilScaling ##########################################################################
# Compute the optimal scaling
############################################################################################
struct PencilScaling{T}
    γ::T
    δ::T
end

function pencilscaling(A0::M, A1::M, A2::M; scaling = nothing) where {T,M<:AbstractMatrix{T}}
    scaling === nothing && return PencilScaling(T(1), T(1))
    n0, n1, n2 = opnorm.(Matrix.((A0, A1, A2)))
    τ = n1 / sqrt(n0 * n2)
    if τ <= 1
        γ = sqrt(n0 / n2)
        δ = 2 / (n0 + γ * n1)
    elseif scaling === -
        # distinct tropical roots, choose γ+ to favor large eigenvalues
        γ = n1 / n2
        δ = 1 / max(n2 * γ^2, n1 * γ, n0)
    else
        # distinct tropical roots, choose γ- to favor small eigenvalues
        γ = n0 / n1
        δ = 1 / max(n2 * γ^2, n1 * γ, n0)
    end
    return PencilScaling(T(γ), T(δ))
end

function scale!(A0, A1, A2, s::PencilScaling)
    γ, δ = s.γ, s.δ
    if γ != 1 || δ != 1
        A0 .*= γ^2 * δ
        A1 .*= γ * δ
        A2 .*= δ
    end
    return A0, A1, A2
end

### QuadPencil #############################################################################
# Encodes the quadratic pencil problem
############################################################################################
struct QuadPencil{T<:Complex,M<:AbstractMatrix{T}}
    A0::M
    A1::M
    A2::M
    scaling::PencilScaling{T}
end

function quadpencil(A0, A1, A2; kw...)
    size(A0) == size(A1) == size(A2) && size(A0, 1) == size(A0, 2) ||
        throw(DimensionMismatch("Expected square matrices of the same size"))
    ptype = complex(promote_type(eltype(A0), eltype(A1), eltype(A2)))
    A0´ = copy_to_eltype(A0, ptype)
    A1´ = copy_to_eltype(A1, ptype)
    A2´ = copy_to_eltype(A2, ptype)
    s = pencilscaling(A0´, A1´, A2´; kw...)
    scale!(A0´, A1´, A2´, s)
    return QuadPencil(A0´, A1´, A2´, s)
end

# This also ensures any Adjoint/Transpose wrapper is removed
copy_to_eltype(A, ::Type{T}) where {T} = copy!(similar(A, T, size(A)), A)

Base.size(q::QuadPencil, n...) = size(q.A0, n...)

### QuadPencilQR ###########################################################################
# Pivoted QR-factorization
############################################################################################
struct QuadPencilPQR{T,M}
    pencil::QuadPencil{T,M}
    Q0::M
    RP0::M
    Q2´::M
    RP2::M
end

function pqr(pencil::QuadPencil)
    qr0 = pqr(pencil.A0)
    qr2 = pqr(pencil.A2)
    Q0, RP0 = getQ(qr0), getRP´(qr0)
    Q2´, RP2 = getQ´(qr2), getRP´(qr2)
    return QuadPencilPQR(pencil, Q0, RP0, Q2´, RP2)
end

# Sparse QR from SparseSuite is also pivoted
pqr(a::SparseMatrixCSC) = qr(a)
pqr(a) = qr!(copy(a), Val(true))

### Linearized Pencils #####################################################################
# Build second companion linearization, or its Q*C2*V rotation
############################################################################################
struct Linearization{T,M<:AbstractMatrix{T}}
    A::M
    B::M
    V::M
    scaling::PencilScaling{T}
end

linearize(A0, A1, A2; kw...) = linearize(pqr(quadpencil(A0, A1, A2; kw...)))

function linearize(q::QuadPencilPQR)
    A0, A1, A2 = q.pencil.A0, q.pencil.A1, q.pencil.A2
    o, z = one(A1), zero(A1)
    V = [o z; z q.Q0]
    A = [q.Q2´*A1 -q.Q2´*q.Q0; q.RP0 z]
    B = [q.RP2 z; z o]
    B .= .- B
    return Linearization(A, B, V, q.pencil.scaling)
end

function Base.show(io::IO, l::Linearization{T,M}) where {T,M}
    print(io, summary(l), "\n",
"  Matrix size    : $(size(l.A, 1)) × $(size(l.A, 2))
  Matrix type    : $M
  Scalings γ, δ  : $(real(l.scaling.γ)), $(real(l.scaling.δ))
  Deflated       : $(deflationstring(l))")
end

Base.summary(l::Linearization{T}) where {T} =
    "Linearization{T}: second companion linearization of quadratic pencil"

deflationstring(l::Linearization) =
    size(l.A) == size(l.V) ? "false" : "true ($(size(l.V, 1)) -> $(size(l.A, 1)))"

Base.size(l::Linearization, n...) = size(l.A, n...)

### deflate ################################################################################
# Compute deflated pencil
############################################################################################
deflate(A0, A1, A2; kw...) = deflate(linearize(A0, A1, A2); kw...)

function deflate(l::Linearization{T}; atol = sqrt(eps(real(T)))) where {T}
    n = size(l, 1) ÷ 2
    r0, r2 = nonzero_rows(l, atol)
    r0 == n || r2 == n && return l
    s = n - r2
    X = view(l.A, 1+r2:n, 1:r2+s+r0) # [X21 X22 X23]
    ZX´ = fod_z´(X, 1+s:s+r0+r2)
    A, B = deflatedAB(l.A, l.B, ZX´, r0, r2, s)
    V = view(l.V, :, 1:n+r0) * ZX´
    return Linearization(A, B, V, l.scaling)
end

# get some columns of Z' in a full orthogonal decomposition of X
fod_z´(X::SparseMatrixCSC, cols = :) = _fod_z´(sparse(X'), cols)
fod_z´(X::SubArray{<:Any,2,<:SparseMatrixCSC}, cols = :) = _fod_z´(sparse(X'), cols)
fod_z´(X, cols = :) = _fod_z´(X', cols)
_fod_z´(X´, cols) = getQ(qr(X´), cols)

# carry out quadeig deflation
function deflatedAB(lA::SparseMatrixCSC, lB, ZX´, r0, r2, s)
    selected_rows = Isparse(size(lA, 1), [1:r2; 1+s+r2:s+r0+r2], :)
    A = view(selected_rows * lA, :, 1:s+r0+r2) * ZX´
    B = view(selected_rows * lB, :, 1:s+r0+r2) * ZX´
    return A, B
end

function deflatedAB(lA::AbstractMatrix, lB, ZX´, r0, r2, s)
    A = similar(lA, r0+r2, r0+r2)
    mul!(view(A, 1:r2, :), view(lA, 1:r2, 1:r2+s+r0), ZX´)
    mul!(view(A, 1+r2:r0+r2, :), view(lA, 1+s+r2:r2+s+r0, 1:r2+s+r0), ZX´)
    B = similar(A)
    mul!(view(B, 1:r2, :), view(lB, 1:r2, 1:r2+s+r0), ZX´)
    mul!(view(B, 1+r2:r0+r2, :), view(lB, 1+s+r2:r2+s+r0, 1:r2+s+r0), ZX´)
    return A, B
end

### quadeig ################################################################################
# Generalized Schur factorization (aka QZ factorization)
############################################################################################
function eigbasis(l::Linearization{T}; filter = missing) where {T}
    n = size(l.V, 1) ÷ 2
    s = schur(Matrix(l.A), Matrix(l.B))
    γ = l.scaling.γ
    λ = γ .* s.α ./ s.β
    if filter !== missing
        which = filter.(λ)
        howmany = count(which)
        ordschur!(s, which)
        resize!(s.α, howmany)
        resize!(s.β, howmany)
        λ = γ .*  s.α ./ s.β
        basis = view(l.V, 1:n, :) * view(s.Z, :, 1:howmany)
    else
        basis = view(l.V, 1:n, :) * s.Z
    end
    chop!(basis)
    return λ, basis
end

### Tools ##################################################################################

getQ(qr::Factorization, cols = :) = qr.Q * Idense(size(qr, 1), cols)
getQ(qr::SuiteSparse.SPQR.QRSparse, cols = :) =  Isparse(size(qr, 1), :, qr.prow) * sparse(qr.Q * Idense(size(qr, 1), cols))

getQ´(qr::Factorization, cols = :) = qr.Q' * Idense(size(qr, 1), cols)
getQ´(qr::SuiteSparse.SPQR.QRSparse, cols = :) = sparse((qr.Q * Idense(size(qr, 1), cols))') * Isparse(size(qr,1), qr.prow, :)

getRP´(qr::Factorization) = qr.R * qr.P'
getRP´(qr::SuiteSparse.SPQR.QRSparse) = qr.R * Isparse(size(qr, 2), qr.pcol, :)

getPR´(qr::Factorization) = qr.P * qr.R'
getPR´(qr::SuiteSparse.SPQR.QRSparse) = Isparse(size(qr, 2), :, qr.pcol) * qr.R'

Idense(n, ::Colon) = Matrix(I, n, n)

function Idense(n, cols)
    m = zeros(Bool, n, length(cols))
    for (j, col) in enumerate(cols)
        m[col, j] = true
    end
    return m
end

# Equivalent to I(n)[rows, cols], but faster
Isparse(n, rows, cols) = Isparse(inds(rows, n), inds(cols, n))

function Isparse(rows, cols)
    rowval = Int[]
    nzval = Bool[]
    colptr = Vector{Int}(undef, length(cols) + 1)
    colptr[1] = 1
    for (j, col) in enumerate(cols)
        push_rows!(rowval, nzval, rows, col)
        colptr[j+1] = length(rowval) + 1
    end
    return SparseMatrixCSC(length(rows), length(cols), colptr, rowval, nzval)
end

inds(::Colon, n) = 1:n
inds(is, n) = is

function push_rows!(rowval, nzval, rows::AbstractUnitRange, col)
    if col in rows
        push!(rowval, 1 + rows[1 + col - first(rows)] - first(rows))
        push!(nzval, true)
    end
    return nothing
end

function push_rows!(rowval, nzval, rows, col)
    for (j, row) in enumerate(rows)
        if row == col
            push!(rowval, j)
            push!(nzval, true)
        end
    end
    return nothing
end

function nonzero_rows(l::Linearization{T}, atol = sqrt(eps(real(T)))) where {T}
    n = size(l, 2) ÷ 2
    rank0, rank2 = nonzero_rows(view(l.A, n+1:2n, 1:n), atol), nonzero_rows(view(l.B, 1:n, 1:n), atol)
    return rank0, rank2
end

function nonzero_rows(p::QuadPencilPQR{T}, atol = sqrt(eps(real(T)))) where {T}
    rank0, rank2 = nonzero_rows(p.qr0.R, atol), nonzero_rows(p.qr2.R, atol)
    return rank0, rank2
end

function nonzero_rows(m::AbstractMatrix{T}, atol = sqrt(eps(real(T)))) where {T}
    n = 0
    for row in eachrow(m)
        all(z -> abs(z) < atol, row) && break
        n += 1
    end
    return n
end

function chop!(A::AbstractArray{T}, atol = sqrt(eps(real(T)))) where {T}
    for (i, a) in enumerate(A)
        abs(a) < atol && (A[i] = zero(T))
    end
    return A
end

end # Module
