module QuadEig

using LinearAlgebra, SparseArrays, SuiteSparse

export quadpencil, pqr, linearize, deflate

### QuadPencil #############################################################################
# Encodes the quadratic pencil problem
############################################################################################
struct QuadPencil{T<:Complex,M<:AbstractMatrix{T}}
    A0::M
    A1::M
    A2::M
end

function quadpencil(A0, A1, A2)
    size(A0) == size(A1) == size(A2) && size(A0, 1) == size(A0, 2) ||
        throw(DimensionMismatch("Expected square matrices of the same size"))
    ptype = complex(promote_type(eltype(A0), eltype(A1), eltype(A2)))
    A0´ = unwrap_adjoint(A0, ptype)
    A1´ = unwrap_adjoint(A1, ptype)
    A2´ = unwrap_adjoint(A2, ptype)
    return QuadPencil(A0´, A1´, A2´)
end

unwrap_adjoint(A::Adjoint, ::Type{T}) where {T} = copy!(similar(A, T, size(A)), A)
unwrap_adjoint(A::AbstractMatrix{T´}, ::Type{T}) where {T´,T} = copy!(similar(A, T, size(A)), A)
unwrap_adjoint(A::AbstractMatrix{T}, ::Type{T}) where {T} = A

Base.size(q::QuadPencil, n...) = size(q.A0, n...)

### QuadPencilQR ###########################################################################
# pivoted QR-factorization
############################################################################################
struct QuadPencilPQR{T,M,Q<:Factorization{T}}
    pencil::QuadPencil{T,M}
    qr0::Q
    qr2::Q
end

function pqr(pencil::QuadPencil)
    qr0 = pqr!(copy(pencil.A0))
    qr2 = pqr!(copy(pencil.A2))
    return QuadPencilPQR(pencil, qr0, qr2)
end

pqr!(a::SparseMatrixCSC) = qr(a)
pqr!(a) = qr!(a, Val(true))

### Linearized Pencils #####################################################################
# Build second companion linearization, or its Q*C2*V rotation
############################################################################################
struct Linearization{T,M<:AbstractMatrix{T}}
    # Q::M
    A::M
    B::M
    V::M
end

linearize(A0, A1, A2; kw...) = linearize(quadpencil(A0, A1, A2); kw...)
linearize(p::QuadPencil; qr = false) = qr ? linearize(pqr(p)) : linearizeC2(p)

function linearize(q::QuadPencilPQR)
    A0, A1, A2 = q.pencil.A0, q.pencil.A1, q.pencil.A2
    o, z = one(A1), zero(A1)
    # Q0, Q2´ = getQ(q.qr0), getQ´(q.qr2)
    Q0, Q0´, Q2´ = getQ(q.qr0), getQ´(q.qr0), getQ´(q.qr2)
    RP0, RP2 = getRP´(q.qr0), getRP´(q.qr2)
    V = [o z; z Q0]
    Q = [Q2´ z; z Q0´]
    A = [Q2´*A1 -Q2´*Q0; RP0 z]
    B = [RP2 z; z o]
    B .= .- B
    # l = linearizeC2(q.pencil)
    # A = Q * l.A * V
    # B = Q * l.B * V
    # return Linearization(Q, A, B, V)
    return Linearization(A, B, V)
end

function linearizeC2(p::QuadPencil)
    n = size(p, 1)
    o, z = one(p.A1), zero(p.A1)
    A = [p.A1 -o; p.A0 z]
    B = [-p.A2 z; z -o]
    V = one(A)
    Q = one(A)
    # return Linearization(Q, A, B, V)
    return Linearization(A, B, V)
end

function Base.show(io::IO, l::Linearization{T,M}) where {T,M}
    print(io, summary(l), "\n",
"  Matrix size    : $(size(l.A, 1)) × $(size(l.A, 2))
  Matrix type    : $M
  QR-transformed : $(l.V != I)
  Deflated       : $(deflationstring(l))")
end

Base.summary(l::Linearization{T}) where {T} =
    "Linearization{T}: second companion linearization of quadratic pencil"

matrixtype(::Linearization{<:Any,M}) where {M} = M

deflationstring(l::Linearization) =
    size(l.A) == size(l.V) ? "false" : "true ($(size(l.V, 1)) -> $(size(l.A, 1)))"

Base.size(l::Linearization, n...) = size(l.A, n...)

### deflate ################################################################################
# compute deflated pencil
############################################################################################
deflate(A0, A1, A2; kw...) = deflate(quadpencil(A0, A1, A2); kw...)
deflate(q::QuadPencil; kw...) = deflate(linearize(q; qr = true); kw...)

function deflate(l::Linearization{T}; atol = sqrt(eps(real(T)))) where {T}
    r0, r2 = nonzero_rows(l, atol)
    n = size(l, 1) ÷ 2
    s = n - r2
    X = view(l.A, 1+r2:n, 1:r2+s+r0) # [X21 X22 X23]
    ZX´ = fod_z´(X, 1+s:s+r0+r2)
    deflatedAB(l.A, l.B, ZX´, r0, r2, s)
    A, B = deflatedAB(l.A, l.B, ZX´, r0, r2, s)
    V = view(l.V, :, 1:n+r0) * ZX´
    # Q = l.Q[[1:r2; n+1:n+r0], :]
    # return Linearization(Q, A, B, V)
    return Linearization(A, B, V)
end

# function deflate_full(l::Linearization{T}; atol = sqrt(eps(real(T)))) where {T}
#     r0, r2 = nonzero_rows(l, atol)
#     n = size(l, 1) ÷ 2
#     s = n - r2
#     X = view(l.A, 1+r2:n, 1:r2+s+r0) # [X21 X22 X23]
#     QX´, ZX´ = fod_qz´(X)
#     # ZXfull´ = [ZX´ spzeros(s+r2+r0, n-r0); spzeros(n-r0, s+r2+r0) I(n-r0)] * I(2n)[[1+s:s+r0+r2; 1:s; 1+n+r0:2n],:]
#     ZXfull´ = [ZX´ spzeros(s+r2+r0, n-r0); spzeros(n-r0, s+r2+r0) I(n-r0)][:,[1+s:s+r0+r2; 1:s; 1+n+r0:2n]]
#     QXfull´ = [I(r2) spzeros(r2,s) spzeros(r2,r0) spzeros(r2,n-r0);
#                spzeros(r0,r2) spzeros(r0,s) I(r0) spzeros(r0,n-r0);
#                spzeros(s, r2) QX´ spzeros(s,r0) spzeros(s,n-r0);
#                spzeros(n-r0,r2) spzeros(n-r0,s) spzeros(n-r0, r0) I(n-r0)]
#     A = QXfull´[1:r0+r2, :] * l.A * ZXfull´[:,1:r0+r2]
#     B = QXfull´[1:r0+r2, :] * l.B * ZXfull´[:,1:r0+r2]
#     V = l.V * ZXfull´
#     # Q = QXfull´ * l.Q
#     # return Linearization(Q, A, B, V)
#     return Linearization(A, B, V)
# end

# get some columns of Z' in a full orthogonal decomposition of X
fod_z´(X::SparseMatrixCSC, cols = :) = _fod_z´(sparse(X'), cols)
fod_z´(X::SubArray{<:Any,2,<:SparseMatrixCSC}, cols = :) = _fod_z´(sparse(X'), cols)
fod_z´(X, cols = :) = _fod_z´(X', cols)
_fod_z´(X´, cols) = getQ(qr(X´), cols)

# fod_qz´(X::SparseMatrixCSC) = _fod_qz´(sparse(X'))
# fod_qz´(X::SubArray{<:Any,2,<:SparseMatrixCSC}) = _fod_qz´(sparse(X'))
# fod_qz´(X) = _fod_qz´(X')
# function _fod_qz´(X´)
#     q = qr(X´)
#     Z´ = getQ(q)
#     RP´ = getRP´(q)
#     q´ = qr(sparse(RP´'))
#     Q´ = getQ´(q´)
#     return Q´, Z´
# end

# carry out quadeig deflation
function deflatedAB(lA::SparseMatrixCSC, lB, ZX´, r0, r2, s)
    selected_rows = Isparse(size(lA, 1), [1:r2; 1+s+r2:s+r0+r2], :)
    tmp = selected_rows * lA
    # display(sparse(tmp))
    A = view(selected_rows * lA, :, 1:s+r0+r2) * ZX´
    B = view(selected_rows * lB, :, 1:s+r0+r2) * ZX´
    # display(sparse(ZX´))
    return A, B
end

function deflatedAB(lA::AbstractMatrix, lB, ZX´, r0, r2, s)
    A = similar(lA, r0+r2, r0+r2)
    mul!(view(A, 1:r2, :), view(lA, 1:r2, 1:s+r0+r2), ZX´)
    mul!(view(A, 1+r2:r0+r2, :), view(lA, 1+s+r2:s+r0+r2, 1:s+r0+r2), ZX´)
    B = similar(A)
    mul!(view(B, 1:r2, :), view(lB, 1:r2, 1:s+r2), view(ZX´, 1:s+r2, :))
    B[1+r2:r0+r2, :] .= ZX´[1+s+r2:s+r0+r2, :]
    chop!(A)
    chop!(B)
    return A, B
end

### Tools ################################################################################

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

Isparse(n, rows, cols) = _Isparse(n, inds(rows, n), inds(cols, n))

inds(::Colon, n) = 1:n
inds(is, n) = is

function _Isparse(n, rows, cols)
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
