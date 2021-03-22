module QuadEig

using LinearAlgebra, SparseArrays

### QuadPencil
# Encodes the quadratic pencil problem
###
struct QuadPencil{T>:Complex,M<:AbstractMatrix{T}}
    A0::M
    A1::M
    A2::M
end

function quadpencil(A0, A1, A2)
    size(A0) == size(A1) == size(A2) && issquare(A0) ||
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

### QuadPencilQR
# pivoted QR-factorization
###
struct QuadPencilPQR{T,M,Q<:Factorization{T}}
    pencil::QuadPencil{T,M}
    qr0::Q
    qr2::Q
end

function quadpencilPQR(pencil::QuadPencil{T}; kw...) where {T}
    qr0 = pqr!(copy(q.A0))
    qr2 = pqr!(copy(q.A2))
    return QuadPencilQR(pencil, qr0, qr2)
end

pqr!(a::SparseMatrix) = qr!(a)
pqr!(a) = qr!(a, Val(true))

### LinearizedPencil
# Build second companion linearization, or Q*C2*V rotation thereof
###
struct LinearizedPencil{T,M<:AbstractMatrix{T}}
    A::M
    B::M
    V::M
end

function linearizedpencil(p::QuadPencil)
    n = size(p, 1)
    A = [p.A1 -I(n); p.A0 0I(n)]
    B = [-p.A2 0I(n); 0I(n) -I(n)]
    V = [I(n) 0I(n); 0I(n) I(n)]
    return LinearizedPencil(A, B, V)
end

function linearizedpencil(q::QuadPencilPQR{T}) where {T}
    A0, A1, A2 = q.pencil.A0, q.pencil.A1, q.pencil.A2
    o, z = one(A1), zero(A1)
    Q0, Q2´ = getQ(q.qr0), getQ´(q.qr2)
    RP0, RP2 = getRP´(qrA0), getRP´(qrA2)
    V = [o z; z Q0]
    A = [Q2´*A0 -Q2´*Q0; RP0 z]
    B = [RP2 z; z o]
    B .= .- B
    return LinearizedPencil(A, B, V)
end

function linearizedpencil(q::QuadPencilPQR{T,<:SparseMatrix}) where {T}
    A0, A1, A2 = q.pencil.A0, q.pencil.A1, q.pencil.A2
    o, z = one(A1), zero(A1)
    Q0, Q2 = q.qr0.Q, q.qr2.Q
    RP0, RP2 = qrA0.R * qrA0.P', qrA2.R * qrA2.P'
    V = [o z; z Q0]
    A = [Q2'A0 -Q2'Q0; RP0 z]
    B = [RP2 z; z o]
    B .= .- B
    return LinearizedPencil(A, B, V)
end

getQ(qr::QRPivoted) = qr.Q
getQ(qr::QRSuiteSparse.SPQR.QRSparse) =  I(size(qr,1))[:, qr.prow] * sparse(Matrix(qr.Q))

getQ´(qr::QRPivoted) = qr.Q'
getQ´(qr::QRSuiteSparse.SPQR.QRSparse) = sparse(Matrix(qr.Q)') * I(size(qr,1))[qr.prow,:]


getRP´(qr::QRPivoted) = qr.R * qr.P'
getRP´(qr::SuiteSparse.SPQR.QRSparse) = qr.R * I(size(qr,1))[qr.pcol, :]

getPR´(qr::QRPivoted) = qr.P * qr.R'
getPR´(qr::SuiteSparse.SPQR.QRSparse) = I(size(qr,1))[:, qr.pcol] * qr.R'

Base.size(l::LinearizedPencil, n...) = size(l.A, n...)

### deflate
# compute deflated pencil
###
function deflate(l::LinearizedPencil{T}, atol = sqrt(real(eps(T))))
    rank0, rank2 = nonzero_rows(l, atol)
    rank0 == rank2 || throw(ArgumentError("The case with inhomogeneous ranks not yet supported"))
    r = rank0
    n = size(l, 1) ÷ 2
    X = view(l.A, r+1:n, 1:(n+r)) # [X21 X22 X23]
    QX, ZX = qz(X)
end

function nonzero_rows(l::LinearizedPencil{T}, atol = sqrt(real(eps(T)))) where {T}
    n = size(l, 2) ÷ 2
    rank0 = 0
    for row in n+1:2n
        all(z -> abs(z) < atol, view(l.A, row, 1:n)) && break
        rank0 += 1
    end
    rank2 = 0
    for row in 1:n
        all(z -> abs(z) < atol, view(l.B, row, 1:n)) && break
        rank0 += 1
    end
    return rank0, rank2
end

qz(X::SubArray{T,2,<:SparseMatrix}) = qz(sparse(X))

function qz(X::SparseMatrix)
    qrX = qr(X)
    PR´ = getPR´(qrX)
    lq´ = qr(PR´)
    Q = sparse(qrX)
    Z = sparse(lq´.Q')
    return Q, Z
end

function qz(X)
    qrX = qr(X, Val(true))
    PR = getRP´(qrX)
    lq = lq(PR)
    Q = Matrix(qrX)
    Z = lq.Z * Matrix(I, size(X, 2), size(X, 2))
    return Q, Z
end

