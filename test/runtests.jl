using QuadEig
using Test, LinearAlgebra, SparseArrays

function check(A₀, A₁, A₂; atol = sqrt(eps(real(eltype(A₀)))))
    A, B, V = linearize(A₀, A₁, A₂)
    Ad, Bd, Vd = deflate(A₀, A₁, A₂; atol)
    λ, φ = eigen(Matrix(A), Matrix(B), sortby = abs)
    λd, φd = eigen(Matrix(Ad), Matrix(Bd), sortby = sortby = abs)
    v = normalize_vecs(V * φ)
    vd = normalize_vecs(Vd * φd)
    tests = Bool[]
    for (id, ld) in enumerate(λd)
        (isnan(ld) || !(atol <= abs(ld) <= inv(atol))) && continue
        test = false
        for (i, l) in enumerate(λ)
            if ≈(l, ld) && ≈(v[:, i], vd[:, id])
                test = true
                break
            end
        end
        push!(tests, test)
    end
    return all(tests)
end

function normalize_vecs(φ)
    for col in eachcol(φ)
        normalize!(col)
        col ./= sum(col) # phase fixing
    end
    return φ
end

@testset "QuadEig.jl" begin
    A₀ = rand(6,6); A₁ = rand(6, 6); A₂ = rand(6, 6);
    A₀[:, 1:3] .= A₀[:, 4:6]; A₂[:, 2:3] .= A₂[:, 4:5];
    @test check(A₀, A₁, A₂)

    A₀ = rand(ComplexF64, 6, 6); A₁ = rand(6, 6); A₂ = rand(6, 6);
    A₀[:, 1:3] .= A₀[:, 4:6]; A₂[:, 2:3] .= A₂[:, 4:5];
    @test check(A₀, A₁, A₂)

    A₀ = sprand(ComplexF64, 20, 20, 0.1); A₁ = sprand(ComplexF64, 20, 20, 0.1); A₂ = sprand(ComplexF64, 20, 20, 0.1);
    A₀[:, 1:3] .= A₀[:, 4:6]; A₂[:, 2:3] .= A₂[:, 4:5];
    @test check(A₀, A₁, A₂)

    A₀ = sprand(ComplexF64, 100, 100, 0.1); A₁ = sprand(ComplexF64, 100, 100, 0.1); A₂ = sprand(ComplexF64, 100, 100, 0.1);
    A₀[:, 1:3] .= A₀[:, 4:6]; A₂[:, 2:3] .= A₂[:, 4:5];
    @test check(A₀, A₁, A₂)
end
