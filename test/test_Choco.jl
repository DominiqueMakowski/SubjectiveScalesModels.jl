@testset "Choco" begin
    @test begin
        # using Test
        using Distributions

        mean(Choco()) ≈ 0.5
        mean(Choco(p1=0.5, μ0=0.3, ϕ0=1, μ1=0.3, ϕ1=1)) ≈ 0.5

        pdf(Choco(), 0.5) ≈ 2.0

        # Zero-one inflation
        logpdf(Choco(), 0.0) ≈ -Inf
        logpdf(Choco(), 1.0) ≈ -Inf
        floor(logpdf(Choco(; k0=0.1), 0); digits=5) ≈ -2.30259
        floor(logpdf(Choco(; k1=0.9), 1); digits=5) ≈ -2.30259

    end
end
