@testset "Choco" begin
    @test begin
        # using Test
        using Distributions
        # pdf(BetaPhi2(0.5, 1), 0.5) ≈ 1.0

        mean(Choco()) ≈ 0.5
        mean(Choco(p1=0.5, μ0=0.3, ϕ0=1, μ1=0.3, ϕ1=1)) ≈ 0.5

        # @test_throws DomainError BetaPhi2(-1.0, 1.0)
        # logpdf(BetaPhi2(eps(), 1.0), 0) ≈ -Inf
    end
end
