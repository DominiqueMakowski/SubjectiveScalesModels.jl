@testset "BetaPhi2" begin
    @test begin
        # using Test
        using Distributions
        pdf(BetaPhi2(0.5, 1), 0.5) ≈ 1.0

        mean(BetaPhi2(0.5, 1)) ≈ 0.5
        mean(BetaPhi2(0.7, 3)) ≈ 0.7
        mean(BetaPhi2(0.7, 0.5)) ≈ 0.7

        @test_throws DomainError BetaPhi2(-1.0, 1.0)
        logpdf(BetaPhi2(eps(), 1.0), 0) ≈ -Inf
    end
end
