@testset "BetaPhi2" begin
    @test begin
        using Distributions
        pdf(BetaPhi2(0.5, 1), 0.5) ≈ 1.0
    end
end
