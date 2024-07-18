@testset "SubjectiveScaleModels.jl" begin
    @test begin
        using Distributions
        pdf(BetaMuPhi(0.5, 2), 0.5) ≈ 1.0
    end
end
