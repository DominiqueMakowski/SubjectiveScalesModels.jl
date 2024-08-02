@testset "OrderedBeta" begin
    @test begin
        # using Test
        using Distributions
        using Random

        # Mean
        mean(OrderedBeta(0.5, 3, 0.1, 1)) - mean(rand(OrderedBeta(0.5, 3, 0.1, 1), 100_000)) < 0.01

        # Shifted
        x = range(0, 1, length=10)
        logpdf.(OrderedBeta(0.5, 3, 0, 1), x) == logpdf.(BetaPhi2(0.5, 3), x)
        logpdf(OrderedBeta(0.5, 3, 0, 1), 0.3) == logpdf(1 + OrderedBeta(0.5, 3, 0, 1), 1.3)

        # logpdf at bounds
        logpdf(OrderedBeta(0.5, 3, 0, 1), 0) ≈ -Inf
        logpdf(OrderedBeta(0.5, 3, 0, 1), 1) ≈ -Inf
        floor(logpdf(OrderedBeta(0.5, 3, 0.1, 1), 0); digits=5) ≈ -2.30259
        floor(logpdf(OrderedBeta(0.5, 3, 0, 0.9), 1); digits=5) ≈ -2.30259

        logpdf(0.5 * OrderedBeta(0.5, 3, 0, 1), 0) ≈ -Inf
        floor(logpdf(0.5 * OrderedBeta(0.5, 3, 0.1, 1), 0); digits=5) ≈ -1.60944

        # Logpdf
        floor(logpdf(OrderedBeta(0.5, 3, 0, 1), 0.3); digits=5) ≈ 0.2799
        floor(logpdf(OrderedBeta(0.5, 3, 0.1, 1), 0.3); digits=5) ≈ 0.17454
        floor(logpdf(OrderedBeta(0.5, 3, 0, 0.9), 0.3); digits=5) ≈ 0.17454
        floor(logpdf(OrderedBeta(0.5, 3, 0.1, 0.9), 0.3); digits=5) ≈ 0.05675

        # Random
        @testset "Random" begin
            x = rand(MersenneTwister(1234), OrderedBeta(0.5, 1.0, 0.0, 0.85), 100_000)
            @test sum(x .== 0) == 0
            @test sum(x .== 1) / length(x) - pdf(OrderedBeta(0.5, 1, 0, 0.85), 1) < 0.01
        end

        dist_zero = OrderedBeta(μ, ϕ, 0.0, k1)
        dist_one = OrderedBeta(μ, ϕ, k0, 1.0)
        dist_both = OrderedBeta(μ, ϕ, 0.0, 1.0)

        sample_zero = rand(dist_zero)
        sample_one = rand(dist_one)
        sample_both = rand(dist_both)

        @test 0 <= sample_zero <= 1
        @test 0 <= sample_one <= 1
        @test 0 <= sample_both <= 1



        # Test against R -----------------------------------------------------------------------
        _logistic(x::Real) = 1 / (1 + exp(-x))

        # ordbetareg::dordbeta(c(0, 1), mu = 0.5, phi = 4, cutpoints = c(-1, 1), log = FALSE)
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(-1), _logistic(1)), 0); digits=6) ≈ 0.268941
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(-1), _logistic(1)), 1); digits=6) ≈ 0.268941
        # ordbetareg::dordbeta(c(0, 1), mu = 0.5, phi = 4, cutpoints = c(-0.5, 1), log = FALSE)
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(-0.5), _logistic(1)), 0); digits=6) ≈ 0.37754
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(-0.5), _logistic(1)), 1); digits=6) ≈ 0.268941
        # ordbetareg::dordbeta(c(0, 1), mu = 0.5, phi = 4, cutpoints = c(0, 2), log = FALSE)
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(0), _logistic(2)), 0); digits=6) ≈ 0.5000000
        floor(pdf.(OrderedBeta(0.5, 1, _logistic(0), _logistic(2)), 1); digits=6) ≈ 0.119202
        # ordbetareg::dordbeta(c(0.5), mu = 0.5, phi = 4, cutpoints = c(0.5, -0.5), log = FALSE)
        pdf(OrderedBeta(0.5, 1, _logistic(0.5), _logistic(-0.5)), 0.5) ≈ 0 # Note: R outputs -0.367378

    end
end
