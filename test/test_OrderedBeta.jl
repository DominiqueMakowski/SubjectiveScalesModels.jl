@testset "OrderedBeta" begin
    @test begin
        # using Test
        using Distributions

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

        # Test random
        # using CairoMakie
        # d = OrderedBeta(0.5, 3, _logistic(-1), _logistic(1))
        # fig = hist(rand(d, 1000), color=:forestgreen, normalization=:pdf, bins=10)
        # lines!(range(0, 1, length=1000), pdf.(d, range(0, 1, length=1000)), color=:red)
        # fig

    end
end
