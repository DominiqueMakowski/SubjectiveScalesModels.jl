@testset "data_rescale" begin
    @test begin
        data_rescale([1, 2, 3]) == [0.0, 0.5, 1.0]
        data_rescale([1, 2, 3]; old_range=[1, 6], new_range=[1, 0]) == [1.0, 0.8, 0.6]
        data_rescale([1.0, 2.0, 3], new_range=[0, 1]) == [0.0, 0.5, 1.0]
    end
end
