using CSV
using DataFrames
using Distributions
using GLMakie
using Downloads
using Random
using SubjectiveScalesModels

# Data ==========================================================================================
cd(@__DIR__)

function rescale_param(p; original_range=(-1, 1), new_range=(-3, 3))
    p = (p - original_range[1]) / (original_range[2] - original_range[1])
    p = p * (new_range[2] - new_range[1]) + new_range[1]
    return p
end

function change_param(frame; frame_range=(0, 1), param_range=(0, 1))
    frame = rescale_param(frame; original_range=frame_range, new_range=(1π, 2π))
    p = rescale_param(cos(frame); original_range=(-1, 1), new_range=param_range)
    return p
end



# ExtremeBeta =====================================================================================

# Figure
fig = Figure()

μ = Observable(0.5)
ϕ = Observable(3.0)
k0 = Observable(0.0)
k1 = Observable(0.0)

ax1 = Axis(
    fig[1, 1],
    title=@lift("ExtremeBeta(μ = $(round($μ, digits = 1)), ϕ =  $(round($ϕ, digits = 1)), k0 = $(round($k0, digits = 3)), k1 = $(round($k1, digits = 3)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)

hist!(ax1, @lift(rand(ExtremeBeta($μ, $ϕ, $k0, $k1), 40_000)), bins=200, normalization=:pdf, color=:grey)

ax2 = Axis(
    fig[2, 1],
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)

hist!(ax2, @lift(rand(ExtremeBeta($μ, $ϕ, $k0, $k1), 40_000)), bins=200, normalization=:pdf, color=:grey)
vlines!(ax2, @lift([$k0]), color=:red)
xlims!(ax2, 0, 0.3)
ylims!(ax2), 0, 10
fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.3
        k0[] = change_param(frame; frame_range=(0.0, 0.3), param_range=(0.0, 0.1))
    end
    if frame >= 0.35 && frame < 0.65
        k1[] = change_param(frame; frame_range=(0.35, 0.65), param_range=(0.0, 0.1))
        k0[] = change_param(frame; frame_range=(0.35, 0.65), param_range=(0.1, 0.0))
    end
    # Return to normal
    if frame >= 0.7 && frame < 0.9
        k1[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.1, 0.0))
    end
    ylims!(ax1)

end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_ExtremeBeta.gif", frames; framerate=15)

