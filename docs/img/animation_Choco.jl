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

p0 = Observable(0.5)
μ0 = Observable(0.5)
ϕ0 = Observable(1.0)
μ1 = Observable(0.5)
ϕ1 = Observable(1.0)

ax1 = Axis(
    fig[1, 1],
    title=@lift("Choco(p0 = $(round($p0, digits = 1)), μ0 = $(round($μ0, digits = 1)), ϕ0 = $(round($ϕ0, digits = 1)), μ1 = $(round($μ1, digits = 1)), ϕ1 = $(round($ϕ1, digits = 1)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)
ylims!(ax1; low=0)

xaxis = range(0, 1, length=10_000)
lines!(ax1, xaxis, @lift(pdf.(Choco($p0, $μ0, $ϕ0, $μ1, $ϕ1), xaxis)), color=:blue)

fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.2
        ϕ0[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(1.0, 3.0))
        ϕ1[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(1.0, 1 / 3))
    end
    if frame >= 0.25 && frame < 0.35
        p0[] = change_param(frame; frame_range=(0.25, 0.35), param_range=(0.5, 0.1))
    end
    # Return to normal
    if frame >= 0.7 && frame < 0.9
        ϕ0[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(3.0, 1.0))
        ϕ1[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(1 / 3, 1.0))
        p0[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.1, 0.5))
    end
    ylims!(ax1; low=0)
end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_Choco.gif", frames; framerate=15)

