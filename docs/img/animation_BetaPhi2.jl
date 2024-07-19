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



# BetaPhi2 =====================================================================================

# Figure
fig = Figure()

μ = Observable(0.5)
ϕ = Observable(1.0)

ax1 = Axis(
    fig[1:2, 2],
    title=@lift("BetaPhi2(μ = $(round($μ, digits = 1)), ϕ =  $(round($ϕ, digits = 1)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)
xlims!(ax1, 0, 1)
ylims!(ax1, 0, 8)
x = range(0, 1, length=1000)
y = @lift(pdf.(BetaPhi2($μ, $ϕ), x))
band!(ax1, x, 0, y, color="#2196F3")


ax2 = Axis(
    fig[1, 1],
    title="Parameter Space",
    xlabel="μ",
    ylabel="ϕ"
)

function pdf_concave(d)
    dx = diff(pdf.(d, range(0, 1, length=10)))
    dx2 = diff(dx)
    dx2 = dx2[isfinite.(dx2)]
    return mean(dx2)
end

contourf!(ax2,
    range(0.05, 0.95, 2000),
    range(0.05, 12, 2000),
    [pdf_concave(BetaPhi2(μ, ϕ)) for μ in range(0.05, 0.95, 2000), ϕ in range(0.05, 12, 2000)],
    colormap=:balance,
    levels=range(-1.5, 1.5, length=41))

vlines!(ax2, [0.5], color=:purple)
hlines!(ax2, [1], color=:purple)
scatter!(ax2, @lift([$μ]), @lift([$ϕ]), color=:red, markersize=10, marker=:cross)


ax3 = Axis(
    fig[2, 1],
    title="Traditional Beta Parameters",
    xlabel="α",
    ylabel="β"
)

ablines!(ax3, [0], [1], color=:purple)
ablines!(ax3, [2], [-1], color=:purple)
vlines!(ax3, [1], color=:black, linestyle=:dash)
hlines!(ax3, [1], color=:black, linestyle=:dash)
scatter!(ax3,
    @lift([params(BetaPhi2($μ, $ϕ))[1]]),
    @lift([params(BetaPhi2($μ, $ϕ))[2]]),
    color=:red, markersize=10, marker=:cross)
xlims!(ax3, 0, 20)
ylims!(ax3, 0, 20)

fig



# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.10
        ϕ[] = change_param(frame; frame_range=(0.0, 0.10), param_range=(1.00, 0.5))
    end
    if frame >= 0.15 && frame < 0.25
        μ[] = change_param(frame; frame_range=(0.15, 0.25), param_range=(0.5, 0.05))
    end
    if frame >= 0.30 && frame < 0.40
        μ[] = change_param(frame; frame_range=(0.30, 0.40), param_range=(0.05, 0.90))
    end
    if frame >= 0.45 && frame < 0.55
        ϕ[] = change_param(frame; frame_range=(0.45, 0.55), param_range=(0.50, 11.0))
    end
    if frame >= 0.60 && frame < 0.65
        μ[] = change_param(frame; frame_range=(0.60, 0.65), param_range=(0.90, 0.5))
    end
    # Return to normal
    if frame >= 0.7 && frame < 0.9
        # μ[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.90, 0.5))
        ϕ[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(11.00, 1.0))
    end
end

# animation settings
frames = range(0, 1, length=240)
record(make_animation, fig, "animation_BetaPhi2.gif", frames; framerate=15)

