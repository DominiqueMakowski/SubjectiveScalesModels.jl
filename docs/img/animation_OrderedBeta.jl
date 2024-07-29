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



# OrderedBeta =====================================================================================
# Figure
fig = Figure(size=(1200, 800))

μ = Observable(0.5)
ϕ = Observable(3.0)
k1 = Observable(-5.5)
k2 = Observable(3.5)

ax1 = Axis(
    fig[1:2, 5:6],
    title=@lift("OrderedBeta(μ = $(round($μ, digits = 1)), ϕ =  $(round($ϕ, digits = 1)), k1 = $(round($k1, digits = 2)), k2 = $(round($k2, digits = 2)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)

# hist!(ax1, @lift(rand(OrderedBeta($μ, $ϕ, $k0, $k1), 40_000)), bins=1000, normalization=:pdf, color=:grey)
xaxis = range(0, 1, length=1000)
lines!(ax1, xaxis, @lift(pdf.(OrderedBeta($μ, $ϕ, $k1, $k2), xaxis)), color=:red)


# Contour plot -----------------------------------------------------------------------------------

ax3 = Axis(
    fig[1, 1:2],
    title="Proportion of zeros",
    xlabel="k1",
    ylabel="k2",
    yticksvisible=false,
    xticksvisible=false,
)
contourf!(ax3,
    range(-6, 6, 100),
    range(-6, 6, 100),
    [pdf(OrderedBeta(0.5, 3, k1, k2), 0) for k1 in range(-6, 6, 100), k2 in range(-6, 6, 100)],
    colormap=:amp,
    levels=range(0, 1, length=50))

xlims!(ax3, -6, 6)
ylims!(ax3, -6, 6)

ax4 = Axis(
    fig[2, 1:2],
    title="Proportion of ones",
    xlabel="k1",
    ylabel="k2",
    yticksvisible=false,
    xticksvisible=false
)

contourf!(ax4,
    range(-6, 6, 100),
    range(-6, 6, 100),
    [pdf(OrderedBeta(0.5, 3, k1, k2), 1) for k1 in range(-6, 6, 100), k2 in range(-6, 6, 100)],
    colormap=:amp,
    levels=range(0, 1, length=50))

xlims!(ax4, -6, 6)
ylims!(ax4, -6, 6)

# [(k1, k2) for k2 in range(1, 3, 3), k1 in range(4, 6, 3)]

# Individuals -----------------------------------------------------------------------------------

function _make_axis(row, col)
    return Axis(
        fig[row, col],
        yticksvisible=false,
        xticksvisible=false,
        yticklabelsvisible=false,
    )
end

xaxis = range(0, 1, length=1000)


ax = _make_axis(1, 3)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, -5, 3), xaxis), color=:orange, label="k1 = -5, k2 = 3")
ylims!(0, 2)
axislegend(ax, position=:rt)
_make_axis(1, 4)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0, 3), xaxis), color=:green, label="k1 = 0, k2 = 3")
ylims!(0, 2)
axislegend(ax, position=:rt)
_make_axis(2, 4)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0, 0), xaxis), color=:blue, label="k1 = 0, k2 = 0")
ylims!(0, 2)
axislegend(ax, position=:rt)
_make_axis(2, 3)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, -5, 2), xaxis), color=:purple, label="k1 = -5, k2 = 2")
ylims!(0, 2)
axislegend(ax, position=:rt)

# Points
for ax in [ax3, ax4]
    poly!(ax, Point2f[(-5, 3), (0, 3), (0, 0), (-5, 2)], color=(:grey, 0.2))

    vlines!(ax, [-5], color=(:grey, 0.8), linestyle=:dash)
    hlines!(ax, [3], color=(:grey, 0.8), linestyle=:dash)

    # arc!(ax, Point2f(-3, 2), 2, -π, π)
    scatter!(ax, @lift([$k1]), @lift([$k2]), color=:red, markersize=10)
    scatter!(ax, [-5], [3], color=:orange, markersize=10, marker=:cross)
    scatter!(ax, [0], [3], color=:green, markersize=10, marker=:cross)
    scatter!(ax, [0], [0], color=:blue, markersize=10, marker=:cross)
    scatter!(ax, [-5], [2], color=:purple, markersize=10, marker=:cross)
end


fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.2
        k1[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(-5.5, 0.5))
    end
    if frame >= 0.25 && frame < 0.45
        k2[] = change_param(frame; frame_range=(0.25, 0.45), param_range=(3.5, -0.5))
    end
    if frame >= 0.5 && frame < 0.7
        k1[] = change_param(frame; frame_range=(0.5, 0.70), param_range=(0.5, -5.5))
        k2[] = change_param(frame; frame_range=(0.5, 0.70), param_range=(-0.5, 1.5))
    end
    # Return to normal
    if frame >= 0.75 && frame < 0.95
        k2[] = change_param(frame; frame_range=(0.75, 0.95), param_range=(1.5, 3.5))
    end
    ylims!(ax1)

end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_OrderedBeta.gif", frames; framerate=15)
