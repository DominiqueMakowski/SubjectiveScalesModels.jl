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
k0 = Observable(0.05)
k1 = Observable(0.95)

ax1 = Axis(
    fig[1:2, 5:6],
    title=@lift("OrderedBeta(μ = $(round($μ, digits = 1)), ϕ =  $(round($ϕ, digits = 1)), k0 = $(round($k0, digits = 2)), k1 = $(round($k1, digits = 2)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)

# hist!(ax1, @lift(rand(OrderedBeta($μ, $ϕ, $k0, $k0), 40_000)), bins=1000, normalization=:pdf, color=:grey)
xaxis = range(0, 1, length=1000)
band!(ax1, xaxis, [0], @lift(pdf.(OrderedBeta($μ, $ϕ, $k0, $k1), xaxis)), color=("#F44336", 0.1))
lines!(ax1, @lift([$k0, $k0]), @lift([0, pdf.(OrderedBeta($μ, $ϕ, $k0, $k1), $k0)]), color=("#1E88E5", 1), linewidth=2, linestyle=:dot, label="k0")
lines!(ax1, @lift([$k1, $k1]), @lift([0, pdf.(OrderedBeta($μ, $ϕ, $k0, $k1), $k1)]), color=("#4CAF50", 1), linewidth=2, linestyle=:dot, label="k1")
lines!(ax1, xaxis, @lift(pdf.(OrderedBeta($μ, $ϕ, $k0, $k1), xaxis)), color="#E53935", linewidth=5)
axislegend(position=:rt)


# Contour plot -----------------------------------------------------------------------------------


function get_pdf(k0, k1, x)
    if k1 > k0
        return pdf(OrderedBeta(0.5, 3, k0, k1), x)
    else
        return -1
    end
end

ax3 = Axis(
    fig[1, 1:2],
    title="Proportion of zeros",
    xlabel="k0",
    ylabel="k1",
    yticksvisible=false,
    xticksvisible=false,
)
contourf!(ax3,
    range(0, 1, 200),
    range(0, 1, 200),
    [get_pdf(k0, k1, 0) for k0 in range(0, 1, 200), k1 in range(0, 1, 200)],
    colormap=:amp,
    levels=range(0, 1, length=40))

xlims!(ax3, 0, 1)
ylims!(ax3, 0, 1)

ax4 = Axis(
    fig[2, 1:2],
    title="Proportion of ones",
    xlabel="k0",
    ylabel="k1",
    yticksvisible=false,
    xticksvisible=false
)



contourf!(ax4,
    range(0, 1, 200),
    range(0, 1, 200),
    [get_pdf(k0, k1, 1) for k0 in range(0, 1, 200), k1 in range(0, 1, 200)],
    colormap=:amp,
    levels=range(0, 1, length=40))

xlims!(ax4, 0, 1)
ylims!(ax4, 0, 1)

# [(k0, k1) for k1 in range(1, 3, 3), k0 in range(4, 6, 3)]

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


_make_axis(1, 3)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0.2, 0.8), xaxis), color=:orange, label="k0 = 0.2, k1 = 0.8")
ylims!(0, 2)
axislegend(position=:rt)
_make_axis(1, 4)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0.5, 0.8), xaxis), color=:green, label="k0 = 0.5, k1 = 0.8")
ylims!(0, 2)
axislegend(position=:rt)
_make_axis(2, 4)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0.5, 0.5), xaxis), color=:blue, label="k0 = 0.5, k1 = 0.5")
ylims!(0, 2)
axislegend(position=:rt)
_make_axis(2, 3)
lines!(xaxis, pdf.(OrderedBeta(0.5, 3, 0.2, 0.5), xaxis), color=:purple, label="k0 = 0.2, k1 = 0.5")
ylims!(0, 2)
axislegend(position=:rt)

# Points
for ax in [ax3, ax4]
    poly!(ax, Point2f[(0.2, 0.8), (0.5, 0.8), (0.5, 0.5), (0.2, 0.5)], color=(:grey, 0.2))

    vlines!(ax, [0.5], color=(:grey, 0.8), linestyle=:dash)
    hlines!(ax, [0.5], color=(:grey, 0.8), linestyle=:dash)

    # arc!(ax, Point2f(-3, 2), 2, -π, π)
    scatter!(ax, @lift([$k0]), @lift([$k1]), color=:red, markersize=10)
    scatter!(ax, [0.2], [0.8], color=:orange, markersize=10, marker=:cross)
    scatter!(ax, [0.5], [0.8], color=:green, markersize=10, marker=:cross)
    scatter!(ax, [0.5], [0.5], color=:blue, markersize=10, marker=:cross)
    scatter!(ax, [0.2], [0.5], color=:purple, markersize=10, marker=:cross)
end


fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.2
        k0[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(0.05, 0.5))
    end
    if frame >= 0.25 && frame < 0.45
        k1[] = change_param(frame; frame_range=(0.25, 0.45), param_range=(0.95, 0.5))
    end
    if frame >= 0.5 && frame < 0.7
        k0[] = change_param(frame; frame_range=(0.5, 0.70), param_range=(0.5, 0.05))
    end
    # Return to normal
    if frame >= 0.75 && frame < 0.95
        k1[] = change_param(frame; frame_range=(0.75, 0.95), param_range=(0.5, 0.95))
    end
    ylims!(ax1)

end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_OrderedBeta.gif", frames; framerate=15)



# Plot Example ==================================================================================
using CSV
using DataFrames
using Distributions
using GLMakie
using Downloads
using Random
using SubjectiveScalesModels

cd(@__DIR__)

Random.seed!(121)

vals = rand(OrderedBeta(0.65, 2.2, 0.015, 0.987), 3000)

n_bins = 100
bin_zero = [-1 / n_bins, eps()] # eps() is the smallest positive number to encompass zero
bin_one = [1, 1 + 1 / n_bins]
bins = vcat(bin_zero, range(eps(), 1, n_bins - 1), bin_one)


fig = Figure()
ax1 = Axis(fig[1, 1],
    title="Typical distribution from slider scales",
    xlabel="Score",
    ylabel="Frequency",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)
hist!(ax1, vals[(vals.>0).&(vals.<1)], bins=bins, color=:forestgreen)
hist!(ax1, vals[(vals.==0).|(vals.==1)], bins=bins, color=:red)
ylims!(ax1, 0, 72)

groups = vcat(rand(Normal(0, 0.1), 1500), rand(Normal(1, 0.1), 1500))

ax2 = Axis(fig[2, 1],
    title="By group",
    ylabel="Score",
    yticksvisible=false,
    xticksvisible=false,
    xlabelvisible=false,
    yticklabelsvisible=false, xticks=([0, 1], ["Control", "Treatment"]),
)

scatter!(ax2, groups[(vals.>0).&(vals.<1)], vals[(vals.>0).&(vals.<1)], color=(:black, 0.2))
scatter!(ax2, groups[(vals.==0).|(vals.==1)], vals[(vals.==0).|(vals.==1)], color=(:red, 0.5))


fig
save("./illustration_orderedbeta.png", fig)