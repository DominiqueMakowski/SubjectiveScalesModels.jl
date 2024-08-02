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



# Choco =====================================================================================

# Figure
fig = Figure()

# ax1 = Axis(
#     fig[1, 1],
#     title="Choice-Confidence Distribution",
#     xlabel="Score",
#     ylabel="Distribution",
#     yticksvisible=false,
#     xticksvisible=false,
#     yticklabelsvisible=false,
# )

# xaxis = range(0, 1, length=10_000)
# lines!(ax1, xaxis, pdf.(Choco(0.5, 0.7, 1, 0.7, 1), xaxis), color=:darkblue, linewidth=3, label="p1 = 0.5, μ0 = 0.7, ϕ0 = 1, μ1 = 0.7, ϕ1 = 1")
# lines!(ax1, xaxis, pdf.(Choco(0.5, 0.7, 1, 0.5, 1), xaxis), color=:darkgreen, linewidth=3, label="p1 = 0.5, μ0 = 0.7, ϕ0 = 1, μ1 = 0.7, ϕ1 = 1")
# lines!(ax1, xaxis, pdf.(Choco(0.5, 0.7, 1, 0.3, 1), xaxis), color=:darkred, linewidth=3, label="p1 = 0.5, μ0 = 0.7, ϕ0 = 1, μ1 = 0.7, ϕ1 = 1")

# # lines!(ax1, xaxis, pdf.(Choco(0.5, 0.7, 3, 0.7, 3), xaxis), color=:darkgreen, linewidth=3, label="p1 = 0.5, μ0 = 0.7, ϕ0 = 3, μ1 = 0.7, ϕ1 = 3")
# # lines!(ax1, xaxis, pdf.(Choco(0.5, 0.3, 3, 0.3, 3), xaxis), color=:darkred, linewidth=3, label="p1 = 0.5, μ0 = 0.3, ϕ0 = 3, μ1 = 0.3, ϕ1 = 3")

# xlims!(ax1, 0, 1)
# axislegend(ax1, position=:rt)
# ylims!(ax1; low=0)

p1 = Observable(0.5)
μ0 = Observable(0.5)
ϕ0 = Observable(1.0)
μ1 = Observable(0.5)
ϕ1 = Observable(1.0)

ax2 = Axis(
    fig[1, 1],
    title=@lift("Choco(p1 = $(round($p1, digits = 1)), μ0 = $(round($μ0, digits = 1)), ϕ0 = $(round($ϕ0, digits = 1)), μ1 = $(round($μ1, digits = 1)), ϕ1 = $(round($ϕ1, digits = 1)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)
ylims!(ax2; low=0)

xaxis = collect(range(0, 1, length=10_000))
hist!(ax2, @lift(rand(Choco($p1, $μ0, $ϕ0, $μ1, $ϕ1), 10_000)), bins=50, color=:dimgrey, normalization=:pdf)
lines!(ax2, xaxis, @lift(pdf.(Choco($p1, $μ0, $ϕ0, $μ1, $ϕ1), xaxis)), color=:darkorange, linewidth=3)

fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.2
        μ0[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(0.5, 0.4))
        μ1[] = change_param(frame; frame_range=(0.0, 0.2), param_range=(0.5, 0.4))
    end
    if frame >= 0.25 && frame < 0.35
        μ0[] = change_param(frame; frame_range=(0.25, 0.35), param_range=(0.4, 0.7))
        μ1[] = change_param(frame; frame_range=(0.25, 0.35), param_range=(0.4, 0.7))
    end
    if frame >= 0.40 && frame < 0.50
        ϕ0[] = change_param(frame; frame_range=(0.40, 0.50), param_range=(1.0, 3.0))
        ϕ1[] = change_param(frame; frame_range=(0.40, 0.50), param_range=(1.0, 3.0))
    end
    if frame >= 0.55 && frame < 0.65
        p1[] = change_param(frame; frame_range=(0.55, 0.65), param_range=(0.5, 0.9))
    end
    # Return to normal
    if frame >= 0.7 && frame < 0.9
        μ0[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.7, 0.5))
        μ1[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.7, 0.5))
        ϕ0[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(3.0, 1.0))
        ϕ1[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(3.0, 1.0))
        p1[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.9, 0.5))
    end
    ylims!(ax2; low=0)
end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_Choco1.gif", frames; framerate=15)




# Choco2 =====================================================================================

p1 = Observable(0.3)
μ0 = Observable(0.6)
ϕ0 = Observable(1)
μ1 = Observable(0.4)
ϕ1 = Observable(2.0)
p_mid = Observable(0.0)
ϕ_mid = Observable(100.0)


# Figure
fig = Figure()

ax1 = Axis(
    fig[1, 1],
    title=@lift("Choco(p1 = $(round($p1, digits = 1)), μ0 = $(round($μ0, digits = 1)), ϕ0 = $(round($ϕ0, digits = 1)), μ1 = $(round($μ1, digits = 1)), ϕ1 = $(round($ϕ1, digits = 1)), p_mid = $(round($p_mid, digits = 1)), ϕ_mid = $(round($ϕ_mid, digits = 1)))"),
    xlabel="Score",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false,
)
ylims!(ax1; low=0)

xaxis = range(0, 1, length=10_000)
hist!(ax1, @lift(rand(Choco($p1, $μ0, $ϕ0, $μ1, $ϕ1, $p_mid, $ϕ_mid, 0, 1), 10_000)), bins=50, color=:dimgrey, normalization=:pdf)
lines!(ax1, xaxis, @lift(pdf.(Choco($p1, $μ0, $ϕ0, $μ1, $ϕ1, $p_mid, $ϕ_mid, 0, 1), xaxis)), color=:crimson, linewidth=3)

fig


# Animation =====================================================================================
function make_animation(frame)
    if frame < 0.3
        p_mid[] = change_param(frame; frame_range=(0.0, 0.3), param_range=(0.0, 0.7))
    end
    if frame >= 0.4 && frame < 0.6
        ϕ_mid[] = change_param(frame; frame_range=(0.4, 0.6), param_range=(100, 2))
    end
    # Return to normal
    if frame >= 0.7 && frame < 0.9
        p_mid[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(0.7, 0.0))
        ϕ_mid[] = change_param(frame; frame_range=(0.7, 0.9), param_range=(2, 100))
    end
    ylims!(ax1; low=0)
end

# animation settings
frames = range(0, 1, length=120)
record(make_animation, fig, "animation_Choco2.gif", frames; framerate=15)

