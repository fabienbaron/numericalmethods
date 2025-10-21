using KernelFunctions, LinearAlgebra
using Plots, Plots.PlotMeasures
default(; lw=1.0, legendfontsize=8.0)
using Random: seed!
seed!(42); # reproducibility


function visualize(k::Kernel, X)

function mvn_sample(K, v)
    L = cholesky(K + 1e-6 * I)
    f = L.L * v
    return f
end;

    xlim=[minimum(X), maximum(X)]
    num_samples = 7
    num_inputs = length(X)
    v = randn(num_inputs, num_samples);
    K = kernelmatrix(k, X)
    f = mvn_sample(K,v)
    p_kernel_2d = heatmap(
        X,
        X,
        K;
        yflip=true,
        colorbar=false,
        ylabel=string(nameof(typeof(k))),
        ylim=xlim,
        yticks=([xlim[1], 0, xlim[end]], ["\u22125", raw"$x'$", "5"]),
        vlim=(0, 1),
        title=raw"$k(x, x')$",
        aspect_ratio=:equal,
        left_margin=5mm,
    )

    p_kernel_cut = plot(
        X,
        k.(X, 0.0);
        title=string(raw"$k(x, x_\mathrm{ref})$"),
        label=raw"$x_\mathrm{ref}=0.0$",
        legend=:topleft,
        foreground_color_legend=nothing,
    )
    plot!(X, k.(X, 1.5); label=raw"$x_\mathrm{ref}=1.5$")

    p_samples = plot(X, f; c="blue", title=raw"$f(x)$", ylim=(-3, 3), label=nothing)

    return plot(
        p_kernel_2d,
        p_kernel_cut,
        p_samples;
        layout=(1, 3),
        xlabel=raw"$x$",
        xlim=xlim,
        xticks=collect(xlim),
    )
end;

num_inputs = 101
X = range(-5,5, length=num_inputs);

# plot(visualize(SqExponentialKernel(), X); size=(800, 210), bottommargin=5mm, topmargin=5mm)

# plot(visualize(Matern52Kernel(), X); size=(800, 210), bottommargin=5mm, topmargin=5mm)

# plot(visualize(PeriodicKernel(r=[5.0]), X); size=(800, 210), bottommargin=5mm, topmargin=5mm)


kernels = [
    ConstantKernel(),
    LinearKernel(),
    Matern12Kernel(),
    Matern32Kernel(),
    WhiteKernel(),
    compose(PeriodicKernel(), ScaleTransform(0.2)),
    NeuralNetworkKernel(),
    GibbsKernel(; lengthscale=x -> sum(exp ∘ sin, x)),
]
plot(
    [visualize(k, X) for k in kernels]...;
    layout=(length(kernels), 1),
    size=(800, 220 * length(kernels) + 100),
)