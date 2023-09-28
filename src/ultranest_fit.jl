using PyCall, PyPlot, UltraNest

function fit_chi2_ultranest(f, data, σ, lbounds, hbounds;
    verbose = true, cornerplot = true, min_num_live_points = 400, cluster_num_live_points = 100, use_stepsampler=false, nsteps=400,frac_remain=0.001)
;
    nparams = length(lbounds)
    param_names = repeat([""], nparams)

    function prior_transform(u::AbstractVector{<:Real}) # To be modified to accept other distributions via distributions.jl?
        Δx = hbounds - lbounds
        u .* Δx .+ lbounds
    end

    prior_transform_vectorized = let trafo = prior_transform
        (U::AbstractMatrix{<:Real}) -> reduce(vcat, (u -> trafo(u)').(eachrow(U)))
    end

    loglikelihood=param::AbstractVector{<:Real}->-norm( (data - f(param))./σ)^2;
    loglikelihood_vectorized = let loglikelihood = loglikelihood
        # UltraNest has variate in rows:
        (X::AbstractMatrix{<:Real}) -> loglikelihood.(eachrow(X))
    end

    
    smplr = ultranest.ReactiveNestedSampler(param_names, loglikelihood_vectorized, transform = prior_transform_vectorized, vectorized = true)
    if use_stepsampler==true
        smplr.stepsampler = pyimport("ultranest.stepsampler").RegionSliceSampler(nsteps=nsteps, adaptive_nsteps="move-distance")
    end

    result = smplr.run(min_num_live_points = min_num_live_points, cluster_num_live_points = cluster_num_live_points, frac_remain=frac_remain)

    minx = result["maximum_likelihood"]["point"]
    model = f(minx);
    minf = norm( (data - model)./σ)^2;

    if verbose == true
        printstyled("Log Z: $(result["logz_single"]) Chi2r: $minf \t parameters:$minx ",color=:red)
    end

    if cornerplot == true
        PyDict(pyimport("matplotlib")."rcParams")["font.size"]=[10];
        pyimport("ultranest.plot").cornerplot(result);
    end

    return (minf,minx,model, result);
end
