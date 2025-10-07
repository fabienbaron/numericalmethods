using PyPlot, LinearAlgebra, NLopt, Statistics
N=5
σ=rand(N);
X=sort(rand(N));
θ1=5
θ2=7
Y=θ1 .+ θ2*X + σ.*randn(N);
clf(); 
scatter(X,Y)
errorbar(X,Y,σ, linestyle="none")
chi2=(θ,dummy)->norm((Y - (θ[1] .+ θ[2]*X) )./σ)^2
#Minimization
θ_init = [1.0,1.0];
optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
min_objective!(optimizer, chi2);
(minchi2,θ_opt,ret) = optimize(optimizer, θ_init);
println("got $minchi2 at $θ_opt (returned $ret)");

# LsqFit (Levenberg-Marquardt algorithm)
using LsqFit
m = (X,θ) ->θ[1] .+ θ[2]*X
fit = curve_fit(m, X, Y, 1.0./σ.^2 , θ_init)
cov = estimate_covar(fit)
se = standard_errors(fit)
margin_of_error = margin_error(fit, 0.3)
confidence_intervals = confidence_interval(fit, 0.1) # 10%

# Cramer-Rao bound (TBD)
# This will be completed when we see the CRB and Fisher information

#
# Jacknife
#
X_orig = deepcopy(X)
Y_orig = deepcopy(Y)
σ_orig = deepcopy(σ)
θ_opt = zeros(Float64, 2, N)
for i=1:N
    # Change the dataset on which we do minimization
    # iteration i -> we leave out point i
    X = X_orig[setdiff(1:N,i)]
    Y = Y_orig[setdiff(1:N,i)]
    σ = σ_orig[setdiff(1:N,i)]
    # Do minimization
    θ_init = [1.0,1.0];
    optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
    min_objective!(optimizer, chi2);
    (minchi2,θ_opt[:,i],ret) = optimize(optimizer, θ_init);
    #println("got $minchi2 at $θ_opt (returned $ret)");
end
mean(θ_opt, dims=2) # Mean values -> nominal values
std(θ_opt, dims=2) # Std -> Jacknife uncertainties

#
# Bootstrap with replacement ->>> WORKS
#
Nboot = 1000
θ_opt = zeros(Float64, 2, Nboot)
for i=1:Nboot
    # Change the dataset on which we do minimization
    indx = Int.(ceil.(N*rand(N)));
    X = X_orig[indx]
    Y = Y_orig[indx]
    σ = σ_orig[indx]

    # Do minimization
    θ_init = [1.0,1.0];
    optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
    min_objective!(optimizer, chi2);
    (minchi2,θ_opt[:,i],ret) = optimize(optimizer, θ_init);
end
mean(θ_opt[1,:])
std(θ_opt[1,:])
mean(θ_opt[2,:])
std(θ_opt[2,:])
