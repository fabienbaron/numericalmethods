using PyPlot, LinearAlgebra, NLopt, Statistics
N=5
σ=rand(N);
X=sort(rand(N));
θ=[5.0,7.0]
Y= θ[2] .+ θ[1]*X + σ.*randn(N);
chi2=(θ,dummy)->norm((Y - (θ[2] .+ θ[1]*X) )./σ)^2
#Minimization
θ_init = [1.0,1.0];
optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
min_objective!(optimizer, chi2);
(minchi2,θ_hat,ret) = optimize(optimizer, θ_init);
println("got $minchi2 at $θ_opt (returned $ret)");

# --- Plot data and fitted line ---
figure(figsize=(7,5))
scatter(X, Y, label="Data", color="blue")
errorbar(X,Y,σ, linestyle="none")
plot(X,     θ[1] * X .+ θ[2], "g--", label="True line")
plot(X, θ_hat[1] * X .+ θ_hat[2], "r-", label="Estimated line")
xlabel("x")
ylabel("y")
legend()
title("Linear model with Gaussian noise")

# LsqFit (Levenberg-Marquardt algorithm)
using LsqFit
m = (X,θ) -> θ[1]*X .+ θ[2] 
fit = curve_fit(m, X, Y, 1.0./σ.^2 , θ_init);
θ_hat2 = fit.param
norm(θ_hat2-θ_hat)
cov = estimate_covar(fit)
println(θ_hat2[1], " ± " , sqrt(cov[1,1]), "\t vs truth = ", θ[1])
println(θ_hat2[2], " ± " , sqrt(cov[2,2]), "\t vs truth = ", θ[2])
##se = standard_errors(fit)
#margin_of_error = margin_error(fit, 0.3)
#confidence_intervals = confidence_interval(fit, 0.1) # 10%

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
# Note: sometimes the Jacknife will fail because of outlier
# in this case median(θ_opt, dims=2) will differ strongly from mean(θ_opt, dims=2)
# Or use a library. 
using RobustModels; 
mean(MEstimator{HuberLoss}(), θ_opt; dims=2); 
mean_and_std(SEstimator{TukeyLoss}(),θ_opt , dims=2)
mean_and_std(TauEstimator{YohaiZamarLoss}(),θ_opt , dims=2)
#
# Bootstrap with replacement 
#

Nboot = 10000
θ_opt = zeros(Float64, 2, Nboot)
for i=1:Nboot
    # Change the dataset on which we do minimization
    indx = Int.(ceil.(N*rand(N)));
    X = X_orig[indx]
    Y = Y_orig[indx]
    σ = σ_orig[indx]

    # Do minimization
    θ_init = [0.0,0.0];
  #  optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
  #  min_objective!(optimizer, chi2);
  #  (minchi2,θ_opt[:,i],ret) = optimize(optimizer, θ_init);

    fit = curve_fit(m, X, Y, 1.0./σ.^2 , θ_init);
    θ_opt[:,i] = fit.param

end
mean(θ_opt[1,:])
std(θ_opt[1,:])
mean(θ_opt[2,:])
std(θ_opt[2,:])
hist(θ_opt[1,:], bins=100);