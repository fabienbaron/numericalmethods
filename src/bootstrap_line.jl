using PyPlot, LinearAlgebra, NLopt
N=30
σ=rand(N);
X=sort(rand(N));
θ1=5
θ2=7
Y=θ1 .+ θ2*X + σ.*randn(N);
scatter(X,Y)


chi2=(θ,dummy)->norm((Y - (θ[1] .+ θ[2]*X) )./σ)^2
#Minimization
θ_init = [1.0,1.0];
optimizer = Opt(:LN_NELDERMEAD, length(θ_init));
min_objective!(optimizer, chi2);
(minchi2,θ_opt,ret) = optimize(optimizer, θ_init);
println("got $minchi2 at $θ_opt (returned $ret)");

X_orig = deepcopy(X)
Y_orig = deepcopy(Y)
σ_orig = deepcopy(σ)

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
    (minchi2,θ_opt,ret) = optimize(optimizer, θ_init);
    println("got $minchi2 at $θ_opt (returned $ret)");
end
