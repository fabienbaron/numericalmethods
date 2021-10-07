using PyPlot;
using LinearAlgebra

N=30
σ=rand(N);
X=sort(rand(N));
θ1=5
θ2=7
Y=θ1 .+ θ2*X + σ.*randn(N);
scatter(X,Y)

chi2=θ->norm((Y - (θ[1] .+ θ[2]*X) )./σ)^2
