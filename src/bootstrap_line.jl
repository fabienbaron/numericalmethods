N=30
σ=rand(N);
X=sort(rand(N));
θ1=5
θ2=7
Y=θ1 .+ θ2*X + σ.*randn(N);
using PyPlot;
scatter(X,Y,s=3)
