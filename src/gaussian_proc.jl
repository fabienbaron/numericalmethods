using LinearAlgebra,PyPlot

function K(x,xp, σ, l)
K = zeros(Float64, length(x), length(xp));
for i=1:length(x)
    for j=1:length(xp)
        K[i,j]=σ*exp(-0.5*(x[i]-xp[j])^2/l^2)
    end
end
return K
end

x=collect(1:100)
σn=2.0; σ=1.0; l=2.0
Σ=K(x,x, σ, l)
L=cholesky(Σ).U' # norm(L*L'-Σ) ~ 0
plot(x,L*rand(100)) # repeat this to plot more samples
y=L*rand(100);
xstar=[1.5, 2.5, 3.5, 101, 102];
plot(x,y)
fstar=K(xstar, x, σ, l)*((K(x,x,σ,l)+σn*I)\y);
covfstar=K(xstar, xstar,σ,l)*((K(x,x,σ,l)+σn*I)\K(xstar, x, σ, l)[1,:]);
scatter(xstar, fstar)
