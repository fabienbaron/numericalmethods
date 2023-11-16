using LinearAlgebra,PyPlot

function K(x,xp, σ, l)
K = zeros(Float64, length(x), length(xp));
for i=1:length(x)
    for j=1:length(xp)
        K[i,j]=σ^2*exp(-0.5*(x[i]-xp[j])^2/l^2)
    end
end
return K
end

f=x->sin(0.1*x)*(x/5)^2
close(); clf()
plot(1:100, f.(1:100), linestyle="dotted", label="Truth")
x = sort(unique(ceil.(Int, rand(30)*100)))
N = length(x)
σn=20.0;
y = f.(x) + σn*randn(N)
scatter(x,y)
errorbar(x,y,σn,linestyle="none", label="Data")
legend()

σn=2.0; σ=100.0; l=30.0
xp=1:100
Σ=K(xp,xp, σ, l)
L=cholesky(Σ).U' # norm(L*L'-Σ) ~ 0
plot(xp,L*rand(100))

xstar=[2.5, 23.5, 50.5, 82.5, 102];
fstar=K(xstar, x, σ, l)*((K(x,x,σ,l)+σn^2*I)\y);
covfstar=K(xstar, xstar, σ,l)-K(xstar, x, σ, l)*((K(x, x, σ, l)+σn^2*I)\K(x, xstar, σ, l))
σstar=sqrt.([covfstar[i,i] for i=1:length(xstar)])
scatter(xstar, fstar)
errorbar(xstar, fstar, σstar,linestyle="none", label="Inferred values")

