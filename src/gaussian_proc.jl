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

σ=100.0; l=30.0
xp=1:100
Σ=K(xp,xp, σ, l)
L=cholesky(Σ + 1e-10*I(size(Σ,1))).U' # norm(L*L'-Σ) ~ 0
plot(xp,L*randn(100))

xstar=[2.5, 23.5, 50.5, 82.5, 102];
fstar=K(xstar, x, σ, l)*((K(x,x,σ,l)+σn^2*I)\y);
covfstar=K(xstar, xstar, σ,l)-K(xstar, x, σ, l)*((K(x, x, σ, l)+σn^2*I)\K(x, xstar, σ, l))
σstar=sqrt.([covfstar[i,i] for i=1:length(xstar)])
scatter(xstar, fstar)
errorbar(xstar, fstar, σstar,linestyle="none", label="Inferred values")




using GaussianProcesses
using Random
Random.seed!(20140430)
# Training data
n=10;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05*randn(n);   #regressors

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern,logObsNoise)       # Generate the GP and fits the GP function to the data

# we can calculate the predicted mean and variance
μ, σ² = predict_y(gp,range(0,stop=2π,length=100));
using Plots  #Load Plots.jl package
plot(gp; xlabel="x", ylabel="y", title="Gaussian process", legend=false, fmt=:png)      # Plot the GP

optimize!(gp) # optimizes the Kernel parameters to fit the data better
plot(gp; legend=false, fmt=:png)   #Plot the GP after the hyperparameters have been optimised 


