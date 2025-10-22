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

# Let's inspect the kernel
σ=1.0; l=8.0; # Kernel parameters
N = 100
xp=1:N # set of coordinates
Σ=K(xp,xp, σ, l) # Evaluate the kernel for the set

L=cholesky(Σ + 1e-10*I(size(Σ,1))).U' 
norm(L*L'-Σ) #~ 0
m = zeros(N) # mean function
for _=1:100
plot(xp, m+L*randn(N) )
end


f=x->sin(0.1*x)*(x/5)^2
 clf()
plot(1:100, f.(1:100), linestyle="dotted", label="Truth")
x = sort(unique(ceil.(Int, rand(30)*100)))
N = length(x)
σn=20.0;
y = f.(x) + σn*randn(N)
scatter(x,y)
errorbar(x,y,σn,linestyle="none", label="Data")
legend()

# Here is how to calculate the mean+std values of a few points
σ=100.0; l=2.0; # Kernel parameters -- try to change these and check what happens
xstar=[2.5, 23.5, 50.5, 82.5, 102];
fstar=K(xstar, x, σ, l)*((K(x,x,σ,l)+σn^2*I)\y);
covfstar=K(xstar, xstar, σ, l)-K(xstar, x, σ, l)*((K(x, x, σ, l)+σn^2*I)\K(x, xstar, σ, l))
σstar=sqrt.([covfstar[i,i] for i=1:length(xstar)])
scatter(xstar, fstar, zorder=10)
errorbar(xstar, fstar, σstar,linestyle="none", label="Inferred values", zorder=10)

# Here is how to plot the GP
xgrid = 1:0.5:100
# Compute GP mean at these points
K_xstar_x = K(xgrid, x, σ, l)
K_x_x = K(x, x, σ, l)
fmean = K_xstar_x * ((K_x_x + σn^2*I)\y)
# Compute GP covariance at these points
K_xstar_xstar = K(xgrid, xgrid, σ, l)
fcov = K_xstar_xstar - K_xstar_x * ((K_x_x + σn^2*I)\K_xstar_x')
fstd = sqrt.([fcov[i,i] for i in 1:length(xgrid)])
# Plot
clf()
plot(1:100, f.(1:100), linestyle="dotted", label="Truth")
scatter(x, y)
errorbar(x, y, σn, linestyle="none", label="Data")
plot(xgrid, fmean, color="red", label="GP mean")
fill_between(xgrid, fmean - fstd, fmean + fstd, color="red", alpha=0.2, label="GP ±1σ")
legend()


#
# Optimization of the GP parameters
#
using Optim
# Log marginal likelihood function
function neg_log_marginal_likelihood(params)
    σ, l = params
    K_xx = K(x, x, σ, l) + σn^2*I
    return 0.5*y'*(K_xx\y) + 0.5*log(det(K_xx)) + 0.5*length(y)*log(2π)
end

# Initial guesses
lower = [10.0, 1.0]   # σ > 50 (signal amplitude ~400)
upper = [1000.0, 50.0] # l in a reasonable range
init_params = [400.0, 10.0]

res = optimize(neg_log_marginal_likelihood, lower, upper, init_params, Fminbox())
σ_opt, l_opt = Optim.minimizer(res)
println("Optimized σ = $σ_opt, l = $l_opt")

# You can check with a grid search
# σ_vals = range(50, 1000, length=100)
# l_vals = range(1, 20, length=100)
# map_lml = reshape([neg_log_marginal_likelihood([i,j]) for i in σ_vals for j in l_vals], length(l_vals), length(σ_vals) )



σ = σ_opt
l = l_opt

# Here is how to plot the GP
xgrid = 1:0.5:100
# Compute GP mean at these points
K_xstar_x = K(xgrid, x, σ, l)
K_x_x = K(x, x, σ, l)
fmean = K_xstar_x * ((K_x_x + σn^2*I)\y)
# Compute GP covariance at these points
K_xstar_xstar = K(xgrid, xgrid, σ, l)
fcov = K_xstar_xstar - K_xstar_x * ((K_x_x + σn^2*I)\K_xstar_x')
fstd = sqrt.([fcov[i,i] for i in 1:length(xgrid)])
# Plot
clf()
plot(1:100, f.(1:100), linestyle="dotted", label="Truth")
scatter(x, y)
errorbar(x, y, σn, linestyle="none", label="Data")
plot(xgrid, fmean, color="red", label="GP mean")
fill_between(xgrid, fmean - fstd, fmean + fstd, color="red", alpha=0.2, label="GP ±1σ")
legend()
