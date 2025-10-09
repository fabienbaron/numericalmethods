using LinearAlgebra,  Random, Distributions,  PyPlot

N = 50
x = range(-2, 2, length=N)
#x = [-2, -1.9, -1.8,-1.6, 2.0]
θ = [1.5, -0.7]

σ = 0.5
Σ = (σ^2) * I(N) # Covariance matrix (white Gaussian noise)

# Simulate data
y =  θ[1] * x .+  θ[2] + σ .* randn(N)

# --- Compute Fisher information matrix ---
X = [x ones(N)]
Σinv = inv(Σ)
Iθ = X' * Σinv * X

# Cramér–Rao bound
CRLB = inv(Iθ)

# --- Estimate parameters by least squares ---
θ_hat = Iθ \ (X' * Σinv * y)

println("Least squares results with Cramér–Rao bounds: ")
print(θ_hat[1], " ± ", sqrt(CRLB[1,1]))
print(θ_hat[2], " ± ", sqrt(CRLB[2,2]))

# Example of a package doing this for you
using LsqFit
m = (X,θ) ->θ[2] .+ θ[1]*X
θ_init = [1.0,1.0] 
fit = curve_fit(m, x, y, Σinv , θ_init);
fit.param # results
cov = estimate_covar(fit) # this gives the same result as CRB


# --- Plot data and fitted line ---
figure(figsize=(7,5))
scatter(x, y, label="Data", color="blue")
plot(x,     θ[1] * x .+ θ[2], "g--", label="True line")
plot(x, θ_hat[1] * x .+ θ_hat[2], "r-", label="Estimated line")
xlabel("x")
ylabel("y")
legend()
title("Linear model with Gaussian noise")