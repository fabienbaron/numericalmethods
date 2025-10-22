using Random, Stheno, AbstractGPs, KernelFunctions, Distributions, Optim
using PyPlot

# --- Data ---
Random.seed!(42)
x = [0.0, 0.5, 6.0, 8.0]
true_a, true_b = 1.5, 0.5
σ = [0.6, 2.3, 5.0, 3.0]                   # large, varying noise


#x = [0.0, 3.5, 8.0]
#σ = [5.6, 1.3, 7.0]                   # large, varying noise

y = true_a .* x .+ true_b .+ σ .* randn(length(x))

# --- GP kernel: small SE kernel to allow correlated residuals ---
k = 1.0 * with_lengthscale(SEKernel(), 2.0)
f = GP(k)

# --- Log marginal likelihood for a,b ---
function log_marginal_likelihood_ab(ab)
    a, b = ab
    μ = a .* x .+ b
    resid = y .- μ
    fx = f(x, σ.^2)
    return logpdf(fx, resid)
end

obj(ab) = -log_marginal_likelihood_ab(ab)
res = optimize(obj, [0.0, 0.0], iterations=300)
a_hat, b_hat = Optim.minimizer(res)
println("Fitted slope â = $(round(a_hat, digits=3)), intercept b̂ = $(round(b_hat, digits=3))")

# --- Posterior predictive (latent) ---
μ_train = a_hat .* x .+ b_hat
resid_train = y .- μ_train
post = posterior(f(x, σ.^2), resid_train)

xnew = collect(range(minimum(x), maximum(x); length=200))
μ_resid, var_resid = mean_and_var(post(xnew))

μ_pred = a_hat .* xnew .+ b_hat .+ μ_resid
std_pred = sqrt.(var_resid)

figure(figsize=(7,5))
errorbar(x, y, yerr=σ, fmt="o", color="black", label="data (σ)", capsize=4)
plot(xnew, μ_pred, "-", color="blue", label="mean prediction")
fill_between(xnew, μ_pred .- 2std_pred, μ_pred .+ 2std_pred, color="blue", alpha=0.2, label="±2σ GP")
plot(xnew, true_a .* xnew .+ true_b, "--", color="red", label="true line")  # <--- added line
xlabel("x"); ylabel("y")
legend()
tight_layout()
show()








using Random, Stheno, AbstractGPs, KernelFunctions, Distributions, Optim
using PyPlot

# --- Data ---
Random.seed!(42)
x = [0.0, 0.5, 6.0, 8.0]
true_a, true_b = 1.5, 0.5
σ = [0.6, 0.3, 2.0, 1.0]
y = true_a .* x .+ true_b .+ σ .* randn(length(x))

# --- GP model with LinearKernel(scale, c=intercept) ---
function make_gp(log_scale, c)
    scale = exp(log_scale)                 # ensure scale > 0
    k = scale * LinearKernel(c=c)
    return GP(k)
end

# --- Log marginal likelihood ---
function log_marginal_likelihood(params)
    log_scale, c = params
    f = make_gp(log_scale, c)
    fx = f(x, σ.^2)
    return logpdf(fx, y)
end

# --- Optimize log_scale and intercept c ---
obj(params) = -log_marginal_likelihood(params)
res = optimize(obj, [0.0, 0.0], iterations=300)  # log_scale=0 → scale=1
log_scale_hat, c_hat = Optim.minimizer(res)
a_hat = exp(log_scale_hat)
println("Fitted slope â = $(a_hat), intercept ĉ = $(c_hat))")

# --- Posterior predictive ---
f_post = make_gp(log_scale_hat, c_hat)
post = posterior(f_post(x, σ.^2), y)

xnew = collect(range(minimum(x), maximum(x); length=200))
μ_pred, var_pred = mean_and_var(post(xnew))
std_pred = sqrt.(var_pred)

# --- Plot ---
figure(figsize=(7,5))
errorbar(x, y, yerr=σ, fmt="o", color="black", label="data (σ)", capsize=4)
plot(xnew, μ_pred, "-", color="blue", label="GP prediction (LinearKernel)")
fill_between(xnew, μ_pred .- 3std_pred, μ_pred .+ 3std_pred, color="blue", alpha=0.2, label="±3σ GP")
plot(xnew, true_a .* xnew .+ true_b, "--", color="red", label="true line")
xlabel("x"); ylabel("y")
legend()
tight_layout()
show()













