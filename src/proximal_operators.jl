using PyPlot

# --- Define proximal operators ---

prox_l1(x, λ)      = sign.(x) .* max.(abs.(x) .- λ, 0.0)
prox_l1_plus(x, λ) = max.(x .- λ, 0.0)
prox_l0(x, λ)      = ifelse.(abs.(x) .> sqrt(2*λ), x, 0.0)
prox_l0_plus(x, λ) = max.(ifelse.(abs.(x) .> sqrt(2*λ), x, 0.0), 0.0)

# --- Set up test range ---
x = LinRange(-3, 3, 400)
λ = 1.0

# --- Compute each proximal ---
y_l1      = prox_l1(x, λ)
y_l1_plus = prox_l1_plus(x, λ)
y_l0      = prox_l0(x, λ)
y_l0_plus = prox_l0_plus(x, λ)

# --- Plot ---
figure(figsize=(7,5))
plot(x, x, "k--", alpha=0.5, label="Identity y = x")
plot(x, y_l1, "b", lw=2, label="prox_l1 (soft)")
plot(x, y_l1_plus, "g", lw=2, label="prox_l1_plus (soft + ≥0)")
plot(x, y_l0, "r", lw=2, label="prox_l0 (hard)")
plot(x, y_l0_plus, "m", lw=2, label="prox_l0_plus (hard + ≥0)")

axhline(0, color="gray", linestyle=":")
axvline(0, color="gray", linestyle=":")
axvline(λ, color="r", linestyle="--", alpha=0.3, label="λ threshold")
axvline(-λ, color="r", linestyle="--", alpha=0.3)

xlabel("x")
ylabel("prox(x, λ)")
title("Comparison of Proximal Operators (λ = 1)")
legend()
grid(true)
tight_layout()
show()
