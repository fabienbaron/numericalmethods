using Flux

# --- Data -----------------------------------------------------
# X is 3×4 (columns are examples)
X = Float32[
    0 0 1;
    0 1 1;
    1 0 1;
    1 1 1
]'

# y is 1×4 to match model(X)
y = Float32[0, 1, 1, 0]'

# Put data into an iterator of (x,y) pairs (one batch here)
train_set = [(X, y)]

# --- Model ----------------------------------------------------
model = Chain(
    Dense(3, 4, σ),   # σ is sigmoid
    Dense(4, 1, σ)
)

# --- Loss -----------------------------------------------------
loss(m, x, y) = sum((m(x) .- y).^2) / size(y, 2)

# --- Optimizer & state (new API) -----------------------------
opt = ADAM(0.05)
opt_state = Flux.setup(opt, model)   # create optimiser state for the model

# --- Training ---
for epoch in 1:20000
    Flux.train!(model, train_set, opt_state) do m, xb, yb
        loss(m, xb, yb)
    end
    if epoch % 2000 == 0
        println("Epoch: ", epoch, " Loss: ", loss(model, X, y))
    end
end

# --- Result ---------------------------------------------------
println(model(X))
