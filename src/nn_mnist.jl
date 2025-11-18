using Statistics, Flux, MLDatasets
using Flux: onecold, onehotbatch, DataLoader

# 1. Load data
train = MNIST(split = :train)
X = Float32.(reshape(train.features, 28^2, :))
Y = onehotbatch(train.targets, 0:9)
imshow((reshape(X[:,6],28,28))', cmap="gray")
onecold(Y[:,6])

train_loader = DataLoader((X, Y), batchsize=128, shuffle = true)

# 2. Define model
model = Chain(
  Dense(28^2, 128, relu),
  Dense(128, 64, relu),
  Dense(64, 10),
  softmax
)

# 3. Define loss: note, first argument is the model
loss(m, x, y) = Flux.crossentropy(m(x), y)

# 4. Setup optimizer state
opt_state = Flux.setup(Flux.Adam(1e-3), model);  # setup returns optimizer state for model updates :contentReference[oaicite:0]{index=0}

# 5. Train using Flux.train!
for epoch in 1:5
    Flux.train!(model, train_loader, opt_state) do m, x, y
    # this block is the function being differentiated
    loss(m, x, y)
  end
end

using PyPlot

# Load test set
test = MNIST(split=:test)
Xtest = Float32.(reshape(test.features, 28^2, :))
Ytest = test.targets 

# Predict
ŷ = onecold(model(Xtest)).-1
accuracy = mean(ŷ .== Ytest)
println("Test accuracy: ", round(accuracy*100, digits=2), "%")

# Display first 36 images in a 6x6 grid
figure(figsize=(8,8))
for i in 1:36
    subplot(6,6,i)
    img = reshape(Xtest[:, i], 28, 28)
    imshow(img', cmap="gray")  # transpose for correct orientation
    title("Pred:$(ŷ[i])|True:$(Ytest[i])", fontsize=8)
    axis("off")
end
tight_layout()
show()