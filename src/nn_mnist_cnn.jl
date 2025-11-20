using Statistics, Flux, MLDatasets
using Flux: onecold, onehotbatch, DataLoader

# 1. Load data
train = MNIST(split = :train)
# Reshape for CNN: (28, 28, 1, num_samples)
X = Float32.(reshape(train.features, 28, 28, 1, :))
Y = onehotbatch(train.targets, 0:9)

train_loader = DataLoader((X, Y), batchsize=128, shuffle = true)

# 2. Define CNN model

# 2. Define CNN model (small)
model = Chain(
    Conv((5, 5), 1 => 16, relu),      # 28x28x1 -> 24x24x16
    MaxPool((2, 2)),                   # 24x24x16 -> 12x12x16
    Conv((5, 5), 16 => 32, relu),      # 12x12x16 -> 8x8x32
    MaxPool((2, 2)),                   # 8x8x32 -> 4x4x32
    Flux.flatten,                      # 4x4x32 -> 512
    Dense(512, 10),                    # 512 -> 10
    softmax
)
# # Historical LeNet-5 model (Yann LeCun, 1998)
# model = Chain(
#     Conv((5, 5), 1 => 6, tanh),        # 28x28x1 -> 24x24x6
#     MaxPool((2, 2)),                   # 24x24x6 -> 12x12x6
#     Conv((5, 5), 6 => 16, tanh),       # 12x12x6 -> 8x8x16
#     MaxPool((2, 2)),                   # 8x8x16 -> 4x4x16
#     Flux.flatten,                      # 4x4x16 -> 256
#     Dense(256, 120, tanh),             # 256 -> 120
#     Dense(120, 84, tanh),              # 120 -> 84
#     Dense(84, 10),                     # 84 -> 10
#     softmax
# )

# A larger CNN
# model = Chain(
#     # First convolutional block
#     Conv((3, 3), 1 => 32, relu, pad=1),
#     MaxPool((2, 2)),
    
#     # Second convolutional block
#     Conv((3, 3), 32 => 64, relu, pad=1),
#     MaxPool((2, 2)),
    
#     # Flatten and fully connected layers
#     Flux.flatten,
#     Dense(7 * 7 * 64, 128, relu),
#     Dropout(0.5),
#     Dense(128, 10),
#     softmax
# )

# 3. Define loss: note, first argument is the model
loss(m, x, y) = Flux.crossentropy(m(x), y)

# 4. Setup optimizer state
opt_state = Flux.setup(Flux.Adam(1e-3), model);

# 5. Train using Flux.train!
for epoch in 1:5
    Flux.train!(model, train_loader, opt_state) do m, x, y
        # this block is the function being differentiated
        loss(m, x, y)
    end
    println("Epoch $epoch completed")
end

using PyPlot

# Load test set
test = MNIST(split=:test)
Xtest = Float32.(reshape(test.features, 28, 28, 1, :))
Ytest = test.targets

# Predict
ŷ = onecold(model(Xtest)) .- 1
accuracy = mean(ŷ .== Ytest)
println("Test accuracy: ", round(accuracy*100, digits=2), "%")

# Display first 36 images in a 6x6 grid
figure(figsize=(8,8))
for i in 1:36
    subplot(6,6,i)
    img = reshape(Xtest[:, :, 1, i], 28, 28)
    imshow(img', cmap="gray")  # transpose for correct orientation
    title("Pred:$(ŷ[i])|True:$(Ytest[i])", fontsize=8)
    axis("off")
end
tight_layout()
show()