# A Neural Network in Julia
# Converted from: https://iamtrask.github.io/2015/07/12/basic-python-network/

using Random, Statistics

# Sigmoid activation function
function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

# Derivative of sigmoid
function sigmoid_derivative(x)
    return x .* (1 .- x)
end

println("=" ^ 50)
println("Part 1: 2 Layer Neural Network")
println("=" ^ 50)

# Input dataset
X = [0 0 1;
     0 1 1;
     1 0 1;
     1 1 1]

# Output dataset
y = [0; 0; 1; 1]

# Seed random numbers for reproducibility
Random.seed!(10)

# Initialize weights randomly with mean 0
syn0 = 2 * rand(3, 1) .- 1

for n in 1:10000
    # Forward propagation
    l0 = copy(X)
    l1 = sigmoid(l0 * syn0)
    # Calculate error
    l1_error = y - l1
    if (n % 1000) == 0
        println("Error: ", mean(abs.(l1_error)))
    end    
    # Multiply error by slope of sigmoid at l1
    l1_delta = l1_error .* sigmoid_derivative(l1)
    # Update weights
    syn0 .+= l0' * l1_delta
end

println("\nOutput After Training:")
println(sigmoid(X * syn0))

println("\n" * "=" ^ 50)
println("Part 2: 3 Layer Neural Network (XOR Problem)")
println("=" ^ 50)

# Input dataset
X = [0 0 1;
     0 1 1;
     1 0 1;
     1 1 1]

# Output dataset (XOR pattern)
y = [0; 1; 1; 0]

Random.seed!(1)

# Randomly initialize weights with mean 0
syn0 = 2 * rand(3, 4) .- 1
syn1 = 2 * rand(4, 1) .- 1

for j in 1:60000
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(l0 * syn0)
    l2 = sigmoid(l1 * syn1)
    # How much did we miss the target value?
    l2_error = y - l2
    if (j % 5000) == 0
        println("Error: ", mean(abs.(l2_error)))
    end
    # In what direction is the target value?
    # Were we really sure? If so, don't change too much.
    l2_delta = l2_error .* sigmoid_derivative(l2)
    # How much did each l1 value contribute to the l2 error?
    l1_error = l2_delta * syn1'
    # In what direction is the target l1?
    # Were we really sure? If so, don't change too much.
    l1_delta = l1_error .* sigmoid_derivative(l1)
    # Update weights
    syn1 .+= l1' * l2_delta
    syn0 .+= l0' * l1_delta
end

println("\nFinal Output:")
println(sigmoid(sigmoid(X * syn0) * syn1))

println("\n" * "=" ^ 50)
println("Compact Version (11 lines)")
println("=" ^ 50)

# The compact version from the blog post
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1]
y = [0; 1; 1; 0]
syn0 = 2 * rand(3, 4) .- 1
syn1 = 2 * rand(4, 1) .- 1
for j in 1:60000
    l1 = sigmoid(X * syn0)
    l2 = sigmoid(l1 * syn1)
    l2_delta = (y - l2) .* sigmoid_derivative(l2)
    l1_delta = (l2_delta * syn1') .* sigmoid_derivative(l1)
    syn1 .+= l1' * l2_delta
    syn0 .+= X' * l1_delta
end

println("\nCompact Version Output:")
println(sigmoid(sigmoid(X * syn0) * syn1))