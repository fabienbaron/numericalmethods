using LinearAlgebra, PyPlot

N = 20 # number of data points
x = sort(rand(N)) # a variable of interest (time, space, number of carrots)

M = θ->θ[1] .+ θ[2]*x # The underlying model law
θtrue = [10., 2.3] # The true model parameters describing reality

y_noiseless = M(θtrue) # Noiseless data
σ =2.0*rand(N) # heteroskedastic error bars
y = y_noiseless + σ.*randn(N)

scatter(x,y_noiseless, label = "Noiseless truth")
errorbar(x,y, σ ,linestyle="none", label="Noisy data");

θ = rand(2) # some random value we want to try
m = M(θ) # model values
R = (m - y)./σ # residuals
χ2= θ-> sum((y - M(θ)).^2)  # also = norm(R)^2
χ2r= θ-> sum((y - M(θ)).^2)/N  # also = norm(R)^2

θ = rand(2) # some random value we want to try
χ2(θ)
