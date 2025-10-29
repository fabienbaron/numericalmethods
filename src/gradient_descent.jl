using LinearAlgebra
using PyPlot
#f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0
a=1; b=100; 
# Rosenbrock
f =x->(a - x[1])^2 + b*(x[2] - x[1]^2)^2;
# Analytic gradient (2)
g = x-> [-2(a-x[1])-4b*(x[2]-x[1]^2)*x[1], 2b*x[2]- 2b*x[1]^2];
# Analytic Hessian (2x2)
h = x->[2-4b*x[2]+12b*x[1]^2 -4b*x[1] ; -4b*x[1] 2b]

# Steepest descent
x = [5.,-2.]; # Our initial point
# Naive implementation with α small
α=1e-3
# low values -> slow 
# high values -> Inf or NaN
niter = 1000
for n=0:niter
    println("Iteration $n: x = $(round.(x, digits=4)), f(x)= $(f(x))")
    x = x .- α'.*g(x)
end

# Manual line search
x_new = x .- α'.*g(x)
y_new = [f(x_new[:,i]) for i=1:ntry]
x_new = x[indmin(y_new)[1]]
scatter(α, y_new) # plot the function to check
set_yscale('log')

function linesearch_alpha(x) # note: f and g are predefined
    ntry = 100 # 100 alphas will be tried
    α = 10.0.^(range(-5, 5, ntry)) # exponential scale
    x_new = x .- α'.*g(x)
    y_new = [f(x_new[:,i]) for i=1:ntry]
    return α[findmin(y_new)[2]]
end

N=1000
fhist = zeros(N)
x = zeros(N+1, 2)
x[1,:] .= 10*rand(2).-5
fhist[1] = f(x[1,:])

for i=1:N
    α = linesearch(x[i,:])
    x[i+1,:] = x[i,:] - α*g(x[i,:])
    fhist[i] = f(x[i,:])
end

rr = collect(range(-5,5,length=1000));
map = [f([i,j]) for i in rr for j in rr]
imshow(reshape(map.^.2,(1000,1000)))
scatter((x[:,1].+5)*100, (x[:,2].+5)*100, s=1, color=:red)
