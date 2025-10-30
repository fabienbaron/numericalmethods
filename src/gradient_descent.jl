using LinearAlgebra
using PyPlot
#f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0
a=1.0; b=100.0; 
# Rosenbrock
f =x->(a - x[1])^2 + b*(x[2] - x[1]^2)^2;
# Analytic gradient (2)
g = x-> [-2(a-x[1])-4b*(x[2]-x[1]^2)*x[1], 2b*x[2]- 2b*x[1]^2];
# Analytic Hessian (2x2)
h = x->[2-4b*x[2]+12b*x[1]^2 -4b*x[1] ; -4b*x[1] 2b]

# Visualize the surface
rr = collect(range(-5,5,length=1000));
yy = repeat(rr,1,1000); xx=yy';
map = reshape([f([i,j]) for i in rr for j in rr],(1000,1000))
#plot_surface(xx, yy, map, cmap="Spectral_r", edgecolor="none")
figure(figsize=(12,6),"Gradient descent")
subplot(1,2,1)
imshow(map.^.2, cmap="Spectral_r");  

N=100; # Number of iterations
x = zeros(N+1, 2) # History of the x locations
fhist = zeros(N+1) # History of the function values at x
x[1,:] .= [4.0, -2.0]
fhist[1] =  f(x[1,:])
# Steepest descent
# Naive implementation with α small
α=1e-7
# low values -> slow 
# high values -> Inf or NaN
for n=1:N
    println("Iteration $n: x = $(round.(x[n,:], digits=4)), f(x)= $(f(x[n,:]))")
    x[n+1,:] = x[n,:] .- α'.*g(x[n,:])
    fhist[n+1] = f(x[n+1,:]) 
end
subplot(1,2,1)
# Initial point
scatter((x[1,1] +5)*100+1,(x[1,2]+5)*100+1, color=:blue, s=10)
# History of the points
scatter((x[2:end-1,1].+5)*100, (x[2:end-1,2].+5)*100, s=1, color=:red)
# End point
scatter((x[end,1].+5)*100, (x[end,2].+5)*100, s=10, color=:green)
subplot(1,2,2)
scatter(1:N+1, fhist)
xlabel("Iteration number")
ylabel("Function value f(x)")
tight_layout();

# Manual line search
n=23
ntry = 100 # 100 alphas will be tried
α = 10.0.^(range(-5, 5, ntry)) # exponential scale
x_try = x[n,:] .- α'.*g(x[n,:])
y_try = [f(x_try[:,i]) for i=1:ntry]

figure(figsize=(10,6),"Line search for iteration $n")
scatter(α, y_try, s=2) # plot the function to check
xlabel("Line search parameter α")
ylabel("Function value f(x-α*g(x))")
gca().set_yscale("log")
gca().set_xscale("log")

best_indx = findmin(y_try)[2]
α[best_indx]
x_search_min = x_try[:,α_best]

# Compare these values
f(x[n,:]) # Original value
f(x_search_min) # best move we could do
f(x[n+1,:]) # What we did

function linesearch_alpha(x) # note: f and g are predefined
    ntry = 100 # 100 alphas will be tried
    α = 10.0.^(range(-5, 5, ntry)) # exponential scale
    x_new = x .- α'.*g(x)
    y_new = [f(x_new[:,i]) for i=1:ntry]
    return α[findmin(y_new)[2]]
end

# Now let's redo our gradient descent with line search
for n=1:N
    println("Iteration $n: x = $(round.(x[n,:], digits=4)), f(x)= $(f(x[n,:]))")
    α = linesearch_alpha(x[n,:])
    x[n+1,:] = x[n,:] - α*g(x[n,:])
    fhist[n+1] = f(x[n,:])
end

# Visualize the surface
rr = collect(range(-5,5,length=1000));
yy = repeat(rr,1,1000); xx=yy';
map = reshape([f([i,j]) for i in rr for j in rr],(1000,1000))
#plot_surface(xx, yy, map, cmap="Spectral_r", edgecolor="none")
figure(figsize=(12,6),"Gradient descent")
subplot(1,2,1)
imshow(map.^.2, cmap="Spectral_r");  
# Initial point
scatter((x[1,1] +5)*100+1,(x[1,2]+5)*100+1, color=:blue, s=10)
# History of the points
scatter((x[2:end-1,1].+5)*100, (x[2:end-1,2].+5)*100, s=1, color=:red)
# End point
scatter((x[end,1].+5)*100, (x[end,2].+5)*100, s=10, color=:green)
subplot(1,2,2)
scatter(2:N, fhist[2:end])
xlabel("Iteration number")
ylabel("Function value f(x)")
tight_layout();
