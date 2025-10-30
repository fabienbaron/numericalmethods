using LinearAlgebra
using PyPlot
a=1.0; b=100.0; 
# Rosenbrock
f =x->(a - x[1])^2 + b*(x[2] - x[1]^2)^2;
# Analytic gradient (2)
g = x-> [-2(a-x[1])-4b*(x[2]-x[1]^2)*x[1], 2b*x[2]- 2b*x[1]^2];

function linesearch_alpha(x, direction) # note: f and g are predefined
    ntry = 500 # 100 alphas will be tried
    α = 10.0.^(range(-8, 8, ntry)) # exponential scale
    x_try = x .+ α'.*direction
    y_try = [f(x_try[:,i]) for i=1:ntry]
    return α[findmin(y_try)[2]]
end


N=100; # Number of iterations
x = zeros(N, 2) # History of the x locations
fhist = zeros(N) # History of the function values at x
ghist = zeros(N, 2)
y = zeros(N, 2)
d = zeros(N, 2)
β = zeros(N)

x[1,:] .= [4.0, -2.0]
fhist[1] =  f(x[1,:])
ghist[1,:] = g(x[1,:])
d[1,:] = -ghist[1,:]
y[1,:] = [0.0,0.0]

for n=2:N
    ghist[n,:] = g(x[n-1,:]);
    y[n,:] = ghist[n,:]-ghist[n-1,:]
    β[n] = norm(ghist[n,:])^2/norm(ghist[n-1,:])^2; #Fletcher-Reeves
  #  β[n] = (ghist[n,:]'*y[n,:])/norm(ghist[n-1,:])^2; #Polyak-Ribiere
    d[n,:] = -ghist[n,:] + β[n]*d[n-1,:]
    α = linesearch_alpha(x[n-1,:], d[n,:])
    x[n,:] = x[n-1,:] + α * d[n,:];
    # α = linesearch_alpha(x[n-1,:], -ghist[n,:])
    #x[n,:] = x[n-1,:] - α * ghist[n,:];
    fhist[n] = f(x[n,:]);
end

# Visualize the surface
rr = collect(range(-5,5,length=1000));
yy = repeat(rr,1,1000); xx=yy';
map = reshape([f([i,j]) for i in rr for j in rr],(1000,1000))
#plot_surface(xx, yy, map, cmap="Spectral_r", edgecolor="none")
figure(figsize=(18,6),"Gradient descent")
subplot(1,3,1)
imshow(map.^.2, cmap="Spectral_r");  
# Initial point
scatter((x[1,1] +5)*100+1,(x[1,2]+5)*100+1, color=:blue, s=10)
# History of the points
scatter((x[2:end-1,1].+5)*100, (x[2:end-1,2].+5)*100, s=1, color=:red)
# End point
scatter((x[end,1].+5)*100, (x[end,2].+5)*100, s=10, color=:green)
subplot(1,3,2)
scatter(1:N, fhist, s=2)
xlabel("Iteration number")
ylabel("Function value f(x)")
subplot(1,3,3)
scatter(1:N, x[:,1], s=2)
scatter(1:N, x[:,2], s=2)
xlabel("Iteration number")
ylabel("x components")
tight_layout();



