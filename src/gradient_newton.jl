using LinearAlgebra, PyPlot
a=1; b=100; 
# Rosenbrock
f =x->(a - x[1])^2 + b*(x[2] - x[1]^2)^2;
# Analytic gradient (2)
g = x-> [-2(a-x[1])-4b*(x[2]-x[1]^2)*x[1], 2b*x[2]- 2b*x[1]^2];
# Analytic Hessian (2x2)
h = x->[2-4b*x[2]+12b*x[1]^2 -4b*x[1] ; -4b*x[1] 2b]

# Here is how we would have done it with Automatic Differentiation
using Zygote
g_ad = x->gradient(f, x)[1]
h_ad=  x->hessian(f, x)
x=rand(2)*10; 
norm(g(x) - g_ad(x))
norm(h(x) - h_ad(x))

#
# Newton's method
#
N=10; # Number of iterations
x = zeros(N, 2) # History of the x locations
fhist = zeros(N) # History of the function values at x
x[1,:] .= [4.0, -2.0]
fhist[1] =  f(x[1,:])
α = 1.0
for n=2:N
    x[n,:] = x[n-1,:] - α*h(x[n-1,:])\g(x[n-1,:])
    fhist[n] = f(x[n,:])
    println("Iteration $n: x = $(x[n-1,:]), f(x)= $(fhist[n])")
end

# Clamp x (because Newton's method can go in wild directions)
x = max.(min.(x, 5), -5)

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