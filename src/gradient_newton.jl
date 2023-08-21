using PyPlot, LinearAlgebra
a=1; b=100;
f_ros=x->(a .- x[1]).^2 + b*(x[2] - x[1].^2).^2;
g_ros = x -> [-2*(a .- x[1]) - 4b*x[1].*(x[2] .- x[1] .^2), 2b*(x[2] .- x[1].^2)]

# using Sympy
# x,y,a,b = symbols("x y a b")
# f(x,y) = (a - x)^2 + b*(y - x^2)^2
# simplify(diff(f(x,y), x))
# simplify(diff(f(x,y), y))
#
# # Hessian
# simplify(diff(f(x,y), x, x))
# simplify(diff(f(x,y), x, y))
# simplify(diff(f(x,y), y, x))
# simplify(diff(f(x,y), y, y))


function linesearch(x_current, d_current)
α_try = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10., 100., 1000.]
fval = zeros(length(α_try))
for i=1:length(α_try)
fval[i] = f_ros(x_current - α_try[i] * d_current)
end
return α_try[findmin(fval)[2]]
end

N=100
f = zeros(N)
x = zeros(N, 2)
g = zeros(N, 2)
d = zeros(N, 2)
β = zeros(N)
x[1,:] .= 10*rand(2).-5
f[1] = f_ros(x[1,:])
g[1,:] = g_ros(x[1,:])
d[1,:] = - g[1,:]

α=1e-4;
for i=2:N
    g[i,:] = g_ros(x[i-1,:]);
    β[i] = norm(g[i,:])/norm(g[i-1,:]);
    d[i,:] = -g[i,:] + β[i]*d[i-1,:];
    α = linesearch(x[i-1,:], d[i,:]);
    x[i,:] = x[i-1,:] - α * d[i,:];
    f[i] = f_ros(x[i,:]);
end
rr = collect(range(-5,5,length=1000));
map = [f_ros([i,j]) for i in rr for j in rr]
imshow(reshape(map.^.2,(1000,1000)))
scatter((x[:,1].+5)*100, (x[:,2].+5)*100)
