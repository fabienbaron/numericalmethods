using PyPlot
#f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0
a=1; b=100;
f_ros=x->(a .- x[1]).^2 + b*(x[2] - x[1].^2).^2;
# rr = collect(range(-5,5,length=1001));
# yy = repeat(rr,1,1001);
# xx = yy';
# fmap = f_ros(xx,yy)
# # Plots
# clf();imshow(fmap.^.2)
g_ros = x -> [-2*(a .- x[1]) - 4b*x[1].*(x[2] .- x[1] .^2), 2b*(x[2] .- x[1].^2)]

# h_ros = [ ax+y     x^2+b   ]
#         [  by+x^2     y^2+x^2   ]
#
# h_ros(x[1,:]) = [ 0.1   3.4 ]
#                 [ 3.5   -1.3]
using LinearAlgebra
# inv(h_ros(x[1,:])) =  [ 1 2]
#                       [ 3 4]

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
