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
N=400
f = zeros(N)
x = zeros(N, 2)
x[1,:] .= 10*rand(2).-5
f[1] = f_ros(x[1,:])

α=1e-4;
for i=2:N
    α = linesearch(x[i-1,:], g_ros(x[i-1,:]))
    x[i,:] = x[i-1,:] - α* g_ros(x[i-1,:])
    f[i] = f_ros(x[i,:])
end

function linesearch(x_current, g_current)
α_try = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10., 100., 1000.]
fval = zeros(length(α_try))
for i=1:length(α_try)
fval[i] = f_ros(x_current - α_try[i] * g_current)
end
return α_try[findmin(fval)[2]]
end
