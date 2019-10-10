using PyPlot

f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0

a=1; b=100;
f_ros=(x,y)->(a .- x).^2 + b*(y - x.^2).^2;

rr = collect(range(-5,5,length=1000));
xx = repeat(rr,1,1000);
yy = xx';

niter = 1000;
θ = zeros(Float64, niter)
δ = zeros(Float64, niter)

# Initialize Markov Chain
θ[1]=10*rand(2).-5.0

for t=2:niter
δ[t] = 0.01*(rand(2).-0.5)
θ_trial= θ[t] + δ[t]

end
