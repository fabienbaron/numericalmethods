using PyPlot

f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0

a=1; b=100;
f_ros=(x,y)->(a .- x).^2 + b*(y - x.^2).^2;

rr = collect(range(-3,3,length=1001));
yy = repeat(rr,1,1001);
xx = yy';

fmap = f_ros(xx,yy)
# Plots

clf();imshow(fmap)
niter = 2000;
θ = zeros(Float64, niter, 2) # θ[i,:] -> [x[i], y[i]]
δ = zeros(Float64, niter, 2)

# Initial location = starting point
θ[1,:]=10*rand(2).-5.0
# Plot initial location
scatter((θ[1,1]+5)*100+1,(θ[1,2]+5)*100+1, color=:blue)

stepsize = 10.0
# Initialize Markov Chain
for i=2:niter
    δ[i-1,:] = stepsize/sqrt(i)*(rand(2).-0.5);
    θ_trial= min.(max.(θ[i-1,:] + δ[i-1,:],-5.0),5.0)

    f_current = f_ros(θ[i-1,1], θ[i-1,2]) ; # f at current location
    f_trial = f_ros(θ_trial[1], θ_trial[2]) # f at tentative location

    if(f_trial < f_current) # improvement !
        θ[i,:] = θ_trial # accept move
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:white)
    else
        θ[i,:] = θ[i-1,:] # reject move, stay where we are
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:red)
    end
#scatter((θ[i,1]+5)*100+1,(θ[i,2]+5)*100+1, color=:black)
end

f_ros(θ[niter,1],θ[niter,2])
