using PyPlot

f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0

a=1; b=100;
f_ros=(x,y)->(a .- x).^2 + b*(y - x.^2).^2;

rr = collect(range(-3,3,length=1001));
yy = repeat(rr,1,1001);
xx = yy';

f = f_ack

fmap = f(xx,yy)
plot_surface(xx, yy, fmap, cmap="Spectral_r", edgecolor="none")

# Plots

clf();imshow(fmap.^.25, cmap="Spectral_r");tight_layout()
niter = 200;
θ = zeros(Float64, niter, 2) # θ[i,:] -> [x[i], y[i]]
δ = zeros(Float64, niter, 2)

# Initial location = starting point
θ[1,:]=10*rand(2).-5.0
# Plot initial location
scatter((θ[1,1]+5)*100+1,(θ[1,2]+5)*100+1, color=:blue, s=10)

# Setup step length for the problem (it will change over time + random draw)
stepsize = 10.0
accepted_pos = [(θ[1,1]+5)*100+1,(θ[1,2]+5)*100+1]
naccepted=0
# Initialize Markov Chain
for i=2:niter
    δ[i-1,:] = stepsize/sqrt(i)*(rand(2).-0.5); # compute actual move for this iteration
    #δ[i-1,:] = stepsize*(rand(2).-0.5); # compute actual move for this iteration
    
    θ_trial= min.(max.(θ[i-1,:] + δ[i-1,:],-5.0),5.0) # restrict trial move to be within range -5:5
#    θ_trial= rand(2)*10 .-5 # just try a random new point (non-Markov, generally bad)

    f_current = f(θ[i-1,1], θ[i-1,2]) ; # f at current location
    f_trial = f(θ_trial[1], θ_trial[2]) # f at trial location
    if(f_trial < f_current) # improvement !
        θ[i,:] = θ_trial # accept move
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:white, s=10)
        arrow(accepted_pos[1],accepted_pos[2],(θ_trial[1]+5)*100+1-accepted_pos[1],
        (θ_trial[2]+5)*100+1-accepted_pos[2], color=:white,shape="full", length_includes_head=true)
        accepted_pos = [(θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1]
        naccepted+=1
    else
        θ[i,:] = θ[i-1,:] # reject move, stay where we are
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:red, s=10)
    end
end
scatter((θ[end,1]+5)*100+1,(θ[end,2]+5)*100+1, color=:green)
f(θ[niter,1],θ[niter,2])
