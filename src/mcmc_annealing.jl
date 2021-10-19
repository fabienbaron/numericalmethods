using PyPlot

f_ack=(x,y)->-20*exp.(-0.2*sqrt.(0.5*(x.^2+y.^2)))-exp.(0.5*(cos.(2*π*x)+cos.(2*π*y))).+exp(1).+20.0

a=1; b=100;
f_ros=(x,y)->(a .- x).^2 + b*(y - x.^2).^2;

rr = collect(range(-3,3,length=1001));
yy = repeat(rr,1,1001);
xx = yy';

fmap = f_ack(xx,yy)
# Plots

clf();imshow(fmap.^.5);tight_layout()
niter = 2000;
θ = zeros(Float64, niter, 2) # θ[i,:] -> [x[i], y[i]]
δ = zeros(Float64, niter, 2)
acc = zeros(Float64, niter)

# Initial location = starting point
θ[1,:]=10*rand(2).-5.0
# Plot initial location
scatter((θ[1,1]+5)*100+1,(θ[1,2]+5)*100+1, color=:blue,s=10)
accepted_pos = [(θ[1,1]+5)*100+1,(θ[1,2]+5)*100+1]
stepsize = 1.0
t0 = 1.0;
tniter = 0.01;
#temperature = (tniter - t0)/2 * (1 .+ cos.(collect(1:niter)*pi/niter)); #t *= 0.1;
temperature = t0 .- collect(1:niter)*(t0-tniter)/niter;
naccepted =0
# Initialize Markov Chain
for i=2:niter
    δ[i-1,:] = stepsize*(rand(2).-0.5);
    θ_trial= min.(max.(θ[i-1,:] + δ[i-1,:],-5.0),5.0)
    f_current = f_ack(θ[i-1,1], θ[i-1,2]) ; # f at current location
    f_trial = f_ack(θ_trial[1], θ_trial[2]) # f at tentative location
    acc[i] = min(1,exp(-(f_trial-f_current)/temperature[i]))# acceptance rate
    if(rand() < acc[i]) # improvement !
        naccepted+=1
        θ[i,:] = θ_trial # accept move
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:white, s=10)
        arrow(accepted_pos[1],accepted_pos[2],(θ_trial[1]+5)*100+1-accepted_pos[1],(θ_trial[2]+5)*100+1-accepted_pos[2], color=:white,shape="full", length_includes_head=true)
        accepted_pos = [(θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1]
    else
        θ[i,:] = θ[i-1,:] # reject move, stay where we are
        scatter((θ_trial[1]+5)*100+1,(θ_trial[2]+5)*100+1, color=:red, s=10)
    end
#scatter((θ[i,1]+5)*100+1,(θ[i,2]+5)*100+1, color=:black)
end
f_ack(θ[niter,1],θ[niter,2])
