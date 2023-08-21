using PyPlot
# Generate data
N = 20; # number of data points
θ=rand(N)*2*pi;
a = 3
b = 5
σ = 0.5
x=  a *cos.(θ)  + σ*randn(N) #.+0.1; # These additive factors will introduce systematic errors
y = b *sin.(θ)  + σ*randn(N) #.-0.2;
scatter(x,y); axis("equal")
chi2r = sum( (x-a*cos.(θ)).^2  + (y-b*sin.(θ)).^2)/σ^2/(2N)

#Visualize the true ellipse
θ_all=range(-pi,pi,length=1000)
figure(1)
scatter(a*cos.(θ_all), b*sin.(θ_all))
title("Ellipse + data")

# Given data (x,y), estimate a and b
# Grid sample for a and b
Ngrid = 200
 a_grid = collect(range(0, 8, length=Ngrid));
 b_grid = collect(range(0, 8, length=Ngrid));
 chi2 = zeros(length(a_grid),length(b_grid));

 for i=1:length(a_grid)
     for j=1:length(b_grid)
         chi2[i,j] = sum( (x-a_grid[i]*cos.(θ)).^2  + (y-b_grid[j]*sin.(θ)).^2)/σ^2
     end
 end

 minsol = findmin(chi2)
 minchi2 = minsol[1]
 println("Minimum chi2 = ", minchi2," at: a=", a_grid[minsol[2][1]], " b = ", b_grid[minsol[2][2]])
 figure(2)
 imshow(rotl90(chi2).^.1,interpolation="none");
 xlabel("a");ylabel("b"); title("Chi2 surface for grid search")

 figure(3)
 imshow(rotl90(chi2.>(minchi2+1.38)),interpolation="none");
 title("Chi2 valley, chi2 > min_chi2 + exp(1)/2")
 rangea = findall(vec(sum(chi2.<(minsol[1]+exp(1)/2), dims=2)).>0)
 rangeb = findall(vec(sum(chi2.<(minsol[1]+exp(1)/2), dims=1)).>0)
#  a[rangea]
#  b[rangeb]
 print("a = ", a_grid[minsol[2][1]], " + ", (maximum(a_grid[rangea])-a_grid[minsol[2][1]])/((2*sqrt(2*log(2)))*2 , " - ",  (a_grid[minsol[2][1]]-minimum(a_grid[rangea]))/2.355*2)
 print("b = ", b_grid[minsol[2][2]], " + ", (maximum(b_grid[rangeb])-b_grid[minsol[2][2]])/((2*sqrt(2*log(2)))*2 , " - ",  (b_grid[minsol[2][2]]-minimum(b_grid[rangeb]))/2.355*2)

# Estimate best (a,b) with NLopt
using NLopt

function chi2opt(params::Vector, dummy::Vector)
    return sum( (x-params[1]*cos.(θ)).^2  + (y-params[2]*sin.(θ)).^2)/σ^2
end

params_init = [0,0];
opt = Opt(:LN_NELDERMEAD, 2);
min_objective!(opt, chi2opt);
(minchi2,params_opt,ret) = optimize(opt, params_init);
println("got $minchi2 at $params_opt (returned $ret)");
println("Min chi2r found by NLOpt: ", minchi2/2N)
#
# CLASSIC BOOTSTRAP
#
# Generate 1000 data sets off the 20 data plot2d_intensity_allepochs
nboot = 10000
x_data = zeros(nboot,N);
y_data = zeros(nboot,N);
θ_data = zeros(nboot,N);

for i=1:nboot
    indx = Int.(ceil.(N*rand(N)));
    x_data[i,:]= x[indx];
    y_data[i,:]= y[indx];
    θ_data[i,:] = θ[indx];
end

# Run a nboot-bootstrap
a_boot = zeros(nboot);
b_boot = zeros(nboot);
for i=1:nboot
    # magic
    params_init = [10,10];
    opt = Opt(:LN_NELDERMEAD, 2);
    #lower_bounds!(opt, [0., 0.])
    min_objective!(opt, (params::Vector, dummy::Vector)->sum( (x_data[i,:]-params[1]*cos.(θ_data[i,:])).^2  + (y_data[i,:]-params[2]*sin.(θ_data[i,:])).^2)/σ^2);
    (minchi2,params_opt,ret) = optimize(opt, params_init);
    a_boot[i]=params_opt[1];
    b_boot[i]=params_opt[2];

    # optional debug
    # println( (norm(x_data[i,:]-params_opt[1]*cos.(θ)).^2  + norm(y_data[i,:]-params_opt[2]*sin.(θ)).^2) /σ^2 );
    # clf();
    # scatter(params_opt[1]*cos.(θ_all), params_opt[2]*sin.(θ_all));
    # scatter(x_data[i,:], y_data[i,:]);
    # scatter(a_boot[i]*cos.(θ_all), b_boot[i]*sin.(θ_all))
    # readline();

    if(mod(i,100)==0)
        println("Iteration:",i);
    end
end

# Compute error bars from bootstrap
figure(4)
ha = hist(a_boot,100) # Histogram
hb = hist(b_boot,100) # Histogram
ha[1]; #histogram data: counts per intervals
ha[2]; #histogram data: intervals
mode, indx_mode = findmax(ha[1]); # mode of the distribution
indx = findall(ha[1][:].>0.5*mode) # index range where > half max
indx_HM_left = minimum(indx);
indx_HM_right = maximum(indx);
center = (ha[2][indx_mode]+ha[2][indx_mode])/2
FWHM_low = center - (ha[2][indx_HM_left]+ha[2][indx_HM_left+1])/2
FWHM_high = (ha[2][indx_HM_right]+ha[2][indx_HM_right+1])/2 - center
σ_low =  FWHM_low / (2*sqrt(2*log(2))) # Assuming left ~ Gaussian
σ_high = FWHM_high / (2*sqrt(2*log(2))) # Assuming right ~ Gaussian

#
# BOOTSTRAP WITHOUT REPLACEMENT
#
using Random
njack = 10000
x_data = zeros(njack,19);
y_data = zeros(njack,19);
θ_data = zeros(njack,19);

for i=1:nboot
    indx = (randperm(20)([1:19]);
    x_data[i,:]= x[indx];
    y_data[i,:]= y[indx];
    θ_data[i,:] = θ[indx];
end
