using PyPlot
# Generate data
θ=rand(20)*2*pi;
a = 3
b = 5
σ = 0.5
x=a*cos.(θ)+σ*randn(20); y = b*sin.(θ)+σ*randn(20);
scatter(x,y);

#Visualize the true ellipse
θ_all=range(-pi,pi,length=1000)
scatter(3*cos.(θ_all), 5*sin.(θ_all))

# Given data (x,y), estimate a and b

# Grid sample for a and b

 a = collect(range(0, 8, length=400));
 b = collect(range(0, 8, length=400));
 chi2 = zeros(length(a),length(b));

 for i=1:length(a)
     for j=1:length(b)
         chi2[i,j] = sum( (x-a[i]*cos.(θ)).^2  + (y-b[j]*sin.(θ)).^2)/σ^2
     end
 end
 minsol = findmin(chi2)
 minchi2 = minsol[1]
 println("Minimun chi2 = ", minchi2," at: a=", a[minsol[2][1]], " b = ", b[minsol[2][2]])
 clf(); imshow(rotl90(chi2),interpolation="none");
 xlabel("a");ylabel("b");
 imshow(rotl90(chi2.>(minchi2+1.0)),interpolation="none");
 rangea = findall(vec(sum(chi2.<(minsol[1]+1.0), dims=2)).>0)
 rangeb = findall(vec(sum(chi2.<(minsol[1]+1.0), dims=1)).>0)
 a[rangea]
 b[rangeb]
 print("a = ", a[minsol[2][1]], " + ", maximum(a[rangea])-a[minsol[2][1]] , "- ",  a[minsol[2][1]]-minimum(a[rangea]))
 print("b = ", b[minsol[2][2]], " + ", maximum(b[rangeb])-b[minsol[2][2]] , "- ",  b[minsol[2][2]]-minimum(b[rangeb]))


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

#
# CLASSIC BOOTSTRAP
#
# Generate 1000 data sets off the 20 data plot2d_intensity_allepochs
nboot = 10000
x_data = zeros(nboot,20);
y_data = zeros(nboot,20);
θ_data = zeros(nboot,20);
for i=1:nboot
    indx = Int.(ceil.(20*rand(20)));
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
clf();
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
# CLASSIC JACKNIFE
#
nboot = 10000
x_data = zeros(nboot,19);
y_data = zeros(nboot,19);
θ_data = zeros(nboot,19);
for i=1:nboot
    indx = (randperm(20)([1:19]);
    x_data[i,:]= x[indx];
    y_data[i,:]= y[indx];
    θ_data[i,:] = θ[indx];
end
