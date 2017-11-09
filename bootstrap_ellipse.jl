using PyPlot
# Generate data
θ=rand(20)*2*pi;
a = 3
b = 5
σ = 0.5
x=a*cos.(θ)+σ*randn(20); y = b*sin.(θ)+σ*randn(20);
scatter(x,y);

#Visualize the true ellipse
θ_all=linspace(-pi,pi,1000)
scatter(3*cos.(θ_all), 5*sin.(θ_all))

# Given data (x,y), estimate a and b

# Grid sample for a and b

# a = linspace(0, 8, 400);
# b = linspace(0, 8, 400);
# chi2 = zeros(length(a),length(b));
#
# for i=1:length(a)
#     for j=1:length(b)
#         chi2[i,j] = sum( (x-a[i]*cos.(θ)).^2  + (y-b[j]*sin.(θ)).^2)/σ^2
#     end
# end
#
# minpos = ind2sub(size(chi2), indmin(chi2))
# println("Minimun at: a=", a[minpos[1]], " b = ", b[minpos[2]])
# clf(); imshow(rotl90(chi2),interpolation="none");
# xlabel("a");ylabel("b");

# Compute error bars from chi2
# TODO !

# Estimate best (a,b) with NLopt
using NLopt

function chi2opt(params::Vector, dummy::Vector)
    return sum( (x-params[1]*cos.(θ)).^2  + (y-params[2]*sin.(θ).^2)/σ^2)
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
ha = plt[:hist](a_boot,50) # Histogram
hb = plt[:hist](b_boot,50) # Histogram

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
