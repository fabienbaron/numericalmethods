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

a = linspace(0, 8, 400);
b = linspace(0, 8, 400);
chi2 = zeros(length(a),length(b));

for i=1:length(a)
    for j=1:length(b)
        chi2[i,j] = sum( (x-a[i]*cos.(θ)).^2  + (y-b[j]*sin.(θ)).^2)/σ^2
    end
end

minpos = ind2sub(size(chi2), indmin(chi2))
println("Minimun at: a=", a[minpos[1]], " b = ", b[minpos[2]])
clf(); imshow(rotl90(chi2),interpolation="none");
xlabel("a");ylabel("b");

# Compute error bars from chi2
# TODO !



# Estimate best (a,b) with NLopt
using NLopt
function chi2opt(params::Vector, dummy::Vector)
    return sum( (x-params[1]*cos.(θ)).^2  + (y-params[2]*sin.(θ)).^2)/σ^2
end

params_init = [0,0];
opt = Opt(:LN_NELDERMEAD, 2);
min_objective!(opt, chi2opt);
(minchi2,params_opt,ret) = optimize(opt, params_init);
println("got $minchi2 at $params_opt (returned $ret)")

# Estimate best (a,b) with NLopt
using NLopt
function chi2opt(params::Vector, dummy::Vector)
    return sum( (x-params[1]*cos.(θ)).^2  + (y-params[2]*sin.(θ)).^2)/σ^2
end

params_init = [0,0]
opt = Opt(:LN_NELDERMEAD, 2)
min_objective!(opt, chi2opt)
(minchi2,params_opt,ret) = optimize(opt, params_init)
println("got $minchi2 at $params_opt (returned $ret)");












# Compute error bars from bootstrap
