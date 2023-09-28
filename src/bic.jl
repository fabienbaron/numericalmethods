#
# Bayesian Information Criterion
#
using LinearAlgebra, PyPlot
include("ultranest_fit.jl")

# Generate data
n = 50
x = sort(rand(n));
ftrue = θ->θ[1] .+ θ[2]*x;
θ = zeros(2); 
θ[1]=.2; 
θ[2]=5.0;
σ = 4.0*rand(n)
y = ftrue(θ) + σ.*randn(n);
scatter(x,y);
errorbar(x,y, σ ,linestyle="none");
plot(x, ftrue(θ),linestyle="dashed" );

function chi2(data,model,σ)
    return norm((data-model)./σ)^2
end

function logsumexp(X)
    # Compute log(sum(exp(X))) while avoiding numerical over/underflow.
    return log(sum(exp.(X.-maximum(X)))).+maximum(X)
end
  
# 1 parameter, 1D grid search, MODEL M1: constant
f1 = θ->θ[1] .+ 0*x;
gridθ= range(0, 1, length=101)
chi2_M1 = [chi2(y, f1(i) , σ) for i in gridθ]
#plot(gridθ, chi2_M1)
minindx = findmin(chi2_M1)[2]
plot(x, f1(gridθ[minindx[1]]));
# computation of evidence (marginal likelihood)
delta_θ = gridθ[2]-gridθ[1];
logZ1 = log(delta_θ) + logsumexp(-0.5*chi2_M1);
~, minx, ~, result = fit_chi2_ultranest(f1, y, σ, [0.0], [10.0], cornerplot=false);
logZ1_ultranest = result["logz"]
plot(x, f1(minx));


# 2 parameters, 2D grid search, MODEL M2: linear law
f2 = θ->θ[1] .+ θ[2]*x;
gridθ1= range(0, 10, length=101);
gridθ2= range(0, 10, length=101);
chi2_M2 = reshape([chi2(y, f2([i,j]) , σ) for j in gridθ2 for i in gridθ1],101,101)
minindx = findmin(chi2_M2)[2]; 
plot(x, f2([gridθ1[minindx[1]],gridθ2[minindx[2]]]));
# computation of evidence (marginal likelihood)
delta_θ1 = gridθ1[2]-gridθ1[1];
delta_θ2 = gridθ2[2]-gridθ2[1];
logZ2 = log(delta_θ1*delta_θ2) + logsumexp(-0.5*chi2_M2)
~, minx, ~, result = fit_chi2_ultranest(f2, y, σ, [0.0,0.0], [10.0,10.0], cornerplot=false);
logZ2_ultranest = result["logz"]
plot(x, f2(minx));

# 3D grid search, MODEL M3: parabolic law
f3 = θ->θ[1] .+ θ[2]*x .+ θ[3]*x.^2;
gridθ1= range(0,10, length = 101);
gridθ2= range(0,10, length = 101);
gridθ3= range(-5,5, length= 101);
chi2_M3 = reshape([chi2(y, f3([gridθ1[i],gridθ2[j],gridθ3[k]]) , σ) for k=1:length(gridθ3)  for j=1:length(gridθ2) for i=1:length(gridθ1) ],101,101,101)
minindx=  findmin(chi2_M3)[2]; 
plot(x, f3([gridθ1[minindx[1]],gridθ2[minindx[2]],gridθ3[minindx[3]]]));
# computation of evidence (marginal likelihood)
delta_θ1 =gridθ1[2]-gridθ1[1];
delta_θ2 =gridθ2[2]-gridθ2[1];
delta_θ3 =gridθ3[2]-gridθ3[1];
logZ3 = log(delta_θ1*delta_θ2*delta_θ3) + logsumexp(-0.5*chi2_M3)
~, minx, ~, result = fit_chi2_ultranest(f2, y, σ, [0.0,0.0,-5.0], [10.0,10.0,5.0], cornerplot=false);
logZ3_ultranest = result["logz"]
plot(x, f3(minx));


# Results
k=1; println("M1 chi2: ", minimum(chi2_M1), " chi2r: ", minimum(chi2_M1)/(n-k), " log Z1 = ", logZ1, " AIC= ", minimum(chi2_M1)+2*k, " AICc= ", minimum(chi2_M1)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M1)+k*log(n), "\n");

k=2; println("M2 chi2: ", minimum(chi2_M2), " chi2r: ", minimum(chi2_M2)/(n-k), " log Z2 = ", logZ2, " AIC= ", minimum(chi2_M2)+2*k," AICc= ", minimum(chi2_M2)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M2)+k*log(n), "\n");

k=3; println("M3 chi2: ", minimum(chi2_M3), " chi2r: ", minimum(chi2_M3)/(n-k), " log Z3 = ", logZ3, " AIC= ", minimum(chi2_M3)+2*k," AICc= ", minimum(chi2_M3)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M3)+k*log(n));
