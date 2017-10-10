using PyPlot

function logsumexp(X)
  # Compute log(sum(exp(X))) while avoiding numerical over/underflow.
  return log(sum(exp.(X-maximum(X))))+maximum(X)
end


# Generate data
n= 100
x = rand(n);
θ=zeros(2); θ[1]=.2; θ[2]=.5;
σ=.5
y = θ[1]+θ[2]*x + σ*randn(n) ;

scatter(x,y);
plot(x, θ[1]+θ[2]*x);


# 1D grid search, MODEL M0: constant
gridθ= linspace(0,1, 101)
chi2_M0 = zeros(length(gridθ))
for i=1:length(gridθ)
        chi2_M0[i]=norm(y-gridθ[i],2)^2/σ^2;
end
i = indmin(chi2_M0)
plot(x, gridθ[i]+0*x);
println("best chi2 for model M0: ", minimum(chi2_M0))
# computation of evidence (marginal likelihood)
delta_θ = gridθ[2]-gridθ[1];
logZ0 = log(delta_θ) + logsumexp(-0.5*chi2_M0)
println("log Z0 = ", logZ0);



# 2D grid search, MODEL M1: linear law
gridθ1= linspace(0,1, 101)
gridθ2= linspace(0,1, 101)
chi2_M1 = zeros(length(gridθ1),length(gridθ2))
for i=1:length(gridθ1)
    for j=1:length(gridθ2)
        chi2_M1[i,j]=norm(y-(gridθ1[i]+gridθ2[j]*x),2)^2/σ^2;
    end
end
i,j = ind2sub(size(chi2_M1), indmin(chi2_M1))
plot(x, gridθ1[i]+gridθ2[j]*x);
println("best chi2 for model M1: ", minimum(chi2_M1))
# computation of evidence (marginal likelihood)
delta_θ1 = gridθ1[2]-gridθ1[1];
delta_θ2 = gridθ2[2]-gridθ2[1];
logZ1 = log(delta_θ1*delta_θ2) + logsumexp(-0.5*chi2_M1)
println("log Z1 = ", logZ1);

# 3D grid search, MODEL M2: parabolic law
gridθ1= linspace(0,1, 301);
gridθ2= linspace(0,1, 301);
gridθ3= linspace(-5,5, 301);

chi2_M2 = zeros(length(gridθ1),length(gridθ2),length(gridθ3));
for i=1:length(gridθ1)
    for j=1:length(gridθ2)
        for k=1:length(gridθ3)
        chi2_M2[i,j,k]=norm(y-(gridθ1[i]+gridθ2[j]*x+gridθ3[k]*x.^2),2)^2/σ^2;
        end
    end
end
i,j,k = ind2sub(size(chi2_M2), indmin(chi2_M2))
plot(x, gridθ1[i]+gridθ2[j]*x+gridθ3[k]);

println("best chi2 for model M2: ", minimum(chi2_M2))

# computation of evidence (marginal likelihood)
delta_θ1 =gridθ1[2]-gridθ1[1];
delta_θ2 =gridθ2[2]-gridθ2[1];
delta_θ3 =gridθ3[2]-gridθ3[1];
logZ2 = log(delta_θ1*delta_θ2*delta_θ3) + logsumexp(-0.5*chi2_M2)
println("log Z2 = ", logZ2);
