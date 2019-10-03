using PyPlot

function logsumexp(X)
  # Compute log(sum(exp(X))) while avoiding numerical over/underflow.
  return log(sum(exp.(X.-maximum(X)))).+maximum(X)
end


# Generate data
n= 50
x = sort(rand(n));
θ= zeros(2); θ[1]=.2; θ[2]=.5;
σ=.5*ones(n)
y = θ[1] .+ θ[2]*x + σ.*randn(n) ;
scatter(x,y);
errorbar(x,y, σ ,linestyle="none");
plot(x, θ[1].+θ[2]*x,linestyle="dashed" );


# 1D grid search, MODEL M1: constant
gridθ= range(0,1, length=101)
chi2_M1 = zeros(length(gridθ))
for i=1:length(gridθ)
   chi2_M1[i]=sum( (y .- gridθ[i]).^2 ./ σ.^2);
end
i = findmin(chi2_M1)[2]
plot(x, gridθ[i] .+ 0*x);
# computation of evidence (marginal likelihood)
delta_θ = gridθ[2]-gridθ[1];
logZ1 = log(delta_θ) + logsumexp(-0.5*chi2_M1);

# 2D grid search, MODEL M2: linear law
gridθ1= range(0,1, length=101);
gridθ2= range(0,1, length=101);
chi2_M2 = zeros(length(gridθ1),length(gridθ2));
for i=1:length(gridθ1)
    for j=1:length(gridθ2)
        chi2_M2[i,j]=sum( (y-(gridθ1[i].+gridθ2[j]*x)).^2 ./σ.^2);
    end
end
minindx = findmin(chi2_M2)[2];
i = minindx[1]; j=minindx[2];
plot(x, gridθ1[i].+gridθ2[j]*x);
# computation of evidence (marginal likelihood)
delta_θ1 = gridθ1[2]-gridθ1[1];
delta_θ2 = gridθ2[2]-gridθ2[1];
logZ2 = log(delta_θ1*delta_θ2) + logsumexp(-0.5*chi2_M2)


# 3D grid search, MODEL M3: parabolic law
gridθ1= range(0,1, length = 101);
gridθ2= range(0,1, length = 101);
gridθ3= range(-5,5, length= 101);

chi2_M3 = zeros(length(gridθ1),length(gridθ2),length(gridθ3));
for i=1:length(gridθ1)
    for j=1:length(gridθ2)
        for k=1:length(gridθ3)
        chi2_M3[i,j,k]=sum((y-(gridθ1[i].+gridθ2[j]*x+gridθ3[k]*x.^2)).^2 ./ σ.^2);
        end
    end
end


minindx=  findmin(chi2_M3)[2]; i = minindx[1]; j=minindx[2]; k = minindx[3];
plot(x, gridθ1[i].+gridθ2[j]*x+gridθ3[k]*x.^2);


# computation of evidence (marginal likelihood)
delta_θ1 =gridθ1[2]-gridθ1[1];
delta_θ2 =gridθ2[2]-gridθ2[1];
delta_θ3 =gridθ3[2]-gridθ3[1];
logZ3 = log(delta_θ1*delta_θ2*delta_θ3) + logsumexp(-0.5*chi2_M3)

# Results
k=1;
println("M1 chi2: ", minimum(chi2_M1), " chi2r: ", minimum(chi2_M1)/(n-k), " log Z1 = ", logZ1, " AIC= ", minimum(chi2_M1)+2*k, " AICc= ", minimum(chi2_M1)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M1)+k*log(n), "\n");

k=2;
println("M2 chi2: ", minimum(chi2_M2), " chi2r: ", minimum(chi2_M2)/(n-k), " log Z2 = ", logZ2, " AIC= ", minimum(chi2_M2)+2*k," AICc= ", minimum(chi2_M2)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M2)+k*log(n), "\n");

k=3
println("M3 chi2: ", minimum(chi2_M3), " chi2r: ", minimum(chi2_M3)/(n-k), " log Z3 = ", logZ3, " AIC= ", minimum(chi2_M3)+2*k," AICc= ", minimum(chi2_M3)+2k+(2k^2+2k)/(n-k-1), " BIC=  ", minimum(chi2_M3)+k*log(n));
 
