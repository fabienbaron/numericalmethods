using FITSIO, LinearAlgebra, Printf, SparseArrays
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);nx=size(x0,1)
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/2*rand(Float64, size(x0))
y = x0 + sigma.*randn(Float64,size(x0));
Σ=Diagonal(1.0./sigma.^2); # covariance matrix

# Chi2 and reduced chi2
#
chi2 = sum( (y-x0).^2 ./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = norm((y-x0)./sigma)^2 # using the l2 norm squared
chi2 = (y-x0)'*Σ*(y-x0) # the matricial form for any sigma
chi2r = chi2/length(y)

# Classic Tikhonov "ridge regression"
λ = 10.0.^(range(-10,15,length=201));
nλ = length(λ);
mindist = 1e99;
chi2 = zeros(nλ);
reg =  zeros(nλ);
obj =  zeros(nλ);
xopt = Float64[]
A = I;
Γ = I;

for i=1:nλ
    x=(A'*Σ*A+λ[i]*Γ'*Γ)\(A'*Σ*y);
    #x = x.*(x.>0)
    chi2[i] = ((y-A*x)'*Σ*(y-A*x))[1]
    reg[i] = norm(Γ*x,2)^2;
    obj[i] = chi2[i] + λ[i]*reg[i];
    dist = norm(x-x0,1);
    if dist<mindist
         mindist = deepcopy(dist);
         xopt = deepcopy(x);
    end
    @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n",  i, obj[i], λ[i], chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    #imview(reshape(x, (nx,nx)))
    #readline();
end

# Chi2, reg, λ*reg
fig = figure("Chi2, reg, λ*reg",figsize=(15,5))
scatter(1:nλ, chi2)
scatter(1:nλ, reg)
scatter(1:nλ, λ.*reg)

# Plot L curve - according to this heuristic method, the best lambda is to be found in the corner of the L
# This is the method most often used in practice
fig = figure("L Curve",figsize=(15,5))
xlabel("Regularization")
ylabel("Chi2")
plot(reg,chi2/length(y))
ylim([0,5])
for i=1:nλ
annotate(string("i=",i), (reg[i], chi2[i]/length(y)))
end

# Plot U curve - according to this heuristic method, the best lambda should be found in the U well
fig = figure("U Curve",figsize=(15,5))
plot(1:nλ, 1.0./chi2+ 1.0./(reg))
gca().set_yscale("log")
gca().set_xscale("linear")
xlabel("Iteration number")
ylabel("1/χ2(x) + 1/R(x)")

# # Plot CRESO curve
# fig = figure("CRESO Curve",figsize=(15,5))
# plot(1:nλ-1, diff(λ.*reg - chi2))
# gca().set_yscale("log")
# gca().set_xscale("log")
# xlabel("Iterations")
# ylabel("CRESO")


# Display object, noisy image and reconstruction
imview3(x0,y,xopt,figtitle="Tikhonov regularization");

#
# Total squared variation
#

o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
A = I;
Γ = ∇;

mindist = 1e99;
for i=1:nλ
    x=(A'*Σ*A+λ[i]*Γ'*Γ)\(A'*Σ*y);
    x = x.*(x.>0)
    chi2[i] = ((y-A*x)'*Σ*(y-A*x))[1]
    reg[i] = norm(Γ*x,2)^2;
    obj[i] = chi2[i] + λ[i]*reg[i];
    dist = norm(x-x0,1);
    if dist<mindist
         mindist = deepcopy(dist);
         xopt = deepcopy(x);
    end
    @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n",  i, obj[i], λ[i], chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    #imview(reshape(x, (nx,nx)))
    #readline();
end

clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,y,xopt,figtitle="Total squared variation regularization");
