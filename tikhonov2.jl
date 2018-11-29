using FITSIO
using LinearAlgebra
using Printf
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/2*rand(Float64, size(x0))
y=x0+sigma.*randn(Float64,size(x0));
W=Diagonal(1.0./sigma.^2);

chi2 = sum( (x0-y).^2 ./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = norm((x0-y)./sigma)^2 # using the l2 norm squared
chi2 = (x0-y)'*W*(x0-y) # the matricial form for any sigma


# Tikhonov solution
λ = 10.0.^(range(-20,20,length=41));
nλ = length(λ);
global mindist = 1e99;
global chi2 = zeros(nλ);
global reg =  zeros(nλ);

for i=1:nλ
    x=(W+λ[i]*I)\(W*y)
    chi2[i] = ((x-y)'*W*(x-y))[1]/length(x0)
    reg[i] = norm(x,2)^2;
    xpos = x.*(x.>0)
    dist = norm(xpos-x0,1);
    if dist<mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(xpos);
    end
    @printf("It: %3i λ:%8.1e chi2: %8.2f dist: %8.2f\n", i,λ[i],chi2[i], dist  );
end

clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,y,xopt,figtitle="Tikhonov regularization");

using SparseArrays
nx = 64;
o = ones(nx);
D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
D = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
DtD = D'*D;
global mindist = 1e99;

for i=1:nλ
    x=(W+λ[i]*DtD)\(W*y)
    chi2[i] = ((x-y)'*W*(x-y))[1]/length(x0)
    reg[i] = norm(x,2)^2
    xpos = x.*(x.>0)
    dist = norm(xpos-x0,1);
    if dist < mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(xpos);
    end
    @printf("It: %3i λ:%8.1e chi2: %8.2f dist: %8.2f\n", i,λ[i],chi2[i], dist  );
    imview(reshape(xpos,64,64))
    readline();
end
clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,y,xopt,figtitle="Tikhonov regularization");
