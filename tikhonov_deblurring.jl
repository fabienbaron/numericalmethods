using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);nx=size(x0,1)
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/5*rand(Float64, size(x0))

A = matrixdepot("blur", Float64, 64, 3, 2.0, true)
y = A*x0 + sigma.*randn(Float64,size(x0);
Σ=Diagonal(1.0./sigma.^2); # covariance matrix

# Classic Tikhonov "rdige regression"
λ = 10.0.^(range(-6,5,length=101));
nλ = length(λ);
global mindist = 1e99;
global chi2 = zeros(nλ);
global reg =  zeros(nλ);
global obj =  zeros(nλ);

#A = I;
Γ = I;

for i=1:nλ
    x=(A'*Σ*A+λ[i]*Γ'*Γ)\(A'*Σ*y);
    #x = x.*(x.>0)
    chi2[i] = ((y-A*x)'*Σ*(y-A*x))[1]
    reg[i] = norm(Γ*x,2)^2;
    obj[i] = chi2[i] + λ[i]*reg[i];
    dist = norm(x-x0,1);
    if dist<mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(x);
    end
    @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n", i, obj[i], λ[i], chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    imview(reshape(x, (nx,nx)))
    #readline();
end

# Display object, noisy image and reconstruction
imview3(x0,y,xopt,figtitle="Tikhonov regularization");

#
# Total squared variation
#
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
#A = I;
Γ = ∇;

global mindist = 1e99;
for i=1:nλ
    x=(A'*Σ*A+λ[i]*Γ'*Γ)\(A'*Σ*y);
    x = x.*(x.>0)
    chi2[i] = ((y-A*x)'*Σ*(y-A*x))[1]
    reg[i] = norm(Γ*x,2)^2;
    obj[i] = chi2[i] + λ[i]*reg[i];
    dist = norm(x-x0,1);
    if dist<mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(x);
    end
    @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n", i, obj[i], λ[i],  chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    imview(reshape(x, (nx,nx)))
    #readline();
end


imview3(x0,y,xopt,figtitle="Total squared variation regularization");
