using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays
include("view.jl")
x_truth=read(FITS("saturn64.fits")[1]);nx=size(x_truth,1)
x_truth = vec(x_truth); # note: x_truth is a 2D array, but we will work with vectors
sigma= maximum(x_truth)/10*rand(Float64, size(x_truth))

H = matrixdepot("blur", Float64, 64, 3, 2.0, true)
y = H*x_truth + sigma.*randn(Float64,size(x_truth));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix

#
# Total squared variation
#
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];

function prox_l1(z,α)
return sign.(z).*max.(z.-α,0)
end


λ = 10.0.^(range(-6,5,length=101));
nλ = length(λ);
global mindist = 1e99;
global chi2 = zeros(nλ);
global reg =  zeros(nλ);
global obj =  zeros(nλ);
global mindist = 1e99;

μ = 1e-7;

# initialization
x = deepcopy(y)
z = ∇*x
ρ = 0.01;

for iter=1:50
# x subproblem
global x=(H'*Σ*H+ρ*∇'*∇)\(H'*Σ*y+ρ*∇'*z);

# z subproblem
z0 = ∇*x; α = μ/ρ
global z = prox_l1(z0,α);

chi2 = ((y-H*x)'*Σ*(y-H*x))[1]/length(y)
reg = μ*norm(∇*x,1);
aug = norm(z-z0,2);
println("chi2 = ", chi2, " reg= ", reg, " aug= ", aug);

# increase ρ
global ρ = 1.2*ρ

end


    #obj[i] = chi2[i] + λ[i]*reg[i];
    # dist = norm(x-x_truth,1);
    # if dist<mindist
    #     global mindist = deepcopy(dist);
    #     global xopt = deepcopy(x);
    # end
    # @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n", i, obj[i], λ[i],  chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    # imview(reshape(x, (nx,nx)))
    # readline();
