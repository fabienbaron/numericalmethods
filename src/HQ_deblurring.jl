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
return sign.(z).*max.(abs.(z).-α,0)
end

global mindist = 1e99;

μ = 1e-3   ;

# initialization
x = deepcopy(y)
z = ∇*x
ρ = 1;

for iter=1:20
# x subproblem
global x=(H'*Σ*H+ρ*∇'*∇)\(H'*Σ*y+ρ*∇'*z); # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2
# z subproblem
global z = prox_l1(∇*x,μ/ρ); # should minimize μ*norm(z,1)+0.5*ρ*norm(z-∇*x,2)^2

chi2 = ((y-H*x)'*Σ*(y-H*x))[1]/length(y)
reg = μ*norm(∇*x,1);
aug = norm(z-∇*x,2)^2;
println("chi2 = ", chi2, " reg= ", reg, " aug= ", aug, " ρ*aug= ", ρ*aug);

# increase ρ
global ρ = 1.5*ρ

end

imview(reshape(x,(64,64)))

    #obj[i] = chi2[i] + λ[i]*reg[i];
    # dist = norm(x-x_truth,1);
    # if dist<mindist
    #     global mindist = deepcopy(dist);
    #     global xopt = deepcopy(x);
    # end
    # @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n", i, obj[i], λ[i],  chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
    # imview(reshape(x, (nx,nx)))
    # readline();
