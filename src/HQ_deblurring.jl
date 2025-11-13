using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, PyPlot
include("view.jl")

x_truth=read(FITS("saturn64.fits")[1]);
nx=size(x_truth,1)
x_truth = vec(x_truth); # note: x_truth is a 2D array, but we will work with vectors





# # # Square patches
#   nx = 64
#   x_truth = zeros(nx,nx)
#   x_truth[10:15,34:54] .= 5.0
#   x_truth[26:35,14:24] .= 8.0
#   x_truth[37:60,10:24] .= 9.0
#   x_truth = vec(x_truth)

sigma= maximum(x_truth)/10*rand(Float64, size(x_truth))

H = matrixdepot("blur", Float64, nx, 3, 2.0, true)
y = H*x_truth + sigma.*randn(Float64,size(x_truth));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix

#
# Spatial gradient matrix for total variation
#
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];

prox_pos(x,λ) = max.(x,0.0)
prox_l1(x,λ) = sign.(x).*max.(abs.(x).-λ,0)
prox_l0(x, λ) = ifelse.(abs.(x) .> sqrt(2λ), x, zero(eltype(x)))
prox_l2sq(x, λ) = x / (1 + λ)
function prox_l2(x, λ)
    nrm = norm(x)
    if nrm > λ 
        return (1 - λ/nrm) * x
    else
        return zero(eltype(x),x)
    end
end



global mindist = 1e99;

μ = 1.0;  # use 0.03 for saturn, 1.0 for the square patches
# initialization
x = copy(y)
z = ∇*x
ρ = 0.001;

for iter=1:50
# x subproblem
    x=(H'*Σ*H+ρ*∇'*∇)\(H'*Σ*y+ρ*∇'*z); # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2
# z subproblem
    z = prox_l0(∇*x,μ/ρ); # should minimize μ*norm(z,1)+0.5*ρ*norm(z-∇*x,2)^2

    chi2 = ((y-H*x)'*Σ*(y-H*x))[1]
    reg = μ*norm(∇*x,1);
    aug = norm(z-∇*x,2)^2;
    println("obj= ", chi2+reg, " chi2r= ", chi2/length(y), " reg= ", reg, " aug= ", aug, " ρ*aug= ", ρ*aug);
    # increase ρ
    ρ = 1.5*ρ

    subplot(1,2,1)
    suptitle("Iteration $iter")
    imshow(reshape(x,(64,64)))
    subplot(1,2,2)
    imshow(reshape(∇*x,(64,128)))
end
