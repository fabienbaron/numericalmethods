using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, PyPlot
include("view.jl"); include("admm_utils.jl");
#Lines
nx = 64
x0 = zeros(nx,nx)
x0[35,:] .= 5.0
x0[:,14] .= 8.0
x0[47,24] = 1.0
x0 = vec(x0)
sigma= maximum(x0)/100*ones(Float64, size(x0))
# The first number defines the support size of the blur
# Using a large size will slow down calculations
# The second is the FWHM
#H = matrixdepot("blur", Float64, nx, 8, 2.0, true)
H = matrixdepot("blur", Float64, nx, 3, 2.0, true)

y = H*x0 + sigma.*randn(Float64,size(x0));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix
HtΣH = H'*Σ*H
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
∇t∇ = ∇'*∇

global mindist = 1e99;
μ = 0.5; # good/ok values: .5 for l1 or l0
x = copy(y)
z = ∇*x
ρ = 0.001;

for iter=1:200
# x subproblem
    x=(HtΣH+ρ*∇t∇)\(H'*Σ*y+ρ*∇'*z); # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2
# z subproblem
    z = prox_l1(∇*x,μ/ρ); # should minimize μ*norm(z,1)+0.5*ρ*norm(z-∇*x,2)^2
    chi2 = ((y-H*x)'*Σ*(y-H*x))[1]
    reg = μ*norm(∇*x,1);
    aug = ρ*norm(z-∇*x,2)^2;
    println(@sprintf("iter %3d | full = %.2f | obj = %.2f |  χ²_r = %.2f | reg = %.2f | ||∇x-z||^2 = %.2f", iter, chi2+reg+aug, chi2+reg, chi2/length(y), reg, aug/ρ))
    # increase ρ
    ρ = 1.1*ρ
    if mod(iter, 5)==0
        figure("Reconstruction", (20, 4))
        subplot(1,4,1)
        suptitle("Iteration $iter")
        title("Data y")
        imshow(reshape(y,(64,64)))
        subplot(1,4,2)
        title("Ground truth x0")
        imshow(reshape(x0,(64,64)))
        subplot(1,4,3)
        title("Current x")
        imshow(reshape(x,(64,64)))
        subplot(1,4,4)
        title("Spatial gradient ∇x")
        imshow(reshape(∇*x,(64,128)))
    end
end
