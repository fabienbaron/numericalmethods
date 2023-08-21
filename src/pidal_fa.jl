#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, MatrixDepot, Noise
include("view.jl")
include("admm_utils.jl")

x0 = Float64.(read(FITS("saturn.fits")[1]));
x0 = Float64.(read(FITS("saturn64.fits")[1]));

nx = size(x0,1)
A = matrixdepot("blur", Float64, nx, 3, 2.0, true)
AtA = A'*A
x0 = vec(x0)
y = poisson(A*x0)

Γ = TV_mat(nx);
ΓtΓ = Γ'*Γ

# PIDAL-FA
u1 = deepcopy(y)
u2 = Γ*y
u3 = deepcopy(y)

d1 = zeros(Float64, nx*nx)
d2 = zeros(Float64, 2nx*nx)
d3 = zeros(Float64, nx*nx)

τ = 0.001
μ1 = 1000*τ#60*τ/maximum(y)
μ2 = 10*τ

niter = 2000
z=Float64[]

bigM = (μ1*AtA + μ2*ΓtΓ + I)

for k=1:niter
#z = (AtA + ΓtΓ + I)\(A'*(u1 + d1) + Γ'*(u2 + d2) + (u3 + d3))
z = bigM\(μ1*A'*(u1 + d1) + μ2*Γ'*(u2 + d2) + (u3 + d3));
# Apply proximal operators
u1 = prox_poisson(A*z - d1,μ1,y);
u2 = prox_l1(Γ*z - d2,μ2/τ); # or τ/μ
u3 = max.(z-d3,0.0);
#Update Lagrangian multipliers
d1 = d1 - A*z  + u1;
d2 = d2 - Γ*z  + u2;
d3 = d3 - z  + u3;
println("dist: $(norm(z-x0,1)/length(x0))");
end
imview3(reshape(x0,nx,nx), reshape(y,nx,nx), reshape(z,nx,nx))
