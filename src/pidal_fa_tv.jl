#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, MatrixDepot, Noise
include("view.jl")
include("admm_utils.jl")

x0 = Float64.(read(FITS("saturn.fits")[1]));
x0 = Float64.(read(FITS("saturn64.fits")[1]));

nx = size(x0,1)
K = matrixdepot("blur", Float64, nx, 3, 2.0, true)
KtK = K'*K
x0 = vec(x0)
y = poisson(K*x0)

Γ = TV_mat(nx);
ΓtΓ = Γ'*Γ

# PIDAL-TV
u1 = deepcopy(y)
u2 = Γ*y
u3 = deepcopy(y)

d1 = zeros(Float64, nx*nx)
d2 = zeros(Float64, 2nx*nx)
d3 = zeros(Float64, nx*nx)

τ = 0.001 # regularization weight
#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, MatrixDepot, Noise
include("view.jl")
include("admm_utils.jl")

x0 = Float64.(read(FITS("saturn.fits")[1]));
x0 = Float64.(read(FITS("saturn64.fits")[1]));

nx = size(x0,1)
K = matrixdepot("blur", Float64, nx, 3, 2.0, true)
KtK = K'*K
x0 = vec(x0)
y = poisson(K*x0)

Γ = TV_mat(nx);
ΓtΓ = Γ'*Γ

# PIDAL-TV
u1 = deepcopy(y)
u2 = Γ*y
u3 = deepcopy(y)

d1 = zeros(Float64, nx*nx)
d2 = zeros(Float64, 2nx*nx)
d3 = zeros(Float64, nx*nx)

τ = 0.002 # regularization weight
μ = 60τ/maximum(y) #5.0 # augmented lagrangian hyperparameter

niter = 400
z=Float64[]

bigM = (KtK + ΓtΓ + I)

for k=1:niter
z = bigM\(K'*(u1 + d1) + Γ'*(u2 + d2) + (u3 + d3));
# Apply proximal operators
u1 = prox_poisson(K*z - d1,μ,y);
u2 = prox_l1(Γ*z - d2,τ/μ);
u3 = max.(z-d3,0.0);
#Update Lagrangian multipliers
d1 = d1 - K*z  + u1;
d2 = d2 - Γ*z  + u2;
d3 = d3 - z  + u3;
println("isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");
end
imview3(reshape(x0,nx,nx), reshape(y,nx,nx), reshape(z,nx,nx))
