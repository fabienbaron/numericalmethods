#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, Noise
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
nx=size(x0,1)
otf = fft(gaussian2d(nx,nx,2.0));
otf ./= maximum(abs.(otf));
otf2 = abs2.(otf)  # this is 2D
K = x->conv_otf(otf,x)
Kt = x->conv_otf(conj(otf),x)
y = poisson(K(x0))
P, Pt, Dx2, Dy2 = TV_functions(nx);
dPtP = Dx2 + Dy2

# PIDAL-FA with Frame = TV
u1 = deepcopy(y)
u2 = P(y)
u3 = deepcopy(y)

d1 = zeros(Float64, nx, nx)
d2 = zeros(Float64, nx, nx, 2)
d3 = zeros(Float64, nx, nx)

τ = 0.001
μ = 1e-3

niter = 2000
z=Float64[]

for k=1:niter
z =  real.(ifft(fft(Kt(u1 - d1) + Pt(u2 - d2) + (u3 - d3))./(otf2 + dPtP .+ 1.0) )) ;
# Apply proximal operators
u1 = prox_poisson(K(z) + d1,μ,y);
u2 = prox_l1(P(z) + d2,τ/μ);
u3 = max.(z+d3,0.0);
#Update Lagrangian multipliers
# d1 = d1 - K(z)  + u1;
# d2 = d2 - P(z)  + u2;
# d3 = d3 - z  + u3;
d1 += K(z)  - u1;
d2 += P(z)  - u2;
d3 += z  - u3;
# r1 =
# r2 =
# r3 =
println("isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");
end
imview3(reshape(x0,nx,nx), reshape(y,nx,nx), reshape(z,nx,nx))
