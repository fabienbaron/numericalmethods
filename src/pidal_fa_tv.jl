#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, Noise
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
x0 /= maximum(x0)/10
nx=size(x0,1)
otf = fft(gaussian2d(nx,nx,2.0));
otf ./= maximum(abs.(otf));
otf2 = abs2.(otf)  # this is 2D
H = x->conv_otf(otf,x)
Ht = x->conv_otf(conj(otf),x)
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

τ = 0.1 # Try increasing this x5 multiple times
μ = 1e-3

niter = 1000
z=Float64[]
# Residuals
r1 = zeros(Float64, niter);
r2 = zeros(Float64, niter);
r3 = zeros(Float64, niter);
s1 = zeros(Float64, niter);
s2 = zeros(Float64, niter);
s3 = zeros(Float64, niter);
for k=1:niter
    z =  real.(ifft(fft(Kt(u1 - d1) + Pt(u2 - d2) + (u3 - d3))./(otf2 + dPtP .+ 1.0) )) ;
    # Apply proximal operators
    u1_old = copy(u1);
    u2_old = copy(u2);
    u3_old = copy(u3);
    u1 = prox_poisson(K(z) + d1,μ,y);
    u2 = prox_l1(P(z) + d2,τ/μ);
    u3 = max.(z+d3,0.0);
    # Residuals
    r1[k] = norm(K(z) - u1)
    r2[k] = norm(P(z) - u2)
    r3[k] = norm( z - u3)
    s1[k] = norm(μ*Kt(u1-u1_old))
    s2[k] = norm(μ*Pt(u2-u2_old))
    s3[k] = norm(μ*(u3-u3_old))
    
    #Update Lagrangian multipliers
    d1 += K(z)  - u1;
    d2 += P(z)  - u2;
    d3 += z  - u3;

    println("$k isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");
    if mod(k,50) == 0
        imview3(x0, y, z)
    end
end

