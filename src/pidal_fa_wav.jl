#  Implementation based on Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
#
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW, Noise
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
x0 /= maximum(x0)/10
nx=size(x0,1)
psf = gaussian2d(nx,nx,5.0);
H = x->convolve(x, psf)
Ht = x->correlate(x, psf)
PSD_H = abs2.(ft2(psf)) 
y = poisson(H(x0))

nwavs = 1
W, Wt = Wav_functions(nwavs);
PSD_W = nwavs  # diagonal = nwavs*𝟙

# PIDAL-FA with Frame = Haar
u1 = deepcopy(y)
u2 = W(y)
u3 = deepcopy(y)

d1 = 0*u1
d2 = 0*u2
d3 = 0*u3

τ = 0.1/nwavs
μ = 1e-1

niter = 1000
# Residuals
r1 = zeros(Float64, niter);
r2 = zeros(Float64, niter);
r3 = zeros(Float64, niter);
s1 = zeros(Float64, niter);
s2 = zeros(Float64, niter);
s3 = zeros(Float64, niter);
z=Float64[]
for k=1:niter
    # Note: μ does not appear in the following since it is both at numerator and denominator
    z =  real.(ift2(ft2(Ht(u1 + d1) + Wt(u2 + d2) + (u3 + d3))./(PSD_H .+ PSD_W .+ 1.0) )) ;
    # Apply proximal operators
    u1_old = copy(u1);
    u2_old = copy(u2);
    u3_old = copy(u3);
    u1 = prox_poisson(H(z) - d1,μ,y);
    u2 = prox_l1(W(z) - d2,τ/μ);
    u3 = max.(z-d3,0.0);
    # Residuals
    r1[k] = norm(H(z) - u1)
    r2[k] = norm(W(z) - u2)
    r3[k] = norm( z - u3)
    s1[k] = norm(μ*Ht(u1-u1_old))
    s2[k] = norm(μ*Wt(u2-u2_old))
    s3[k] = norm(μ*(u3-u3_old))
    #Update Lagrangian multipliers
    d1 += - H(z)  + u1;
    d2 += - W(z)  + u2;
    d3 += - z  + u3;
    println("$k r1: $(r1[k]) s1: $(s1[k]) r2: $(r2[k]) s2: $(s2[k]) r3: $(r3[k]) s3: $(s3[k]) isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");
    if k%20 == 0
        imview3(x0,y,u3)
    end
end
