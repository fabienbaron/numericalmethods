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
K = x->conv_otf(otf,x)
Kt = x->conv_otf(conj(otf),x)
y = poisson(K(x0))

nwav = 9
P, Pt = Wav_functions(nwav);
dPtP = nwav  # diagonal = nwav*𝟙




# PIDAL-FA with Frame = Haar
u1 = deepcopy(y)
u2 = P(y)
u3 = deepcopy(y)

d1 = 0*u1
d2 = 0*u2
d3 = 0*u3

τ = 0.01/nwav
μ = 1e-2

niter = 2000
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
z =  real.(ifft(fft(Kt(u1 + d1) + Pt(u2 + d2) + (u3 + d3))./(otf2 .+ dPtP .+ 1.0) )) ;
# Apply proximal operators
u1_old = copy(u1);
u2_old = copy(u2);
u3_old = copy(u3);


u1 = prox_poisson(K(z) - d1,μ,y);
u2 = prox_l1(P(z) - d2,τ/μ);
u3 = max.(z-d3,0.0);

# Residuals
r1[k] = norm(K(z) - u1)
r2[k] = norm(P(z) - u2)
r3[k] = norm( z - u3)

s1[k] = norm(μ*Kt(u1-u1_old))
s2[k] = norm(μ*Pt(u2-u2_old))
s3[k] = norm(μ*(u3-u3_old))


#Update Lagrangian multipliers
d1 += - K(z)  + u1;
d2 += - P(z)  + u2;
d3 += - z  + u3;
println("r1: $(r1[k]) s1: $(s1[k]) r2: $(r2[k]) s2: $(s2[k]) r3: $(r3[k]) s3: $(s3[k]) isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");

if k%20 == 0
    imview3(reshape(x0,nx,nx), reshape(y,nx,nx), reshape(u3,nx,nx))
end
end
