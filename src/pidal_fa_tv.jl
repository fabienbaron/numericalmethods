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

# spatial gradient
c = nx÷2+1;
Gx = zeros(nx,nx); 
Gy = zeros(nx,nx); 
Gx[c-1:c, c] = [ -1 1]
Gy[c, c-1:c] = [-1 1]
∇ = X -> cat(dims=3, convolve(X, Gx), convolve(X,Gy));
∇t = X -> real.( correlate(X[:,:,1], Gx) + correlate(X[:,:,2], Gy)) 
PSD_∇ = abs2.(ft2(Gx)) .+ abs2.(ft2(Gy))

# PIDAL-FA with Frame = TV
u1 = deepcopy(y)
u2 = ∇(y)
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
    z =  real.(ift2(ft2(Ht(u1 - d1) + ∇t(u2 - d2) + (u3 - d3))./(PSD_H + PSD_∇ .+ 1.0) )) ;
    # Apply proximal operators
    u1_old = copy(u1);
    u2_old = copy(u2);
    u3_old = copy(u3);
    u1 = prox_poisson(H(z) + d1,μ,y);
    u2 = prox_l1(∇(z) + d2,τ/μ);
    u3 = max.(z+d3,0.0);
    # Residuals
    r1[k] = norm(H(z) - u1)
    r2[k] = norm(∇(z) - u2)
    r3[k] = norm( z - u3)
    s1[k] = norm(μ*Ht(u1-u1_old))
    s2[k] = norm(μ*∇t(u2-u2_old))
    s3[k] = norm(μ*(u3-u3_old))
    
    #Update Lagrangian multipliers
    d1 += H(z)  - u1;
    d2 += ∇(z)  - u2;
    d3 += z  - u3;

    println("$k isnr: $(10*log10(norm(y-z)^2/norm(x0-z)^2)) MAE: $(norm(x0-z,1)/length(x0))");
    if mod(k,50) == 0
        imview3(x0, y, z)
    end
end

