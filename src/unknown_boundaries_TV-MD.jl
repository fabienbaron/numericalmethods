#  Implementation based on ADMM-MD in "Deconvolving Images with Unknown Boundaries
# Using the Alternating Direction Method of Multipliers" by Almeida & Figueiredo
# 2D images
# Difference of notation with the paper: here H=A
# Note this is using the isotropic TV
# TODO: handle noise σ≠1
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
#x0 = Float64.(load("cameraman256.tif"));
nx0 = size(x0,1)
psf = gaussian2d(nx0,nx0,5.0);
H = x->convolve(x, psf)
Ht = x->correlate(x, psf)
PSD_H = abs2.(ft2(psf)) 

# spatial gradient
c = nx0÷2+1;
Gx = zeros(nx0,nx0); 
Gy = zeros(nx0,nx0); 
Gx[c-1:c, c] = [ -1 1]
Gy[c, c-1:c] = [-1 1]
∇ = X -> cat(dims=3, convolve(X, Gx), convolve(X,Gy));
∇t = X -> real.( correlate(X[:,:,1], Gx) + correlate(X[:,:,2], Gy)) 
PSD_∇ = abs2.(ft2(Gx)) .+ abs2.(ft2(Gy))

nu = 9 # band of nu pixels=zero around
nx = nx0-2*nu # size of data frame and object
M_mat = sparse_mask(nx0, nu+1:nx0-nu, nu+1:nx0-nu)
M    = x->reshape(M_mat*vec(x), nx,nx)
Mt   = x->reshape(M_mat'*vec(x),nx0,nx0)
DMtM = reshape(Array(diag(M_mat'*M_mat)), nx0, nx0)

y = M(H(x0))
σ = 1; #maximum(y)/40 # noise level
y += σ*randn(nx,nx) # add Gaussian white noise
#imview(reshape(y, nx, nx))

# ADMM-MD
u1 = zeros(Float64, nx0,nx0);
u1[nu+1:nx0-nu, nu+1:nx0-nu] = y
d1 = zeros(Float64, nx0, nx0)

u2 = ∇(u1)
d2 = zeros(Float64, nx0, nx0, 2)

λ = 0.05
µ1 = 1e-2
µ2 = 1e-2

niter = 1000
z=Float64[]

r1 = zeros(Float64, niter);
r2 = zeros(Float64, niter);

for k=1:niter
    z = real.(ift2(ft2(μ1*Ht(u1+d1) + μ2*∇t(u2+d2))./(μ1*PSD_H + μ2*PSD_∇)))
    # Apply proximal operators
    #u1 = (M'*M + μ1*I)\(M'*y + μ1*(H(z)-d1)) # eq 49
    u1 = (Mt(y) + μ1*(H(z)-d1))./(DMtM .+ μ1) # eq 49
    u2 = prox_l1(∇(z) - d2,λ/μ2)
    #Update Lagrangian multipliers
    d1 +=  - H(z)  + u1
    d2 +=  - ∇(z)  + u2

    r1[k] = norm(H(z) - u1)
    r2[k] = norm(∇(z) - u2)
    println("$k r1:$(r1[k]) r2: $(r2[k]) dist: $(norm(z-x0,1)/length(x0))")
    if mod(k,100) == 0
        imview3(x0,y,z,titles=["$nx0 × $nx0 ground truth", "$nx × $nx data", "$nx0 × $nx0 reconstruction"])
    end
end

imview(z, title="Reconstruction")
suptitle("$nx0 × $nx0 reconstruction + $nx × $nx boundaries")
plot([nu, nu], [nu, nx0-nu], "w--", color="r")
plot([nx0-nu, nx0-nu], [nx0-nu, nu], "w--", color="r")
plot([nu, nx0-nu], [nx0-nu, nx0-nu], "w--", color="r")
plot([nu, nx0-nu], [nu, nu], "w--", color="r")
tight_layout()

fig = figure(figsize=(10,5))
subplot(1,2,1)
imshow(reshape(y,nx,nx),cmap=:gray)
title("Data $nx × $nx")
subplot(1,2,2)
imshow(M(z), cmap=:gray)
title("TV-reg reconstruction $nx × $nx")

# TBD: Accelerating it with functions -- note cropping/padding is a little different (2 pix wider here)
#M = x->x[1+nu:end-nu, 1+nu:end-nu];
#Mt = x->[zeros(nu, nx0) ; zeros(nx0-2*nu, nu) x zeros(nx0-2*nu, nu); zeros(nu, nx0) ]; #zero padding
#DMtM # TBD: basically ones padded by zeros
