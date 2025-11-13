#  Implementation based on ADMM-MD in "Deconvolving Images with Unknown Boundaries
# Using the Alternating Direction Method of Multipliers" by Almeida & Figueiredo
# 2D images
# A, M have the same meaning as in the paper
# Note this is using the isotropic TV
# TODO: handle noise σ≠1
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
#x0 = Float64.(load("cameraman256.tif"));
nx0 = size(x0,1)
otf = fft(gaussian2d(nx0,nx0,3.0));
otf ./= maximum(abs.(otf));
otf2 = abs2.(otf)  # this is 2D
A = x->real.(ifftshift(ifft(otf.*fft(x))));
At = x->real.(ifftshift(ifft(conj(otf).*fft(x))));
nu = 9 # band of nu pixels=zero around
nx = nx0-2*nu # size of data frame and object
M_mat = sparse_mask(nx0, nu+1:nx0-nu, nu+1:nx0-nu)
M= x->reshape(M_mat*vec(x), nx,nx)
Mt=x->reshape(M_mat'*vec(x),nx0,nx0)
DMtM = reshape(Array(diag(M_mat'*M_mat)), nx0, nx0)

y = M(A(x0))
σ = 1; #maximum(y)/40 # noise level
y += σ*randn(nx,nx) # add Gaussian white noise
imview(reshape(y, nx, nx))
Γ, Γt, Dx2Dy2 = TV_functions(nx0);

# ADMM-MD
u1 = zeros(Float64, nx0,nx0);
u1[nu+1:nx0-nu, nu+1:nx0-nu] = y
d1 = zeros(Float64, nx0, nx0)

u2 = Γ(u1)
d2 = zeros(Float64, nx0, nx0, 2)

λ = 0.05
µ1 = 1e-2
µ2 = 1e-2

niter = 1000
z=Float64[]

r1 = zeros(Float64, niter);
r2 = zeros(Float64, niter);

for k=1:niter
    z = real.(ifft(fft(μ1*At(u1+d1) + μ2*Γt(u2+d2))./(μ1*otf2 + μ2*(Dx2Dy2))))
    # Apply proximal operators
    #u1 = (M'*M + μ1*I)\(M'*y + μ1*(A(z)-d1)) # eq 49
    u1 = (Mt(y) + μ1*(A(z)-d1))./(DMtM .+ μ1) # eq 49
    u2 = prox_l1(Γ(z) - d2,λ/μ2)
    #Update Lagrangian multipliers
    d1 +=  - A(z)  + u1
    d2 +=  - Γ(z)  + u2

    r1[k] = norm(A(z) - u1)
    r2[k] = norm(Γ(z) - u2)
    println("$k r1:$(r1[k]) r2: $(r2[k]) dist: $(norm(z-x0,1)/length(x0))")
    if mod(k,100) == 0
        imview3(x0,y,z)
    end
end

imview(z, title="Reconstruction")
suptitle("Boundaries")
plot([nu, nu], [nu, nx0-nu], "w--")
plot([nx0-nu, nx0-nu], [nx0-nu, nu], "w--")
plot([nu, nx0-nu], [nx0-nu, nx0-nu], "w--")
plot([nu, nx0-nu], [nu, nu], "w--")

fig = figure(figsize=(10,5))
subplot(1,2,1)
imshow(eshape(y,nx,nx),cmap=:gray )
title("Data")
subplot(1,2,2)
imshow(M(z), cmap=:gray)
title("Reconstructed with total variation")

# TBD: Accelerating it with functions -- note cropping/padding is a little different (2 pix wider here)
M = x->x[1+nu:end-nu, 1+nu:end-nu];
Mt = x->[zeros(nu, nx0) ; zeros(nx0-2*nu, nu) x zeros(nx0-2*nu, nu); zeros(nu, nx0) ]; #zero padding
DMtM # TBD: basically ones padded by zeros
