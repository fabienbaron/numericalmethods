#  Implementation based on ADMM-MD in "Deconvolving Images with Unknown Boundaries
# Using the Alternating Direction Method of Multipliers" by Almeida & Figueiredo
# 2D images
# A, M have the same meaning as in the paper
# Note this is using the isotropic TV
using FITSIO, LinearAlgebra, PyPlot, SparseArrays, FFTW
include("view.jl")
include("admm_utils.jl")
x0 = Float64.(read(FITS("saturn.fits")[1]));
#x0 = Float64.(load("cameraman256.tif"));
nx0 = size(x0,1)
otf = fft(gaussian2d(nx0,nx0,2.0));
otf ./= maximum(abs.(otf));
otf2 = abs2.(otf)  # this is 2D
A = x->conv_otf(otf,x)
At = x->conv_otf_t(otf,x)
nu = 9 # band of nu pixels=zero around
nx = nx0-2*nu # size of data frame and object
M = sparse_mask(nx0, nu+1:nx0-nu, nu+1:nx0-nu)
DMtM = reshape(Array(diag(M'*M)), nx0, nx0)
y = reshape(M*vec(A(x0)), nx,nx)
σ = 1; #maximum(y)/40 # noise level
y += σ*randn(nx,nx) # add Gaussian white noise
imview(reshape(y, nx, nx))
Γ, Γt, Dx2, Dy2 = TV_functions(nx0);



# ADMM-MD
u1 = zeros(Float64, nx0,nx0);
u1[nu+1:nx0-nu, nu+1:nx0-nu] = reshape(y,nx,nx)
d1 = zeros(Float64, nx0, nx0)

u2 = Γ(u1)
d2 = zeros(Float64, nx0, nx0, 2)

λ = 0.001
µ1 = minimum([1, 5000λ])
µ2 = 10*λ

niter = 500
z=Float64[]

for k=1:niter
    z = real.(ifft(fft(μ1*At(u1+d1) + μ2*Γt(u2+d2))./(μ1*otf2 + μ2*(Dx2+Dy2))))
    # Apply proximal operators
    #u1 = (M'*M + μ1*I)\(M'*y + μ1*(A(z)-d1)) # eq 49
    u1 = (reshape(M'*vec(y),nx0,nx0) + μ1*(A(z)-d1))./(DMtM .+ μ1) # eq 49
    u2 = prox_l1(Γ(z) - d2,λ/μ2)
    #Update Lagrangian multipliers
    d1 = d1 - A(z)  + u1
    d2 = d2 - Γ(z)  + u2
    println("dist: $(norm(z-x0,1)/length(x0))")
    #if mod(k,100) == 0
    #    imview3(reshape(x0,nx0,nx0), reshape(y,nx,nx), reshape(z,nx0,nx0))
    #end
end
imview3(reshape(x0,nx0,nx0), reshape(y,nx,nx), reshape(z,nx0,nx0))

imview(reshape(z, nx0, nx0))
plot([nu, nu], [nu, nx0-nu], "w--")
plot([nx0-nu, nx0-nu], [nx0-nu, nu], "w--")
plot([nu, nx0-nu], [nx0-nu, nx0-nu], "w--")
plot([nu, nx0-nu], [nu, nu], "w--")
