#  Implementation based on ADMM-MD in "Deconvolving Images with Unknown Boundaries
# Using the Alternating Direction Method of Multipliers" by Almeida & Figueiredo
# 2D images
# A, M have the same meaning as in the paper
# We're using Wavelet transforms instead of TV here
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

nu = 9 # band of nu pixels=zero around
nx = nx0-2*nu # size of data frame and object
M_mat = sparse_mask(nx0, nu+1:nx0-nu, nu+1:nx0-nu)
M= x->reshape(M_mat*vec(x), nx,nx)
Mt=x->reshape(M_mat'*vec(x),nx0,nx0)
DMtM = reshape(Array(diag(M_mat'*M_mat)), nx0, nx0)

y = M(H(x0)) # Cropped
σ = 1; #maximum(y)/40 # noise level
y += σ*randn(nx,nx) # add Gaussian white noise

nwavs = 9
W, Wt = Wav_functions(nwavs)
WtW = nwavs*I;

# ADMM-MD
u1 = zeros(Float64, nx0,nx0);
u1[nu+1:nx0-nu, nu+1:nx0-nu] = reshape(y,nx,nx)
d1 = zeros(Float64, nx0, nx0)

u2 = W(u1)
d2 = zeros(Float64, nx0, nx0, nwavs)

λ = 0.01
µ1 = minimum([1, 5000λ])
µ2 = 100*λ

niter = 1000
z=Float64[]

r1 = zeros(Float64, niter);
r2 = zeros(Float64, niter);

for k=1:niter
    z = real.(ift2(ft2(μ1*Ht(u1+d1) + μ2*Wt(u2+d2))./(μ1*PSD_H .+ μ2*nwavs))) # diag(WtW) = nwavss*𝟙
    # Apply proximal operators
    u1 = (Mt(y) + μ1*(H(z)-d1))./(DMtM .+ μ1) # eq 49
    u2 = prox_l1(W(z) - d2,λ/μ2)
    #Update Lagrangian multipliers
    d1 += - H(z)  + u1
    d2 += - W(z)  + u2 
    r1[k] = norm(H(z) - u1)
    r2[k] = norm(W(z) - u2)
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
title("Wavelet-reg reconstruction $nx × $nx")
