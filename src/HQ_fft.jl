using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, FFTW
include("view.jl")
x_truth=read(FITS("saturn64.fits")[1]);nx=size(x_truth,1)
x_truth = vec(x_truth); # note: x_truth is a 2D array, but we will work with vectors
sigma= maximum(x_truth)/10*rand(Float64, size(x_truth))

H = matrixdepot("blur", Float64, nx, 3, 2.0, true)
y = H*x_truth + sigma.*randn(Float64,size(x_truth));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix

#
# Spatial gradient matrix for total variation
#
o = ones(nx); D_1D = spdiagm(0=>o,1=>-o[1:nx-1]);
∇ = [kron(D_1D, spdiagm(0=>ones(nx))) ; kron(spdiagm(0=>ones(nx)), D_1D) ];

#
# Spatial gradient *function* using Fast Fourier Transforms
#
sx = zeros(nx,nx); sx[1,1]=1; sx[1,end]=-1; fsx = fft(sx);
sy = zeros(nx,nx); sy[1,1]=1; sy[end,1]=-1; fsy = fft(sy);
#gradient operator
f∇ = X -> real.(cat(dims=3, ifft(fsx.*fft(reshape(X, nx,nx))), ifft(fsy.*fft(reshape(X,nx,nx)))));
f∇t = G -> real(ifft( conj(fsx).*fft(G[:,:,1]) + conj(fsy).*fft(G[:,:,2]) ));

#
# Check that we do get the same thing
#
imshow(reshape(∇*x_truth,(64,128)))
imshow(reshape(f∇(x_truth), (64,128)))
norm(reshape(∇*x_truth,(64,128))-reshape(f∇(x_truth), (64,128)))

#
# Same for convolution
#
# sh = Array(H[1,:]); fsh = fft(reshape(sh,nx,nx));
# fH = X -> real.(ifft(fsh.*fft(reshape(X,nx,nx))));
# imshow(fH(x_truth))
#
# fsh2 = fft(sh);
# fH2 = X -> real.(ifft(fsh2.*fft(X)));
# imshow(reshape(fH2(x_truth), nx, nx))

imshow(reshape(H*x_truth, nx, nx))
x_test = zeros(nx*nx); x_test[64*32+32] = 1;
fsh = fft(H*x_test)./fft(x_test);
fH = X -> real.(ifft(fsh.*fft(X)));
fHt = G -> real(ifft( conj(fsh).*fft(G)));
fHtH = X -> real(ifft( abs2.(fsh).*fft(X)));
#imshow(reshape(fH(x_truth), nx, nx))

# Matrix math
x_sol = (H'*H)\(H'*y) #will be slow with large images
imshow(reshape(x_sol, nx, nx))

# FFT math
x = real(ifft( fft(H'*y)./abs2.(fsh) ))
