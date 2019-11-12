using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, Wavelets, FFTW
include("view.jl")
x_truth=read(FITS("saturn64.fits")[1]);nx=size(x_truth,1)
x_truth = vec(x_truth); # note: x_truth is a 2D array, but we will work with vectors
sigma= maximum(x_truth)/10*rand(Float64, size(x_truth))

H = matrixdepot("blur", Float64, 64, 3, 2.0, true)
y = H*x_truth + sigma.*randn(Float64,size(x_truth));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix

#
# Total squared variation
#
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];

# initialization
x = deepcopy(y)
z = ∇*x
ρ = 1;

# The issue is to make the following faster !
global x=(H'*Σ*H+ρ*∇'*∇)\(H'*Σ*y+ρ*∇'*z); # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2

#
#
# Tricks for faster calculations on large images
#
imshow(reshape(∇*x_truth,(64,128)))


x_truth = reshape(x_truth, (64,64))
using FFTW
sx = zeros(nx,nx);
sy = zeros(nx,nx);
sx[1,1]=-1; sx[1,end]=1; fsx = fft(sx);
sy[1,1]=-1; sy[end,1]=1; fsy = fft(sy);
#gradient operator
∇ = X -> real.(cat(dims=3, ifft(fsx.*fft(X)), ifft(fsy.*fft(X))));
∇t = G -> real(ifft( conj(fsx).*fft(G[:,:,1]) + conj(fsy).*fft(G[:,:,2]) ));
imshow(reshape(∇(x_truth)[:,:,1])
imshow(reshape(∇(x_truth)[:,:,2])

wavelet_bases =[WT.db1, WT.db2, WT.db3, WT.db4, WT.db5, WT.db6, WT.db7, WT.db8, WT.haar];
nspat = length(wavelet_bases)
Wu = Array{Float64}(n,nspat);
IWu = Array{Float64}(n,nspat);


function W(mat)
 for i=1:nspat
     Wu[:, i]=vec(dwt(reshape(mat,nv,nv), wavelet(wavelet_bases[i])));
 end
 return Wu;
end

function Wt(mat)
 for i=1:nspat
     IWu[:,i] = vec(idwt(reshape(mat[:,i],(nv,nv)), wavelet(wavelet_bases[i])));
 end
 return sum(IWu,2);
end
WtW = nspat*I;
