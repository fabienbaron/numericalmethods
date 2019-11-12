using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, Wavelets
include("view.jl")

# Square patches
 nx = 64; x_truth = zeros(nx,nx);
 x_truth[10:15,34:54] .= 5.0;
 x_truth[26:35,14:24] .= 8.0;
 x_truth[37:60,10:24] .= 9.0;
 x_truth = vec(x_truth);


 x_truth=read(FITS("saturn64.fits")[1]);nx=size(x_truth,1)
 x_truth = vec(x_truth); # note: x_truth is a 2D array, but we will work with vectors



#
# Wavelet choice
#
wavelet_bases = [WT.haar, WT.db1, WT.db2, WT.db3, WT.db4, WT.db5, WT.db6, WT.db7, WT.db8];
nwav = length(wavelet_bases)

function W(mat)
 n = length(mat)
 nx = Int.(sqrt(n))
 Wu = Array{Float64}(undef, n,nwav);
 for i=1:nwav
     Wu[:, i]=vec(dwt(reshape(mat,nx,nx), wavelet(wavelet_bases[i])));
 end
 return Wu;
end

function Wt(mat)
 n = size(mat,1)
 nx = Int.(sqrt(n))
 IWu = Array{Float64}(undef, n,nwav);
 for i=1:nwav
     IWu[:,i] = vec(idwt(reshape(mat[:,i],(nx,nx)), wavelet(wavelet_bases[i])));
 end
 return vec(sum(IWu,dims=2))/nwav;
end

WtW =I; # Wt(W(x)) ./ x

# Check that L2 norm is conserved ||Wx||_2 = ||x||_2
# this implies we cannot use Tikhonov and get useful results
norm(x_truth,2), [norm(W(x_truth)[:,i],2) for i=1:nwav]



sigma= maximum(x_truth)/10*rand(Float64, size(x_truth));

H = matrixdepot("blur", Float64, nx, 3, 2.0, true);
y = H*x_truth + sigma.*randn(Float64,size(x_truth));
Σ = Diagonal(1.0./sigma.^2); # covariance matrix

function prox_l1(z,α)
return sign.(z).*max.(abs.(z).-α,0)
end

global mindist = 1e99;
μ = 0.06;  # use 0.03 for saturn, 1.0 for the square patches
# initialization
x = deepcopy(y)
z = W(x)
ρ = 0.001;

for iter=1:50
# x subproblem
global x=(H'*Σ*H+ρ*WtW)\(H'*Σ*y+ρ*Wt(z)); # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2
# z subproblem
global z = prox_l1(W(x),μ/ρ); # should minimize μ*norm(z,1)+0.5*ρ*norm(z-∇*x,2)^2

chi2 = ((y-H*x)'*Σ*(y-H*x))[1]/length(y)
reg = μ*norm(W(x),1);
aug = norm(z-W(x),2)^2;
println("chi2 = ", chi2, " reg= ", reg, " aug= ", aug, " ρ*aug= ", ρ*aug);
# increase ρ
global ρ = 1.5*ρ
imview(reshape(x,(64,64)))
end
