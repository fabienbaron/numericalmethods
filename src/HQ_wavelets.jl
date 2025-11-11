using FITSIO, LinearAlgebra, Printf, MatrixDepot, SparseArrays, Wavelets
include("view.jl")

# # Square patches
#  nx = 256; x_truth = zeros(nx,nx);
#  x_truth[10:15,34:54] .= 5.0;
#  x_truth[26:35,14:24] .= 8.0;
#  x_truth[37:60,10:24] .= 9.0;
#  x_truth = vec(x_truth);

# x_truth=read(FITS("saturn.fits")[1]);
# nx=size(x_truth,1)
# # Example of transform
# wt = wavelet(WT.cdf97, WT.Lifting)
# xt = dwt(x_truth, wt)
# imview(xt)
# # To see with better contrast
# imview(abs.(xt).^.2)

# # The inverse wavelet transform:
# idwt(xt, wt)
# norm(idwt(xt, wt) - x_truth) # the difference with the original image is negligible

# norm(abs.(xt).<1,1)
# # Copy
# thresholds = [5, 10, 50, 100, 200, 500, 1000]
# xt_filtered = repeat(xt, 1,1, length(thresholds))

# for i=1:length(thresholds)
#     t = thresholds[i]
#     println("Percentage of coefficients <", t, " = ",norm(abs.(xt).<t,1)/length(xt)*100)
#     xt_filtered[abs.(xt).<t, i] .= 0
#     imshow(idwt(xt_filtered[:,:,i], wt))
#     readline()
# end

#
# Regularization via Wavelet bases
# An ADMM implementation of the analysis method
#
x_truth=vec(read(FITS("saturn64.fits")[1]));
nx=64
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


prox_l1(x,λ) = sign.(x).*max.(abs.(x).-λ,0)
prox_l0(x, λ) = ifelse.(abs.(x) .> sqrt(2λ), x, zero(eltype(x)))
prox_l2sq(x, λ) = x / (1 + λ)
function prox_l2(x, λ)
    nrm = norm(x)
    if nrm > λ 
        return (1 - λ/nrm) * x
    else
        return zero(eltype(x),x)
    end
end

global mindist = 1e99;
μ = 0.06#1.0;#0.06;  # use 0.06 for saturn, 1.0 for the square patches
# initialization
x = deepcopy(y)
z = W(x)
ρ = 0.001;

for iter=1:50
    # x subproblem
    x=(H'*Σ*H+ρ*WtW)\(H'*Σ*y+ρ*Wt(z)); 
    # z subproblem
    z = prox_l1(W(x),μ/ρ); #
    chi2 = ((y-H*x)'*Σ*(y-H*x))[1]/length(y)
    reg = μ*norm(W(x),1);
    aug = norm(z-W(x),2)^2;
    println("chi2r = ", chi2, " reg= ", reg, " aug= ", aug, " ρ*aug= ", ρ*aug);
    # increase ρ
    ρ = 1.5*ρ
    subplot(1,2,1)
    suptitle("Iteration $iter")
    imshow(reshape(x,(nx,nx)))
    subplot(1,2,2)
    imshow(reshape(abs.(W(x))[:,1:2], nx, nx * 2).^.1)
end

subplot(1,2,2)
imshow(reshape(y, nx, nx))
