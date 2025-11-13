using FITSIO, LinearAlgebra, FFTW, Printf, PyPlot
include("view.jl"); include("admm_utils.jl");
# We can work in 2D since we will be using functions
x0 = Float64.(read(FITS("catCC0.fits")[1])[:,:,2])/50;
nx=size(x0,1)
psf = gaussian2d(nx,nx,5.0);
H = x->convolve(x, psf)
Ht = x->correlate(x, psf)
PSD_H = abs2.(ft2(psf)) 

# spatial gradient
c = nx÷2+1;
Gx = zeros(nx,nx); 
Gy = zeros(nx,nx); 
Gx[c-1:c, c] = [ -1 1]
Gy[c, c-1:c] = [-1 1]
∇ = X -> cat(dims=3, convolve(X, Gx), convolve(X,Gy));
∇t = X -> real.( correlate(X[:,:,1], Gx) + correlate(X[:,:,2], Gy)) 
PSD_∇ = abs2.(ft2(Gx)) .+ abs2.(ft2(Gy))


sigma= 1.0; #maximum(x0)/10
y = H(x0) + sigma.*randn(Float64,size(x0));
Σ = 1.0./sigma.^2 *I; # covariance matrix

mindist = 1e99;
μ = .1; 
x = copy(y)
z = ∇(x)
ρ = 1e-3;

for iter=1:200
# x subproblem
    # Precompute the diagonal elements for the division
    D = Σ*PSD_H + ρ*PSD_∇ .+ 1e-12
    x= real(ift2(ft2(Ht(Σ*y)+ρ*∇t(z)) ./ D)) ; # should minimize 0.5*norm(H*x-y,2)^2+0.5*ρ*norm(z-∇*x,2)^2
# z subproblem
    z = prox_l1(∇(x),μ/ρ); # should minimize μ*norm(z,1)+0.5*ρ*norm(z-∇*x,2)^2
    chi2 = norm((y-H(x)./sigma))^2
    reg = μ*norm(∇(x),1);
    aug = ρ*norm(z-∇(x),2)^2;
    println(@sprintf("iter %3d | full = %.2f | obj = %.2f |  χ²_r = %.2f | reg = %.2f | ||∇x-z||^2 = %.2f", iter, chi2+reg+aug, chi2+reg, chi2/length(y), reg, aug/ρ))
    # increase ρ
    ρ = 1.2*ρ
    if mod(iter, 10)==0
        figure("Reconstruction", (20, 4))
        subplot(1,4,1)
        suptitle("Iteration $iter")
        title("Data y")
        imshow(y, cmap="gist_gray")
        subplot(1,4,2)
        title("Ground truth x0")
        imshow(x0, cmap="gist_gray")
        subplot(1,4,3)
        title("Current x")
        imshow(x, cmap="gist_gray")
        subplot(1,4,4)
        title("Spatial gradient ∇x")
        imshow(∇(x)[:,:,1], cmap="gist_gray")
    end
end
