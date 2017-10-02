using FITSIO
include("view.jl")
x0=read(FITS("saturn.fits")[1]);
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/5*rand(size(x0))
b=x0+sigma.*randn(size(x0));
W=spdiagm(1./sigma.^2);
chi2 = sum( (x0-b).^2./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = (x0-b)'*W*(x0-b) # the matricial form for any sigma

# Tikhonov solution
λ = 1e-4
x=(W+λ*I)\(W*b);

imview3(x0,b,x,figtitle="Tikhonov regularization");

norm(x-x0,1);
norm(x-x0,2);
