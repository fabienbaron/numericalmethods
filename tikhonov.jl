using FITSIO
using LinearAlgebra
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/2*rand(Float64, size(x0))
y=x0+sigma.*randn(Float64,size(x0));
W=Diagonal(1.0./sigma.^2);



chi2 = sum( (x0-y).^2 ./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = norm((x0-y)./sigma)^2 # using the l2 norm squared
chi2 = (x0-y)'*W*(x0-y) # the matricial form for any sigma

# Tikhonov solution
λ = 10.0.^(range(-10,10,length=21));
nλ = length(λ);
global mindist = 1e99
#global xopt = zeros(size(x0));
global chi2 = zeros(nλ)
global reg =  zeros(nλ)

for i=1:nλ
    x=(W+λ[i]*I)\(W*y)
    xpos = x.*(x.>0)
    chi2[i] = (x-y)'*W*(x-y)
    reg[i] = norm(x,2)^2;
    dist = norm(xpos-x0,1);
    println("i= $i ", dist);
    if dist<mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(xpos);
    end
end

imview3(x0,y,xopt,figtitle="Tikhonov regularization");
