using FITSIO
using LinearAlgebra
using Printf
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/2*rand(Float64, size(x0))
y=x0+sigma.*randn(Float64,size(x0));
C=Diagonal(1.0./sigma.^2); # covariance matrix

# Chi2 and reduced chi2
#
chi2 = sum( (x0-y).^2 ./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = norm((x0-y)./sigma)^2 # using the l2 norm squared
chi2 = (x0-y)'*C*(x0-y) # the matricial form for any sigma
chi2r = chi2/length(y)

# Classic Tikhonov solution
λ = 10.0.^(range(-15,15,length=201));
nλ = length(λ);
global mindist = 1e99;
global chi2 = zeros(nλ);
global reg =  zeros(nλ);
global obj =  zeros(nλ);

for i=1:nλ
    x=(C+λ[i]*I)\(C*y);
    #x = x.*(x.>0)
    chi2[i] = ((x-y)'*C*(x-y))[1]
    reg[i] = norm(x,2)^2;
    obj[i] = chi2[i] + λ[i]*reg[i];
    dist = norm(x-x0,1);
    if dist<mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(x);
    end
    @printf("It: %3i obj:%8.1e λ:%8.1e chi2r: %5.2f chi2: %8.2f  λ*reg: %8.2f reg: %8.2f dist: %8.2f\n", i, λ[i], obj[i], chi2[i]/length(y), chi2[i], reg[i], λ[i]*reg[i], dist  );
end

# Plot L curve
fig = figure("L Curve",figsize=(15,5))
clf()
scatter(λ.*reg,chi2)
gca().set_yscale("log")
gca().set_xscale("log")
xlabel("Regularization")
ylabel("Chi2")

# Display object, noisy image and reconstruction
imview3(x0,y,xopt,figtitle="Tikhonov regularization");

#
# Total squared variation
#
using SparseArrays
nx = 64;
o = ones(nx);
D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
D = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
DtD = D'*D;
global mindist = 1e99;

for i=1:nλ
    x=(C+λ[i]*DtD)\(C*y)
    chi2[i] = ((x-y)'*C*(x-y))[1]/length(x0)
    reg[i] = norm(x,2)^2
    xpos = x.*(x.>0)
    dist = norm(xpos-x0,1);
    if dist < mindist
        global mindist = deepcopy(dist);
        global xopt = deepcopy(xpos);
    end
    @printf("It: %3i λ:%8.1e chi2: %8.2f dist: %8.2f\n", i,λ[i],chi2[i], dist  );
    #imview(reshape(xpos,64,64))
    #readline();
end
clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,y,xopt,figtitle="Total squared variation regularization");


function matrix_to_vector(input)
    #Converts the input matrix to a vector by stacking the rows in a specific way explained here
    # ouput_vector -- a column vector with size input.shape[0]*input.shape[1]
    # flip the input matrix up-down because last row should go first
    return vec(reverse(input, dims=1)')
end
