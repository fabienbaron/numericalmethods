using FITSIO
include("view.jl")
x0=read(FITS("saturn64.fits")[1]);
x0 = vec(x0); # note: x0 is a 2D array, but we will work with vectors
sigma= maximum(x0)/2*rand(size(x0))
b=x0+sigma.*randn(size(x0));
W=spdiagm(1./sigma.^2);
chi2 = sum( (x0-b).^2./sigma.^2) # the conventional way to write the chi2 for diagonal sigma
chi2 = norm((x0-b)./sigma)^2 # using the l2 norm squared
chi2 = (x0-b)'*W*(x0-b) # the matricial form for any sigma

# Tikhonov solution
λ = 2.^(linspace(-100,100,201))
nλ = length(λ)
chi2=zeros(nλ)
reg = zeros(nλ)
x=zeros(size(x0));
xopt = zeros(size(x0));
dist = 1e99

for i=1:nλ
    x=(W+λ[i]*I)\(W*b)
    chi2[i] = (x-b)'*W*(x-b)/length(x0)
    reg[i] = norm(x,2)^2
    clf();
    println("It: ", i, " ", λ[i], " dist: ", norm(x-x0,1));
    if norm(x-x0,1)<dist
        dist = norm(x-x0,1);
        xopt = copy(x);
    end
#    imview(reshape(x,64,64))
#    readline();
end

clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,b,xopt,figtitle="Tikhonov regularization");


nx = 64;
o = ones(nx);
D_1D = spdiagm((-o[1:nx-1],o), (-1,0), nx, nx);
D = [kron(speye(nx), D_1D) ;  kron(D_1D, speye(nx))];
DtD = D'*D;

for i=1:nλ
    x=(W+λ[i]*DtD)\(W*b)
    chi2[i] = (x-b)'*W*(x-b)/length(x0)
    reg[i] = norm(x,2)^2
    clf();
    println("It: ", i, " ", λ[i], " dist: ", norm(x-x0,1));
    if norm(x-x0,1)<dist
        dist = norm(x-x0,1);
        xopt = copy(x);
    end
#    imview(reshape(x,64,64))
#    readline();
end
clf();
scatter(reg,chi2)
xlabel("Regularization")
ylabel("Chi2")
imview3(x0,b,xopt,figtitle="Tikhonov regularization");
