using FITSIO, LinearAlgebra, Printf, SparseArrays, FFTW, MatrixDepot, Noise

include("view.jl")
x0 = 4*read(FITS("saturn64.fits")[1]);
nx = size(x0,1)
A = matrixdepot("blur", Float64, nx, 3, 2.0, true)
x0 = vec(x0)
y = poisson(A*x0)


function prox_l1(ν, μ)
    return sign.(ν).*max.(abs.(ν).-1/μ,0.0)
end


function prox_poisson(ν, μ, y) # eq 31 Restoration of Poissonian Images Using Alternating Direction Optimization, Mário A. T. Figueiredo
    return 0.5*(ν .- 1/μ + sqrt.( (ν .- 1/μ).^2 + 4*y/μ) )
end

o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
∇ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
A = I;
Γ = ∇;


# PIDAL-FA
u1 = deepcopy(y)
u2 = Γ*y
u3 = deepcopy(y)

d1 = zeros(Float64, nx*nx)
d2 = zeros(Float64, 2nx*nx)
d3 = zeros(Float64, nx*nx)

τ = 0.08
μ = 60*τ/maximum(y)
niter = 40
z=Float64[]
for k=1:niter
z = (A'*A + Γ'*Γ + I)\(A'*(u1 + d1) + Γ'*(u2 + d2) + (u3 + d3))
# Apply proximal operators
u1 = prox_poisson(A*z - d1,μ,y)
u2 = prox_l1(Γ*z - d2,μ/τ)
u3 = max.(z-d3,0.0)
#Update Lagrangian multipliers
d1 = d1 - A*z  + u1
d2 = d2 - Γ*z  + u2
d3 = d3 - z  + u3
end
imview(reshape(z,nx,nx))
