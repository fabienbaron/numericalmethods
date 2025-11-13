using FFTW

function ft2(x)
    return fftshift(fft(ifftshift(x)))
end
   
function ift2(x)
   return ifftshift(ifft(fftshift(x)))
end

function convolve(a, b)
     return real(ift2(ft2(a).*ft2(b)))
 end

function correlate(a, b)
    return real(ift2(ft2(a).*conj(ft2(b))))
end





function sparse_mask(nx, xrange, yrange) # A is 1D vector representing a 2D array
    II = vec(LinearIndices((1:nx,1:nx))[xrange,yrange])
    return sparse(1:length(II), II, ones(Float64, length(II)), length(II), nx*nx)
end

prox_pos(x,λ) = max.(x,zero(eltype(x)))
prox_l1(x, λ)      = sign.(x) .* max.(abs.(x) .- λ, zero(eltype(x)))
prox_l1_plus(x, λ) = max.(x .- λ, zero(eltype(x)))
prox_l0(x, λ)      = ifelse.(abs.(x) .> sqrt(2λ), x, zero(eltype(x)))
prox_l0_plus(x, λ) = max.(ifelse.(abs.(x) .> sqrt(2λ), x, zero(eltype(x))), zero(eltype(x)))

prox_l2sq(x, λ) = x / (1 + λ)
function prox_l2(x, λ)
    nrm = norm(x)
    if nrm > λ 
        return (1 - λ/nrm) * x
    else
        return zero(eltype(x),x)
    end
end

   



function prox_l2dist(u, α, y)
# proximal operator of g = 1/(2α) * || z - y ||^2
# argmin 1/(2α) * || z - y ||^2 + 1/2 || z - u ||^2
# 1/α (z-y) + (z- u) = 0 -> (1/α+1) z = 1/α y + u
# -> (1+α) z = y + α u -> z = (y + α u)/(1+α)
    return (y+α*u)/(1+α)
end

function vsoft(u, α)
     V = sqrt.(u[:,:,1].^2+u[:,:,2].^2)
     R = (u./V).*max.(V.-α,0.0)
     indx = findall(isnan.(R))
     R[indx] .= 0.0
     return R
end

function prox_poisson(u, μ, y)
# note: eq 31 in PIDAL draft paper has a mistake (later corrected)
# argmin ( z + Iℜ(z) - y log z + μ/2 (z - u)^2 )
#      (1 - y/z + μ(z-u)) = 0   and z ≥ 0
#     z - y + μ z(z-u) = 0
#    μ z^2 + (1 - μ u) z - y = 0
#  second degree polynomial with  a = μ, b = (1 - μ u), c = -y
#  solution (-b + √( b^2 - 4 ac)) /2a
# -> (μ u - 1 + √( (1 - μ u)^2 + 4 μ y)/2μ
# -> 0.5*(u - 1/μ + √( (u - 1/μ)^2 + 4 y/μ))
    return 0.5*(u .- 1/μ + sqrt.( (u .- 1/μ).^2 + 4*y/μ) )
end

function TV_mat(nx)
o = ones(nx); D_1D = spdiagm(-1=>-o[1:nx-1],0=>o)
Γ = [kron(spdiagm(0=>ones(nx)), D_1D) ;  kron(D_1D, spdiagm(0=>ones(nx)))];
return Γ
end

function TV_functions(nx)
sx = zeros(nx,nx);
sy = zeros(nx,nx);
sx[1,1]=-1;
sx[1,end]=1;
fsx = fft(sx);
sy[1,1]=-1;
sy[end,1]=1;
fsy = fft(sy);
#gradient operator
grad2d = X -> cat(dims=3, real.(ifft(fsx.*fft(X))), real.(ifft(fsy.*fft(X))));
grad2d_conj = G -> real.(ifft( conj(fsx).*fft(G[:,:,1]) + conj(fsy).*fft(G[:,:,2]) ));
return (grad2d, grad2d_conj, abs2.(fsx)+abs2.(fsy))
end


using Wavelets

function Wav_functions(nwav)
wavelet_bases = [WT.haar, WT.db1, WT.db2, WT.db3, WT.db4, WT.db5, WT.db6, WT.db7, WT.db8];
wavelet_bases = wavelet_bases[1:nwav]
function W(mat)
 n = size(mat)
 Wu = Array{Float64}(undef, n[1], n[2] , nwav);
 for i=1:nwav
     Wu[:,:, i]=dwt(mat, wavelet(wavelet_bases[i]));
 end
 return Wu;
end

function Wt(mat)
 n = size(mat)
 IWu = Array{Float64}(undef, n[1],n[2],nwav);
 for i=1:nwav
     IWu[:,:,i] = idwt(mat[:,:,i], wavelet(wavelet_bases[i]));
 end
 return dropdims(sum(IWu,dims=3),dims=3);
end
return [W, Wt]
end


function gaussian2d(n,m,sigma)
g2d = [exp(-((X-(m÷2+1)).^2+(Y-(n÷2+1)).^2)/(2*sigma.^2)) for X=1:m, Y=1:n]
return g2d/sum(g2d)
end

function conv_otf(otf::Array{ComplexF64,2}, object::Array{Float64,2})
return real.(fftshift(ifft(otf.*fft(object))));
end
