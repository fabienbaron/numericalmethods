using LinearAlgebra
function noll_indexes(j)
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    """

    if j < 1
        error("Zernike index j must be a positive integer")
    end
    n = 0
	j1 = j-1
	while (j1 > n)
		n += 1
		j1 -= n
    end
	m = (-1)^j * (mod(n,2) + 2 * div(j1+mod(n+1,2),2))
    return (n, m)
end


function stirling(n) # improved stirling formula
if(n<20)
     return factorial(n)
 else
     return (sqrt(2*pi*n)*(n/e)^n*exp(1/(12*n)-1/(360*n^3)))
end
end

#
# Based on poppy.zernike
#
function zernike_rad(n, m, rho)
"""Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the unit circle
"""
    m = Int(abs(m))
    n = Int(abs(n))
    output = zeros(size(rho))
    if isodd(n - m)
        return 0
    else
        for k=0:div(n - m,2)
            output += (-1)^k * stirling(n - k) / (stirling(k) * stirling(div(n + m,2) - k) * stirling(div(n - m, 2) - k)) * rho.^(n - 2 * k)
        end
        return output
    end
end

function zernike(j; npix::Int64=128, diameter::Union{Float64,Int64}=128, cent_x::Float64=-1.0, cent_y::Float64=-1.0, outside::Float64=0.0, noll_normalize::Bool=true, centered::Bool=false)
  """Return the Zernike polynomial Z[j] for a given pupil.
  For this function the desired Zernike is specified by the Noll index j.
  See zernike_nm for an equivalent function in which the polynomials are ordered by n,m
  The expressions for the Zernike terms follow the normalization convention
  of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

          Parameters
          ----------
          j : int
              Noll order
          npix: int
              Number of pixels of the support.
          diameter:
              Desired pixel diameter for the circular pupil
          noll_normalize : bool
              As defined in Noll et al. JOSA 1976, the Zernike definition is
              modified such that the integral of Z[n, m] * Z[n, m] over the
              unit disk is pi exactly. To omit the normalization constant,
              set this to False. Default is True.

          Returns
          -------
          zern : 2D array Z[j] evaluated at each (rho, theta)
  """
if (centered == true)|(cent_x==-1.0)|(cent_y==-1.0)
  cent_x = div(npix,2)+1
  cent_y = div(npix,2)+1
end

n, m = noll_indexes(j);
x = (collect(1:npix) .- cent_x) / (diameter / 2.)
y = (collect(1:npix) .- cent_y) / (diameter / 2.)
xx = repeat(x,1,npix)
yy = repeat(y,1,npix)'
rho = sqrt.(xx.^2 + yy.^2)
theta = atan.(yy, xx)
aperture = ones(size(rho))
aperture[findall(rho.>1)] .= outside  # this is the aperture mask
norm_coeff=1.0/(diameter/2)
if (noll_normalize==true)
    norm_coeff = (2*(n+1)/(1+(m==0)))^0.5/(diameter/2)
end
if (m > 0)
    return norm_coeff*zernike_rad(n, m, rho).*cos.(m * theta).*aperture
end
if (m < 0)
    return norm_coeff*zernike_rad(n, m, rho).*sin.(-m * theta).*aperture
end

return norm_coeff*zernike_rad(n, 0, rho).*aperture
end

function circular_aperture(;npix::Int64=128, diameter::Union{Float64,Int64}=128, cent_x::Float64=64.5, cent_y::Float64=64.5, outside::Float64=0.0, centered::Bool = false, normalize::Bool=false)
  """
  Returns a 2D aperture of the desired diameter pixels, centered on (cent_x,cent_y) and on support npix X npix
  """
if centered == true # following fftshift convention
  cent_x = div(npix,2)+1
  cent_y = div(npix,2)+1
end

x = (collect(1:npix) .- cent_x) / (diameter / 2.)
y = (collect(1:npix) .- cent_y) / (diameter / 2.)
xx = repeat(x,1,npix)
yy = repeat(y,1,npix)'
rho = sqrt.(xx.^2 + yy.^2)
aperture = ones(size(rho))
aperture[findall(rho.>1)] .= outside  # this is the aperture mask

if normalize==true
    aperture /= norm(aperture)
end
return aperture
end
