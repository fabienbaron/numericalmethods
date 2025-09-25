using FFTW, PyPlot, FITSIO, LinearAlgebra
function fft2(x)
    return fftshift(fft(fftshift(x)))
end
function ifft2(x)
    return ifftshift(ifft(ifftshift(x)))
end

function convolve(a, b) 
    return real.(ifft2(fft2(a).*fft2(b)));
end
function correlate(a, b) 
    return real.(ifft2(fft2(a).*conj(fft2(b))));
end

function fouriermask(a, mask) 
    return real.(ifft2(fft2(a).*mask));
end
