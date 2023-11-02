using FFTW, PyPlot, FITSIO
# Basic FFT of a 1D signal to identify frequencies
t=range(0,0.5,length=1000); # signal is sampled uniformly
w=10*2π;
x=sum([n*cos.(n*w*t) for n=1:5],dims=1)[1]
figure("Signal"); plot(x);
X=fft(x)
figure("Modulus FFT"); plot(abs.(X)[1:50]);

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

function convolve_fouriermask(a, mask) 
    return real.(ifft2(fft2(a).*mask));
end



# FFTs of a rectangle
x = zeros(64,64)
x[12:35, 23:46] .= 1;
total = sum(x); 
X1 = fft(x);
X2 = fft2(x);
print("Sum(x) = ", total, "\nZero frequency in X = X[1] = ", X[1]);
fig = figure("fftshift: location of zero frequency", figsize=(8,4));
subplot(121); imshow(abs.(X1)); title("Default FFT: zero freq in corner")
subplot(122); imshow(abs.(X2)); title("fftshift: zero freq at center")
tight_layout()



# FFT vs disc size
N=256
yy = repeat(collect(range(1, N, length=N)).-div(N,2), 1, N);
xx = yy'; rr = sqrt.(xx.^2 + yy.^2);
disc1 = rr.<20; # small disc
disc2 = rr.<60; # larger disc
fig = figure("2D objects", figsize=(8,4));
subplot(121); imshow(abs.(fft2(disc1))); title("|FFT| small disc")
subplot(122); imshow(abs.(fft2(disc2))); title("|FFT| larger disc")
tight_layout()

# Fourier filtering - we'll be removing the high or low spatial frequencies
image = read(FITS("saturn.fits")[1]);
N=size(image,1)
yy = repeat(collect(range(1, N, length=N)).-div(N,2), 1, N);
xx = yy'; rr = sqrt.(xx.^2 + yy.^2);
mask = rr.<10; # this will be our mask in Fourier space
mask_hi = mask; # we need to apply our mask at the right location in Fourier space
mask_low = 1 .- mask_hi
high_filtered_image = convolve_fouriermask(image, mask_hi)
low_filtered_image = convolve_fouriermask(image, mask_low)
fig = figure("Filtering example", figsize=(12,4));
subplot(131); imshow(image); title("Truth")
subplot(132); imshow(high_filtered_image); title("High Freq Filtered")
subplot(133); imshow(low_filtered_image); title("Low Freq Filtered")
tight_layout()

#
# Other applications of the FFT
#

# correlation
x1=zeros(64,64); x1[23:34, 16:23].=1; x1[45:56,23:56].=2
x2=circshift(x1,(5,-8))+randn(64,64)

fig = figure()
imshow(correlate(x1,x1)); # Autocorrelation
locmax1 = findmax(correlate(x1,x1))[2] ; # maximum is at the center

imshow(correlate(x1,x2)); # Cross-correlation of x1 with x2
locmax2 = findmax(correlate(x1,x2))[2] ; # maximum is not at the center
print("Estimated shift: ",locmax1 - locmax2);

# convolution

# shift (subpixel shifts are possible !)
yy = repeat(collect(range(1, N, length=N)).-div(N,2), 1, N);
xx = yy';
cis.(0.6*xx) # cis(ϕ) is just exp(iϕ)
shifted_image = real.(ifft(fftshift(cis.(0.6*xx)).*fft(image)))
fig=figure("Image Shifting via FFT")
subplot(121); imshow(image); title("Original")
subplot(122); imshow(shifted_image); title("Shifted")
tight_layout()
