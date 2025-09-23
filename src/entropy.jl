using PyPlot, Statistics
include("fits.jl") # need to install FITSIO
data = readfits("IO.fits.fz")
size(data)
entropy=vec(sum(-data.*log.(data), dims=(1,2))) 

indx = sortperm(entropy)
indx_lowest_entropy = indx[1]
entropy[indx_lowest_entropy]
indx_highest_entropy = indx[end]
entropy[indx_highest_entropy]
imshow(data[:,:,indx_lowest_entropy]) # image with the lowest entropy!

imshow(data[:,:,indx_highest_entropy]) # image with the lowest entropy!





using Distributions
data =  rand.(Distributions.Poisson.(data))
entropy=vec(sum(-data.*log.(data .+ 1e-20), dims=(1,2))) 
indx = sortperm(entropy)
indx_lowest_entropy = indx[1]
entropy[indx_lowest_entropy]
indx_lowest_entropy = indx[end]
entropy[indx_highest_entropy]
stack_lowentropy = sum(data[:,:,indx[1:200]], dims=3)
imshow(stack_lowentropy)
stack_highentropy = sum(data[:,:,indx[end-200:end]], dims=3)
imshow(stack_highentropy)
# In reality we would need to recenter them before stacking...