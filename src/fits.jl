using FITSIO

function readfits(fitsfile; normalize = false, vectorize=false)
    x=[]
    if fitsfile[end-1:end]=="fz"
    x = (read((FITS(fitsfile))[2]))
    else
    x = (read((FITS(fitsfile))[1]))
    end
    if normalize == true
    x ./= sum(x)
    end
    if vectorize == true
    x = vec(x)
    end
return x;
end

function writefits(data, fitsfile)
f = FITS(fitsfile, "w");
write(f, data);
close(f);
end
