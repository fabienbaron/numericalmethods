using PyPlot
x = randn(100000) # Values
nbins = 100; # Number of bins
h = hist(x,bins=nbins); # Histogram
grid("on");xlabel("X");ylabel("Occurences");title("Probability histogram")

x = rand(100000); # Values
nbins = 100; # Number of bins
h = hist(x,bins=nbins); # Histogram
grid("on");xlabel("X");ylabel("Occurences");title("Probability histogram")
