using PyPlot
x = randn(100000) # Values
nbins = 100; # Number of bins
h = plt[:hist](x,nbins) # Histogram
grid("on");xlabel("X");ylabel("Occurences");title("Probability histogram")

x = rand(100000) # Values
nbins = 100; # Number of bins
h = plt[:hist](x,nbins) # Histogram
grid("on");xlabel("X");ylabel("Occurences");title("Probability histogram")

lambda = 1.
y = rand(100000)
x = -log.(1-y)/lambda
