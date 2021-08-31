# This code is meant to be input line by line into the REPL
using Distributions, PyPlot
d = Uniform(0,1) # the most basic Distribution
x = rand(d, 100000)
hist(x,bins=100)
# Statistics on the sample
mean(x)
median(x)
var(x)
# Statistics on the population
mean(d)
median(d)
var(d)
support(d)

#PDF and CDF
pdf(d,0.3)
cdf(d,0.3)
ccdf(d,0.3)
logpdf(d,0.3)

# Dice!
d = DiscreteUniform(1,6)
x = rand(d, 100000)
hist(x,bins=100)
pdf(d,2)
std(d)

# Normal distribution, e.g. IQ distribution
d = Normal(100,15)
std(d)
x = rand(d, 100000)
mean(x)
