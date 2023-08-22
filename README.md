# Numerical methods for Physics and Astronomy

This class will use the Julia programming language as well as Python's matplotlib library for visualizations.

## Prerequisites

## Julia setup

Install the latest Julia from https://julialang.org/downloads/

## VSCode Setup

Enable the Julia extension in VScode.

### PyPlot setup for Windows and Linux

Under Windows and Linux, launch Julia, press ```]``` to open its package manager then type ```add PyPlot```.

### PyPlot setup for OSX
To use the default python within Julia, type :
```
ENV["PYTHON"]=""
ENV["MPLBACKEND"]="qt5agg"
```
Then go to the package manager ```]``` and type ```add PyCall```, then ```add PyPlot```.

#
## Check that you can use the REPL

1. Launch julia from the command line.

2. Check that it can do basic operations, e.g. input ```2+2``` then press enter.

3. Check that it can plot basic functions using the PyPlot library.

```julia
using PyPlot
x=range(-pi, pi, length=100)
plot(x,sin.(x))
grid()
```

## Learning Julia

These three links should get you started:
1. https://learnxinyminutes.com/docs/julia/
2. https://www.sas.upenn.edu/~jesusfv/Chapter_HPC_8_Julia.pdf
3. https://computationalthinking.mit.edu/Fall23/

If you already master Matlab or Python also check https://cheatsheets.quantecon.org/

Also see http://juliaastro.github.io for useful functions in Astronomy.
