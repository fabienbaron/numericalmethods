# Numerical methods for Physics and Astronomy

This class will use the Julia programming language as well as Python's matplotlib library for visualizations.

## Prerequisites

## Julia setup

Install the latest Julia from https://julialang.org/downloads/

Install the Atom editor https://atom.io/.

Install the language-julia Atom package from within the Settings/Preferences of Atom, Install tab: this enables syntax highlighting of Julia code.

Install the latex-completions Atom package from within Atom: this enables the autocompletion of LaTeX symbols (α, β, ...)

### PyPlot setup for Windows and Linux

Under Windows and Linux, launch Julia, press ```]``` to trigger the package manager then type ```add PyPlot```.

### PyPlot setup for OSX
Installing the PyPlot julia plotting library on OSX can be a bit tricky.
All the details are here if you need: https://github.com/JuliaPy/PyPlot.jl but here is a summary:

Mac users will need to have XQuartz installed first.
Then there are three possible paths:

Path 1: you've already got your own python installation and want to use it for this course, go to julia and type ```ENV["PYTHON"]="/pathto/python"``` where ```pathto``` is your actual path (check with ```which python``` from terminal).


Path 2: you don't have a python installation  and just want the default Julia on, go to Julia and type ```ENV["PYTHON"]=""```

Path 3: you want to install your own python from scratch, using HomeBrew https://brew.sh/. This is the most complicated way, and slow, but a surer way to have it work.
```
brew install python gcc freetype pyqt
brew link --force freetype
export PATH="/usr/local/bin:$PATH"
export PYTHONPATH="/usr/local/lib/python2.7:$PYTHONPATH"
pip install numpy scipy matplotlib
```
Then edit your ~/.profile to add the two export lines.
Create a file called ```~/.config/matplotlibrc``` with one line ```backend: Qt5Agg```.
Then go to Julia and type ```ENV["PYTHON"]="/usr/local/bin/python"```.

After any of these paths, go to the package manager ```]``` and type ```add PyCall```, then ```add PyPlot```.

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

These two links should get you started:
1. https://learnxinyminutes.com/docs/julia/
2. https://www.sas.upenn.edu/~jesusfv/Chapter_HPC_8_Julia.pdf

If you already master Matlab or Python also check https://cheatsheets.quantecon.org/

Also see http://juliaastro.github.io for useful functions.
