# Numerical methods for Physics and Astronomy

This class will use the Julia programming language as well as Python's matplotlib library for visualizations.

## Julia setup

Install Julia from https://julialang.org/downloads/

Install [Pyplot](https://github.com/JuliaPy/PyPlot.jl) by running `Pkg.add("PyPlot")` on the Julia command line (or `julia -E Pkg.add("PyPlot")` on the OS command line). Mac users will need to have [XQuartz](https://www.xquartz.org/) installed before, and to create an `alias julia="/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia"` line to their .bash_profile.


Install the Atom editor https://atom.io/.

Install the language-julia Atom package from within Atom (CTRL+comma to open Settings, then Install tab).

More advanced users can directly use Juno (a Julia + Atom remix) http://junolab.org/

## Check that you can use the REPL

1. Launch julia from the command line.

2. Check that it can do basic operations, e.g. input ```2+2``` then press enter.

3. Check that it can plot basic functions using the PyPlot library.

```julia
using PyPlot #note: on MacOS julia will install lots of files !
x=linspace(-pi, pi, 100)
plot(x,sin.(x))
```

## Learning Julia

These two links should get you started:
1. https://learnxinyminutes.com/docs/julia/
2. https://math.mit.edu/~stevenj/Julia-cheatsheet.pdf (ignore MIT specifics)

If you already master Matlab or Python also check https://cheatsheets.quantecon.org/

Astronomy students should also see http://juliaastro.github.io for useful functions.
