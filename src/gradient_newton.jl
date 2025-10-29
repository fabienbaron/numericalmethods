using LinearAlgebra, PyPlot
a=1; b=100; 
# Rosenbrock
f =x->(a - x[1])^2 + b*(x[2] - x[1]^2)^2;
# Analytic gradient (2)
g = x-> [-2(a-x[1])-4b*(x[2]-x[1]^2)*x[1], 2b*x[2]- 2b*x[1]^2];
# Analytic Hessian (2x2)
h = x->[2-4b*x[2]+12b*x[1]^2 -4b*x[1] ; -4b*x[1] 2b]

# Here is how we would have done it with Automatic Differentiation
using Zygote
g_ad = x->gradient(f, x)[1]
h_ad=  x->hessian(f, x)
x=rand(2)*10; 
norm(g(x) - g_ad(x))
norm(h(x) - h_ad(x))

#
# Newton's method
#
α = 1.0
for n=1:20
    x = x - α*h(x)\g(x)
    println("Iteration $n: x = $x, f(x)= $(f(x))")
end
