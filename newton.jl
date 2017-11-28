# Minimize Rosenbrock
x=[5.,-2.];
h=zeros(2,2);
α = 1.0

for n=1:20
g=[-2(1-x[1])-400(x[2]-x[1]^2)*x[1], 200x[2]-200x[1]^2];
h[1,1]=2-400x[2]+1200x[1]^2;
h[1,2]=-400x[1];
h[2,1]=-400x[1];
h[2,2]=200;
x = x - α*h\g
println("Iteration $n: x = $x")
end
