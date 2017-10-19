using PyPlot
# Generate data
θ=rand(20)*2*pi;
a = 3
b = 5
σ = 0.5
x=a*cos.(θ)+σ*randn(20); y = b*sin.(θ)+σ*randn(20);
scatter(x,y);

#Visualize the true ellipse
θ_all=linspace(-pi,pi,1000)
scatter(3*cos.(θ_all), 5*sin.(θ_all))

# Given data (x,y), estimate a and b

# Grid sample for a and b

a = linspace(-10, 10, 100);
b = linspace(-10, 10, 100);
chi2 = zeros(100,100);

for i=1:length(a)
    for j=1:length(b)
        chi2[i,j] = sum( (x-a[i]*cos.(θ)).^2  + (y-b[j]*cos.(θ)).^2)/σ^2
    end
end

minpos = ind2sub(size(chi2), indmin(chi2))
println("Minimun at: a=", a[minpos[1]], " b = ", b[minpos[2]])
imshow(chi2)

# Compute error bars from chi2




# Compute error bars from bootstrap
