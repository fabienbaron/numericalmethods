# Generate data
θ=rand(20)*pi;
x=3*cos.(θ)+.5*randn(20); y = 5*sin.(θ)+.5*randn(20);

#Visualize the true ellipse
θ_all=linspace(-pi,pi,1000)
scatter(3*cos.(θ_all), 5*sin.(θ_all))
