using PyPlot
N = 100 # Number of grains of sand
x = 2*rand(N) .- 1 
y = 2*rand(N) .- 1
indx_out = findall(x.^2 .+ y.^2  .> 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(x,y, color="red")
plt.scatter(x[indx_out],y[indx_out], color="blue")
gca().set_aspect("equal", "box")
gca().add_artist(plt.Circle((0,0), 1, fill=false))
println("Grains in the circle: ", N-length(indx_out)," out of ", N)
println("Approximation of  π: ", 4*(N-length(indx_out))/N)