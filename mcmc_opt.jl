using PyPlot
function ackley(x,y)
    sol = -20.*exp(-0.2*sqrt(0.5*(x.^2 + y.^2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e +20;
    return sol
end

function rosenbrock(x,y,a,b)
    sol = (a - x).^2 + b*(y - x.^2).^2;
    return sol
end

# Hillclimb Map
xx = collect(linspace(-5,5,1000));
yy = collect(linspace(-5,5,1000));
x_x = repmat(xx,1,1000);
y_y = repmat(x_x',1,1);
ackley_map = ackley(x_x,y_y);#rosenbrock(x_x,y_y,1,100);

# Metropolis-Hastings with simulated annealing
tn = 10.; t0 = 1e-5; N = 1000;
# Random walk
x = zeros(N);
y = zeros(N);

x[1] = -4; y[1] = -2;
t = (tn - t0)/2 * (1 + cos(collect(1:N)*pi/N)); #t *= 0.1;
#ackley_vals = [];

clf();
imshow(ackley_map)

for i = 2:N
    stepscale=0.4;
    x_trial = max(min(x[i-1] + stepscale*(rand()*2 - 1),5),-5);
    y_trial = max(min(y[i-1] + stepscale*(rand()*2 - 1),5),-5);
    e_trial = ackley(x_trial,y_trial);
    e_current = ackley(x[i-1],y[i-1]);

    # choose number 6 - temperature schedule
    a = min(1, exp(-(e_trial - e_current)/t[i]));
    println("iteration: $i  acceptance: $a")
    if (rand() < a) # Accept
        x[i] = deepcopy(x_trial);
        y[i] = deepcopy(y_trial);
    else # Reject
        x[i] = deepcopy(x[i-1]);
        y[i] = deepcopy(y[i-1]);
    end

    plot((x[i-1:i]+5)*100,(y[i-1:i]+5)*100);
    #readline();
end

println(x[end], " ", y[end])
