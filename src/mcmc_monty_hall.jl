# Monty Hall Simulation in Julia

# Function to simulate one round of the Monty Hall game
# switch_choice: whether the contestant switches after Monty opens a door
function monty_hall_trial(switch_choice::Bool)::Bool
    doors = [1, 2, 3]
    
    # Randomly place the car behind one of the doors
    car_door = rand(doors)
    
    # Contestant randomly picks a door
    initial_choice = rand(doors)
    
    # Monty opens a door that:
    # - Is not the car door
    # - Is not the contestant's initial choice
    possible_doors_for_monty = setdiff(doors, [car_door, initial_choice])
    monty_opens = rand(possible_doors_for_monty)
    
    # If switching, pick the remaining unopened door
    final_choice = switch_choice ? setdiff(doors, [initial_choice, monty_opens])[1] : initial_choice
    
    # Return true if contestant wins (i.e., chooses the car)
    return final_choice == car_door
end

# Run many simulations and compare win rates
function simulate_monty_hall(n::Int)
    stay_wins = count(_ -> monty_hall_trial(false), 1:n)
    switch_wins = count(_ -> monty_hall_trial(true), 1:n)

    println("After $n simulations:")
    println("  Staying won  $stay_wins times ($(round(100 * stay_wins / n; digits=2))%)")
    println("  Switching won $switch_wins times ($(round(100 * switch_wins / n; digits=2))%)")
end
# Run the simulation
simulate_monty_hall(100_000)
