import time

def gradient_descent(df, starting_x=0, learning_rate=0.1, max_iter=10000, eps=1e-6):
    """
    Execute gradient descent method to find a local minimum.
    
    The iterative process stops when the maximum number of iterations
    is reached or when the two last checked points are closer than
    the epsilon value provided.
    
    Parameters
    ----------
    df : function or lambda
        A function that simulates the derivative of the function of
        which the minimum needs to be found. It has to take a variable
        as input and return a real number (f'(x))
    starting_x : int or float, default 0
        The starting guess for the location of the minimum.
    learning_rate : float, default 0.1
        The factor to reduce the impact of df(x) with.
    max_iter : int, default 10000
        The maximum number of iterations the algorithm will go through
        untill it stops.
    eps : float, default 1e-6
        The epsilon value that is used to check for convergence.
    
    Returns
    -------
    current_x : float
        The value of the local minimum.
    iteration : int
        The number of iterations the algorithm needed to converge.
    """
    iteration = 0
    current_x = starting_x
    step = 1 # a value greater than the error margin so that the first iteration happens
    while iteration < max_iter and abs(step) > eps:
        previous_x = current_x # store current x value in order to be used in the next iteration
        current_x = previous_x - learning_rate * df(previous_x)  # gradient descent
        
        iteration += 1 # iteration count
        step = previous_x - current_x # calculate step for exit condition
    
    return current_x, iteration


# Example derivative
def df(x):
    return 4*pow((x - 5), 3) + 3


# Experiments with various learning rates
lrs = [0.0001, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.5]

for lr in lrs:
    try:
        t1 = time.time()
        minimum, iterations = gradient_descent(df, learning_rate=lr, eps=1e-6)
        t2 = time.time()
    except OverflowError: # if the values become too large
        print("Learning Rate:", lr, "- The algorithm does not converge")
    else:
        print("Learning Rate:", lr, "- Minimum:", minimum, "- Iterations:", iterations, "- Time:", t2-t1)
