import time

def newton_method(df, ddf, starting_x=0, max_iter=10000, eps=1e-6):
    """
    Execute Newton-Raphson method to find a local minimum.
    
    The iterative process stops when the maximum number of iterations
    is reached or when the two last checked points are closer than
    the epsilon value provided.
    
    Parameters
    ----------
    df : function or lambda
        A function that simulates the derivative of the function of
        which the minimum needs to be found. It has to take a variable
        as input and return a real number (f'(x))
    ddf : function or lambda
        A function that simulates the second derivative of the function
        of which the minimum needs to be found. It has to take a variable
        as input and return a real number (f''(x))
    starting_x : int or float, default 0
        The starting guess for the location of the minimum.
    max_iter : int, default 1000
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
        current_x = previous_x - (df(previous_x) / ddf(previous_x))  # newton's method
        
        iteration += 1 # iteration count
        step = previous_x - current_x # calculate step for exit condition
    
    return current_x, iteration


# Example derivative
def df(x):
    return 4*pow((x - 5), 3) + 3

# Example second derivative
def ddf(x):
    return 12*pow((x - 5), 2)



t1 = time.time()
minimum, iterations = newton_method(df, ddf)
t2 = time.time()

print("Minimum:", minimum, "- Iterations:", iterations, "- Time:", t2-t1)
