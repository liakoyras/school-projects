import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

save_location = "./" # path to save the images

# A priori probability distribution
def p_theta_D0(theta):
    return (np.pi / 2) * np.sin(np.pi * theta)

toss_results = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1]) # 1: head 0: tail

thetas = np.linspace(0, 1, 5000) # different theta values for plot
fig, ax = plt.subplots(figsize=(9, 8)) #initialize plot

# calculate p(θ|D^N) using the solved recursive equation,
# with a lambda function as the nominator and the denominator
# integral calculated using scipy
for N in [1, 5, 10]: # number of samples to calculate p(θ|D^N) for
    k = sum(toss_results[0:N]) # count heads
    # calculate values for different thetas so that they can be plotted
    numerator = lambda theta: (theta**k)*(1-theta)**(N-k)*p_theta_D0(theta)
    denominator = integrate.quad(numerator, 0, 1)[0]
    y = numerator(thetas) / denominator
    legend_label = "$" + r"P(\theta | D^{" + str(N) + "})" + "$"
    plt.plot(thetas, y, label=legend_label)

ax.legend(loc='upper left')

plt.savefig(save_location + "p_theta_D.png")

print("Maximum P(theta|D10):", np.max(y), "for theta =", np.argmax(y)/5000)

