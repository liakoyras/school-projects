import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import skew
from scipy.stats import kurtosis

save_location = "./"
np.random.seed(42) # For reproducibility

# Simulate a number of rolls with a six-sided die
def die_rolls(num_rolls):
    rolls = np.random.randint(low=1,high=7, size=num_rolls)
    return rolls


# Plot histogram of 1000 rolls from two dice
z1_1000 = die_rolls(1000)
z2_1000 = die_rolls(1000)

# calculate histogram and bin edges
hist, xedges, yedges = np.histogram2d(z1_1000, z2_1000, bins=6, range=[[0.5, 6.5], [0.5, 6.5]])

# calculate coordinates of bars
z1_grid, z2_grid = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25,)
z1, z2 = z1_grid.ravel(), z2_grid.ravel()
top = hist.ravel()
bottom = np.zeros_like(top)
width = depth = 0.5

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
ax.bar3d(z1, z2, bottom, width, depth, top, shade=True)
ax.set_xlabel('Z1 Results')
ax.set_ylabel('Z2 Results')
ax.set_zlabel('Number of Times')

plt.savefig(save_location + "histogram 2 dice.png")



# Plot histogram of the sum of 1000 rolls by two dice
y_1000 = z1_1000 + z2_1000

fig, ax = plt.subplots(figsize=(8, 8))
ax.bar([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], np.bincount(y_1000)[2:])
ax.set_title("1000 Rolls with two dice", fontsize = 15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig(save_location + "histogram 2 dice sum.png")
