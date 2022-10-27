import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.stats import kurtosis

save_location = "./"
np.random.seed(42) # For reproducibility

# Simulate a number of rolls with a six-sided die
def die_rolls(num_rolls):
    rolls = np.random.randint(low=1,high=7, size=num_rolls)
    return rolls


# Plot histograms of 20, 200 and 1500 rolls that approximate the
# Probability Density Function
rolls_20 = die_rolls(20)
rolls_200 = die_rolls(200)
rolls_1500 = die_rolls(1500)

fig, axes = plt.subplots(1,3, figsize=(15, 5))
axes[0].bar([1, 2, 3, 4, 5, 6], np.bincount(rolls_20)[1:])
axes[0].set_title("20 Rolls", fontsize = 15)
axes[1].bar([1, 2, 3, 4, 5, 6], np.bincount(rolls_200)[1:])
axes[1].set_title("200 Rolls", fontsize = 15)
axes[2].bar([1, 2, 3, 4, 5, 6], np.bincount(rolls_1500)[1:])
axes[2].set_title("1500 Rolls", fontsize = 15)

plt.savefig(save_location + "histograms 1 dice.png")



# Calculate the mean, variance, skewness and kurtosis for the following
# numbers of die rolls and display them along with the difference from the
# theoretical values.
roll_nums = [10, 25, 50, 100,200, 500, 1000]
experiments = [die_rolls(num) for num in roll_nums]

print("====================== MEAN ======================")
for exp, num in zip(experiments, roll_nums):
    m = np.mean(exp)
    print(num, "rolls: Mean", format(m, '.6f'), "- Difference", format(abs(3.5 - m),'.4f'))

print("==================== VARIANCE ====================")
for exp, num in zip(experiments, roll_nums):
    v = np.var(exp)
    print(num, "rolls: Variance", format(v, '.6f'), "- Difference", format(abs(2.91667 - v),'.4f'))

print("==================== SKEWNESS ====================")
for exp, num in zip(experiments, roll_nums):
    s = skew(exp)
    print(num, "rolls: Skewness", format(s, '.6f'), "- Difference", format(abs(0 - s),'.4f'))    
    
print("==================== KURTOSIS ====================")
for exp, num in zip(experiments, roll_nums):
    # fisher=false to calculate the kurtosis without normalizing (-3 so that
    # Normal Distribution has kurtosis of 0)
    k = kurtosis(exp, fisher=False)
    print(num, "rolls: Kurtosis", format(k, '.6f'), "- Difference", format(abs(1.73143 - k),'.4f'))

print("==================================================")



# Plot the mean value graph to check the distribution for
# wide-sense stationarity
means = [np.mean(exp) for exp in experiments]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(roll_nums, means, 'o-',linewidth=2, markersize=8)
ax.set_ylabel('Mean Value')
ax.set_xlabel('Number of Rolls')

plt.savefig(save_location + "mean values 1 dice.png")
