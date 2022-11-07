import numpy as np
import pandas as pd
import itertools as it

import lib

# import data
data = pd.read_csv('data.txt', delimiter=" ", header=None, index_col=0,
                    names =["w1-x1", "w1-x2", "w1-x3",
                            "w2-x1", "w2-x2", "w2-x3",
                            "w3-x1", "w3-x2", "w3-x3"])

data = data.astype('float16')
"""
Classify in w1 and w2, using only feature x1
"""
p1 = p2 = 0.5

# Find normal distribution parameters
mean_w1_x1 = np.mean(data["w1-x1"])
mean_w2_x1 = np.mean(data["w2-x1"])

cov_w1_x1 = np.cov(data["w1-x1"], rowvar=False)
cov_w2_x1 = np.cov(data["w2-x1"], rowvar=False)

# Find point where the discriminant functions are equal by searching
# the sample range for the points that minimize their difference
for n in np.arange(-10, 10, 0.00001):
    difference = lib.discriminant(n, 1, mean_w1_x1, cov_w1_x1, p1) - \
                 lib.discriminant(n, 1, mean_w2_x1, cov_w2_x1, p2)

    if abs(difference) < 0.000001:
        print("Boundary for x =", n, "with error margin", difference)

# Find classification error.
# Errors occur when g_1(x) < g_2(x) for samples of class w1 and 
# g_1(x) > g_2(x) for samples of class w2, since g_i(x) must be
# biggest for the class i that the sample belongs to.
x1_errors = 0
for w1_x1, w2_x1 in zip(data["w1-x1"], data["w2-x1"]):
    g11 = lib.discriminant(w1_x1, 1, mean_w1_x1, cov_w1_x1, p1)
    g21 = lib.discriminant(w1_x1, 1, mean_w2_x1, cov_w2_x1, p2)
    if g11 < g21:
        x1_errors += 1
    
    g12 = lib.discriminant(w2_x1, 1, mean_w1_x1, cov_w1_x1, p1)
    g22 = lib.discriminant(w2_x1, 1, mean_w2_x1, cov_w2_x1, p2)
    if g12 > g22:
        x1_errors += 1
    
total_error_x1 = x1_errors/20
print("Classification error when using only x1:      ", total_error_x1)

"""
Classify in w1 and w2, using features x1, x2
"""
p1 = p2 = 0.5

mean_w1_x12 = np.array(list(np.mean(data[["w1-x1", "w1-x2"]], axis=0)))
mean_w2_x12 = np.array(list(np.mean(data[["w2-x1", "w2-x2"]], axis=0)))

cov_w1_x12 = np.array(np.cov(data[["w1-x1", "w1-x2"]], rowvar=False))
cov_w2_x12 = np.array(np.cov(data[["w2-x1", "w2-x2"]], rowvar=False))

x1x2_errors = 0
for w1_x1, w1_x2, w2_x1, w2_x2 in zip(data["w1-x1"], data["w1-x2"], data["w2-x1"], data["w2-x2"]):
    point_1 = np.array([w1_x1, w1_x2])
    g11 = lib.discriminant(point_1, 2, mean_w1_x12, cov_w1_x12, p1)
    g21 = lib.discriminant(point_1, 2, mean_w2_x12, cov_w2_x12, p2)
    if g11 < g21:
        x1x2_errors += 1
    
    point_2 = np.array([w2_x1, w2_x2])
    g12 = lib.discriminant(point_2, 2, mean_w1_x12, cov_w1_x12, p1)
    g22 = lib.discriminant(point_2, 2, mean_w2_x12, cov_w2_x12, p2)
    if g12 > g22:
        x1x2_errors += 1

total_error_x1x2 = x1x2_errors/20
print("Classification error when using x1 and x2:    ", total_error_x1x2)


"""
Classify in w1 and w2, using features x1, x2, x3
"""
p1 = p2 = 0.5

mean_w1_x123 = np.array(list(np.mean(data[["w1-x1", "w1-x2", "w1-x3"]], axis=0)))
mean_w2_x123 = np.array(list(np.mean(data[["w2-x1", "w2-x2", "w2-x3"]], axis=0)))

cov_w1_x123 = np.array(np.cov(data[["w1-x1", "w1-x2", "w1-x3"]], rowvar=False))
cov_w2_x123 = np.array(np.cov(data[["w2-x1", "w2-x2", "w2-x3"]], rowvar=False))

x1x2x3_errors = 0
for w1_x1, w1_x2, w1_x3, w2_x1, w2_x2, w2_x3 in zip(data["w1-x1"], data["w1-x2"], data["w1-x3"], data["w2-x1"], data["w2-x2"], data["w2-x3"]):
    point_1 = np.array([w1_x1, w1_x2, w1_x3])
    g11 = lib.discriminant(point_1, 3, mean_w1_x123, cov_w1_x123, p1)
    g21 = lib.discriminant(point_1, 3, mean_w2_x123, cov_w2_x123, p2)
    if g11 < g21:
        x1x2x3_errors += 1

    point_2 = np.array([w2_x1, w2_x2, w2_x3])
    g12 = lib.discriminant(point_2, 3, mean_w1_x123, cov_w1_x123, p1)
    g22 = lib.discriminant(point_2, 3, mean_w2_x123, cov_w2_x123, p2)
    if g12 > g22:
        x1x2x3_errors += 1

total_error_x1x2x3 = x1x2x3_errors/20
print("Classification error when using x1, x2 and x3:", total_error_x1x2x3)


"""
Analytically calculate the discriminant functions for all 3 classes
using all features x1, x2, x3 (symbolic calculation).
"""
from sympy import *
init_printing(use_unicode=True)

# define sympy objects
x1, x2, x3 = symbols('x1 x2 x3')
x = Matrix([x1, x2, x3])

# define parameters
p1 = 0.8
p2 = p3 = 0.1

m1 = Matrix(np.array(list(np.mean(data[["w1-x1", "w1-x2", "w1-x3"]], axis=0))).round(4))
m2 = Matrix(np.array(list(np.mean(data[["w2-x1", "w2-x2", "w2-x3"]], axis=0))).round(4))
m3 = Matrix(np.array(list(np.mean(data[["w3-x1", "w3-x2", "w3-x3"]], axis=0))).round(4))

cov1 = Matrix(np.array(np.cov(data[["w1-x1", "w1-x2", "w1-x3"]], rowvar=False)).round(4))
cov2 = Matrix(np.array(np.cov(data[["w2-x1", "w2-x2", "w2-x3"]], rowvar=False)).round(4))
cov3 = Matrix(np.array(np.cov(data[["w3-x1", "w3-x2", "w3-x3"]], rowvar=False)).round(4))

def symbolic_discriminant_3d(x, m, c, p):        
    print(simplify(-0.5 * ((((x-m).T) @ c**-1) @ (x-m))[0] - 1.5 * log(2*pi) - 0.5 * log(Abs(c.det())) + log(p)))

print("g1 = ")
symbolic_discriminant_3d(x, m1, cov1, p1)
print("g2 = ")
symbolic_discriminant_3d(x, m2, cov2, p2)
print("g3 = ")
symbolic_discriminant_3d(x, m3, cov3, p3)

