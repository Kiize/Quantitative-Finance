import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Test 1D libero.

# Function to minimize.
"""
def f(x):
    return x**2 + np.sin(x) - np.cosh(np.sqrt(np.abs(x)))

# Find the minimum.

result = opt.minimize(f, x0 = -2)

x_plot = np.linspace(-1, 1, 1000)
plt.plot(x_plot, f(x_plot))
plt.plot(result.x, result.fun, marker = 'o')
print(result)
plt.show()
"""

# Test 2D con constraint.

# Function to minimize.

def f(vec, param = 1):
    x, y = vec
    return param*x + y + x**2

# Constraint.

def g(vec):
    x, y = vec
    return x + y - 1 


# Define the constraints in the form required by the minimize function.

cons = ({'type': 'eq', 'fun': g})

# Set the initial guess for the optimization
x0 = np.array([3, -5])

# Minimize the objective function subject to the constraints
param = 10
result = opt.minimize(f, x0, constraints=cons, args=(-param))

# Plot.

x_plot = np.linspace(-2*param, 2*param, 100)
X, Y = np.meshgrid(x_plot, x_plot)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, f([X, Y]), alpha = 0.4)
ax.plot(*result.x, f(result.x), marker = 'o')
ax.plot(x_plot, 1 - x_plot, f([x_plot, 1 - x_plot]), color = 'orange')
plt.show()




