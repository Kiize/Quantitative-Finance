"""
Code to test and understand how scipy.optimize.minimize works
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


#=======================================================================
# Unconstrained minimization test in 1D
#=======================================================================

def f(x):
    ''' Function to minimize.
    '''
    return x**2 + np.sin(x) - np.cosh(np.sqrt(np.abs(x)))

# Find the minimum; x0 is an init point
result = opt.minimize(f, x0=-2)
print(result)

# Plot result
plt.figure(1)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Unconstrained minimization test in 1D")
x_plot = np.linspace(-1, 1, 1000)
plt.plot(x_plot, f(x_plot), 'b', label="function")
plt.plot(result.x, result.fun, "ro", label="minimum")
plt.legend(loc='best')
plt.grid()
plt.show()


#=======================================================================
# Constrained minimization test in 2D
#=======================================================================


def f(vec, param=1):
    ''' Function to minimize.
    '''
    x, y = vec
    return param*x + y + x**2

def g(vec):
    ''' Constraint; write as g(x) = 0
    '''
    x, y = vec
    return x + y - 1 

# Define the constraints in the form required by the minimize function.
cons = ({'type': 'eq', 'fun': g})

# Set the initial guess for the optimization
x0 = np.array([3, -5])

# Minimize the objective function subject to the constraints
param = -10
result = opt.minimize(f, x0, constraints=cons, args=(param))
print(result)

# Plot
x_plot = np.linspace(-2*abs(param), 2*abs(param), 100)
X, Y = np.meshgrid(x_plot, x_plot)

fig = plt.figure(2)
ax  = fig.add_subplot(projection='3d')
ax.set_title("Constrained minimization test in 2D")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_zlabel("F(x,y)")

#plot
surf=ax.plot_surface(X, Y, f([X, Y], param), color='blue', alpha=0.4, label="function")
ax.plot(*result.x, f(result.x, param), "ko", label="minimum")
ax.plot(x_plot, 1 - x_plot, f([x_plot, 1 - x_plot], param), color='orange', label="Constraint")

#legend
surf._facecolors2d = surf._facecolor3d # to avoid problem
surf._edgecolors2d = surf._edgecolor3d # with the legend
ax.legend()
plt.show()
