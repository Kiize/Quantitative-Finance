import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

N = 10

# Omega function

def omega2(w, C, sigma):
    Omega = 0.
    for i in range(0, N):
        for j in range(0, N):
            Omega += w[i]*w[j]*C[i, j]*sigma[i]*sigma[j]   

    return Omega

# Phi function.

def phi(w, *G):
    return w @ G

# Constraint

def g(w):
    return np.sum(w) - 1 

# Define the constraints in the form required by the minimize function.
G = np.random.random(N)

cons = ({'type': 'eq', 'fun': g},
        {'type': 'eq', 'fun': phi, 'args' : G})

# Simulation.

C = np.random.random(size=(N, N))
C = np.abs(C)
sigma = np.random.randn(N)


# Set the initial guess for the optimization
x0 = np.array(np.random.randn(N))

# Minimize the objective function subject to the constraints

result = opt.minimize(omega2, x0, constraints=cons, args=(C, sigma))

print(result)


