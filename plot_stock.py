import numpy as np
import cvxopt as opt
import matplotlib.pyplot as plt
from cvxopt import blas, solvers

return_norm = np.load('data/normalized_return.npy', allow_pickle='TRUE')
cov = np.load('data/cross_correlation.npy', allow_pickle='TRUE')
return1d = np.load('data/return.npy', allow_pickle='TRUE')



#plt.plot(return_norm[0:10, :].T, alpha=.4)
#plt.xlabel('time')
#plt.ylabel('returns')
#plt.show()

def q_operator(sigma, C):
    n = len(sigma)
    Q = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            Q[i, j] = sigma[i]*sigma[j]*C[i, j]

    return Q

def opt_portfolio(returns):
    # Constraints.

    # We want w_i >= 0 for all i

    G = -opt.matrix(np.eye(n))  
    h = opt.matrix(0.0, (n ,1))
    Q = 2*opt.matrix(q_operator(sigma, cov))
    p = opt.matrix(np.zeros(n))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    A2 = opt.matrix(np.mean(returns, axis=1))
    A = A + A2.T

    N = 10
    sol = []
    for i, phi in enumerate(np.arange(0., 0.2, 0.2/N)):
        minim = opt.solvers.qp(Q.T, p, G, h, A, b + opt.matrix(phi))
        sol.append(minim['x'])
    
    return sol


n = len(return_norm[:, 0])
sigma = [np.sqrt(np.mean(return1d[i, :]**2)-np.mean(return1d[i, :])**2) for i in range(n)]

w = opt_portfolio(return_norm)
N = 10
guadagno =  np.arange(0., 0.2, 0.2/N)

risks = np.zeros(n)
for i, g in enumerate(guadagno):
    risks[i] = np.array(w[i]).ravel() @ q_operator(sigma, cov) @ np.array(w[i]).ravel()

#print(risks)
plt.plot(risks[:10], guadagno)
plt.show()
        







