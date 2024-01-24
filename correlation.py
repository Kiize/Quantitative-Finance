"""
Code to compute the cross-correlation matrix
"""

import time
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import utilities as ut

start = time.time()

start_story = '2020-01-01' # Start date for stock histories
end_story   = '2023-12-31' # End   date for stock histories

# Read SP500
print('read SP500 history')
SP500 = yf.download('^GSPC', start=start_story, end=end_story, interval='1d', progress=False)
L = len(SP500)

# Read the data created via: build_datset.py
history = np.load(r"data/dataset.npy",allow_pickle='TRUE').item()

# Some titles may have been 'born' during the period of time
# considered and are therefore excluded from the analysis
index = [] # list that will contain the titles to be analyzed

for i, ticker in enumerate(history.keys()):
    
    open_t = history[ticker]['Open']
    
    # If the story is of a different length than that of the
    # SP500 (in principle it can only be shorter)
    # I exclude the title from the analysis
    if len(open_t) != L:
        print(f'The history of {ticker} is too short, it will be excluded')
    else:
        index.append(ticker) # I keep the remaining titles


N = len(index)

print(f'Number of index: {N}, length of time intervall: {L}')

# Some important matrix
opening  = np.zeros((N, L))
closure  = np.zeros((N, L))
return1d = np.zeros((N, L))
retunorm = np.zeros((N, L))
crosscor = np.zeros((N, N))

# matrix for shuffle
retshuff = np.zeros((N, L))
rtnshuff = np.zeros((N, L))
cscshuff = np.zeros((N, N))



for i, ticker in enumerate(index):

    # Compute the normalized return to 1 day
    opening[i, :]  = history[ticker]['Open']
    closure[i, :]  = history[ticker]['Close']
    return1d[i, :] = np.log(closure[i,:]) - np.log(opening[i,:])
    
    sigma = np.sqrt(np.mean(return1d[i, :]**2)-np.mean(return1d[i, :])**2)
    retunorm[i, :] = (return1d[i, :] - np.mean(return1d[i, :]))/sigma

    # Like befor but with a shuffle to destroy correlation
    retshuff[i, :] = np.random.choice(return1d[i, :], size=L, replace=False)
    sigma = np.sqrt(np.mean(retshuff[i, :]**2)-np.mean(retshuff[i, :])**2)
    rtnshuff[i, :] = (retshuff[i, :] - np.mean(retshuff[i, :]))/sigma


# Compute the cross correlation matrix
for i in range(N):
    for j in range(N):
        crosscor[i, j] = ut.corr(retunorm[i, :], retunorm[j, :])
        cscshuff[i, j] = ut.corr(rtnshuff[i, :], rtnshuff[j, :])


np.save("data/cross_correlation.npy", crosscor)
np.save("data/normalized_return.npy", retunorm)
np.save("data/indici.npy", index)

mins = (time.time()-start)//60
sec  = (time.time()-start) % 60

print(f"Elapsed time: {mins} min {sec:.2f} sec")

#==============================================================================
# Some pretty plot
#==============================================================================

sup, pdf = ut.dens_prob(L, N)

plt.figure(1)
x = np.reshape(crosscor, N*N)
plt.title('Distribution of cross-correlation coefficients',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
#plt.yscale('log')
plt.grid()
plt.hist(x, int(np.sqrt(N*N-1)), density=True)


plt.figure(2)
y = np.linalg.eigvalsh(crosscor)
plt.title('Distribution of the eigenvalues of $C_{ij}$',fontsize=15)
plt.xlabel('$\lambda$ of $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.grid()
plt.yscale('log')
plt.plot(sup, pdf, 'k')
plt.hist(y, N, density=True)


plt.figure(3)
x = np.reshape(cscshuff, N*N)
plt.title('Distribution of cross-correlation coefficients\n after shuffle',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
plt.hist(x, int(np.sqrt(N*N-1)), density=True)
#plt.yscale('log')
plt.grid()


plt.figure(4)
y = np.linalg.eigvalsh(cscshuff)
plt.title('Distribution of the eigenvalues of $C_{ij}$ after shuffle',fontsize=15)
plt.xlabel('$\lambda$ of $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.hist(y, int(np.sqrt(N-1)), density=True)
plt.plot(sup, pdf, 'k')
plt.grid()

plt.show()
