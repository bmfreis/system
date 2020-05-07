import numpy as np

def ikeda_map(n, u, x0, y0):
    t = np.arange(n+1)
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0], y[0] = x0, y0
    for i in range(1, n+1):
	    tn   = 0.4 - 6./(1. + x[i-1]**2 + y[i-1]**2)
	    x[i] = 1. + u * (x[i-1] * np.cos(tn) - y[i-1] * np.sin(tn))
	    y[i] =      u * (x[i-1] * np.sin(tn) + y[i-1] * np.cos(tn))
    return t, x, y


def logistic_map(n, r, x0):
    t = np.arange(n+1)
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(1, n+1):
        x[i] = r * x[i-1] * (1. - x[i-1])
    return t, x
