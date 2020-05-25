import numpy as np

def henon_map(n, a, b, x0, y0):
    t = np.arange(n+1)
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0], y[0] = x0, y0
    for i in range(1, n+1):
        x[i] = 1. - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    return t, x, y

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

def logistic_map_lyapunov(n_transient, n, r, x0):
    x = x0
    for i in range(n_transient):
        x = r * x * (1. - x)
    result = np.zeros(n)
    for i in range(n):
        x = r * x * (1. - x)
        result[i] = np.log(np.abs(r-2.*r*x))
    return np.mean(result)


def tent_map(n, mu, x0):
    t = np.arange(n+1)
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(1, n+1):
        x[i] = mu * min(x[i-1], 1-x[i-1])
    return t, x

