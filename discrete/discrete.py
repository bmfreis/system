import numpy as np

def ikeda_map(t, u, xn, yn):
    x = []
    y = []
    t = np.arange(0, t+1, 1)
    for i in t:
	    tn = 0.4 - 6./(1. + xn**2 + yn**2)
	    xn1 = 1. + u * (xn * np.cos(tn) - yn * np.sin(tn))
	    yn1 = u * (xn * np.sin(tn) + yn * np.cos(tn))
	    x.append(xn1)
	    y.append(yn1)
	    xn = xn1
	    yn = yn1
    return t, x, y
