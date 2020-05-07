import numpy as np
from scipy.integrate import odeint

def chua(data, t, alpha, beta, a, b):
    x, y, z = data
    f = b * x + 0.5 * (a-b) * (abs(x+1)-abs(x-1))
    x1 = alpha * (-x + y - f) 
    y1 = x - y + z
    z1 = -beta * y
    return x1, y1, z1

def lorenz(data, t, sigma, rho, beta):
    x, y, z = data
    x1 = sigma * (y - x)
    y1 = x * (rho - z) - y
    z1 = x * y - beta * z
    return x1, y1, z1

def rossler(data, t, a, b, c):
    x, y, z = data
    x1 = -y - z
    y1 = x + a * y
    z1 = b + z * (x - c)
    return x1, y1, z1

def chua_system(t, dt, alpha, beta, a, b, xi, yi, zi):
    t = np.arange(0, t+dt, dt)
    f = odeint(chua, (xi, yi, zi), t, args=(alpha, beta, a, b))
    x, y, z = f.T    
    return t, x, y, z

def lorenz_system(t, dt, sigma, rho, beta, xi, yi, zi):
    t = np.arange(0, t+dt, dt)
    f = odeint(lorenz, (xi, yi, zi), t, args=(sigma, rho, beta))
    x, y, z = f.T    
    return t, x, y, z

def rossler_system(t, dt, a, b, c, xi, yi, zi):
    t = np.arange(0, t+dt, dt)
    f = odeint(rossler, (xi, yi, zi), t, args=(a, b, c))
    x, y, z = f.T    
    return t, x, y, z

