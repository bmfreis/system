import numpy as np
from scipy.integrate import odeint

def chua(data, t, alpha, beta, a, b):
    x, y, z = data
    f = b * x + 0.5 * (a-b) * (abs(x+1)-abs(x-1))
    dxdt = alpha * (-x + y - f) 
    dydt = x - y + z
    dzdt = -beta * y
    return dxdt, dydt, dzdt

def lorenz(data, t, sigma, rho, beta):
    x, y, z = data
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return dxdt, dydt, dzdt

def rossler(data, t, a, b, c):
    x, y, z = data
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return dxdt, dydt, dzdt

def van_der_pol(data, t, mu):
    x, y = data
    dxdt = y
    dydt = mu * (1.0 - x**2) * y - x
    return dxdt, dydt

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

def van_der_pol_system(t, dt, mu, xi, yi):
    t = np.arange(0, t+dt, dt)
    f = odeint(van_der_pol, (xi, yi), t, args=(mu,))
    x, y = f.T    
    return t, x, y
