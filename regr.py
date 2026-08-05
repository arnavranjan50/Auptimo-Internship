import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

pred_norm = np.array([
    0.46, 
    0.26, 
    0.42, 
    0.46, 
    0.44, 
    0.44,    
    0.3, 
    0.283333333, 
    0.266666667, 
    0.266666667,    
    0.3, 
    0.283333333, 
    0.183333333, 
    0.383333333, 
    0.383333333, 
    0.366666667, 
    0.383333333, 
    0.333333333, 
    0.283333333, 
    0.216666667,    
    0.04, 
    0.24, 
    0.26, 
    0.22, 
    0.2, 
    0.283333333, 
    0.3, 
    0.3, 
    0.34, 
    0.24, 
    0.24, 
    0.3, 
])

calc_norm = np.array([
    0.58, 
    0.4, 
    0.52, 
    0.54, 
    0.56, 
    0.54, 
    0.383333333, 
    0.366666667, 
    0.366666667, 
    0.35, 
    0.366666667, 
    0.366666667, 
    0.35, 
    0.466666667, 
    0.483333333, 
    0.483333333, 
    0.5, 
    0.433333333, 
    0.383333333, 
    0.316666667,    
    0.38, 
    0.34, 
    0.34, 
    0.32, 
    0.32, 
    0.383333333, 
    0.4, 
    0.366666667, 
    0.44, 
    0.36, 
    0.34, 
    0.4
])

print(len(pred_norm))
print(len(calc_norm))
X_left = pred_norm.reshape(-1,1)
y =  calc_norm
reg = LinearRegression().fit(X_left, y)
r2 = reg.score(X_left, y)
print(r2)
m = float(reg.coef_[0])
c = float(reg.intercept_)
print(f'Y = {m} x + {c}')