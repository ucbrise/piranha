                                                                                                
# ------- Search for degree d curve approximation --------
# Code will find the best least square fit of a given degree
# and then refine it to get the best curve minimizing L1 loss

import math
import numpy as np
from scipy.optimize import minimize

###################################### MODIFIY ######################################
start = 0.5 
stop = 1.01
step = 0.01
degree = 2

def TARGET(x):
        return 1 / (1 + math.exp(-x)) 

################################# NOT TO BE MODIFIED ################################



############################### Least Squares poly fit ##############################
VECTORIZED_TARGET = np.vectorize(TARGET)

x = np.arange(start, stop, step)
y = VECTORIZED_TARGET(x)
z = np.polyfit(x, y, degree) # Produces array with z[0] as quadratic coefficient 

#################################### L1 poly fit ####################################

beta_init = z   # Intialize the L1 fit from the least squares fit. Hopefully its good
def objective_function(beta, X, Y):
        evaluations = np.polyval(beta, X)
        return abs(max(evaluations - Y, key=abs))

result = minimize(objective_function, beta_init, args=(x,y),
                  method='BFGS', options={'maxiter': 500})

################################ Print polynomials ##################################

def print_poly(z):
    x = z[::-1]
    output = ""
    for i in range(len(x)):
        output+= '{:.6f}'.format(x[i])+"x^"+str(i)+" + "
    return output[:-2]


print("-----------------------------------------------------------------------------")
print("Least squares polynomial fit: ", print_poly(z))
print("-----------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------")
print("Max L1 loss polynomial fit:  ", print_poly(result.x))
print("-----------------------------------------------------------------------------")

print("Max L1 error: ", objective_function(z, x, y), "    (least squares fit)")
print("Max L1 error: ", objective_function(result.x, x, y), "    (L1 fit)")

#print(np.polyval(z, x))
#print(y)
#print(np.polyval(result.x, x))
  
