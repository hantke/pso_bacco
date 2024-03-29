import numpy as np
import time
import pso_bacco

def f(param, a = 0.05, b = 0.05):
    time.sleep(0.01) # To test MPI performance only
    return (param[:,0]-a)**2+(param[:,1]-b)**2

bounds = np.array([[0,2],[0,2]])
params = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
pso = pso_bacco.pso_bacco.global_PSO(bounds = bounds, params = params, npoints = 10)
pso.run(f, func_argv = (0.05,0.05), niter = 100,backup_name = 'backup.h5', vectorize=True)
#pso.plot_history()

# > python example.py 
