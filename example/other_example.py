import numpy as np
import time
import pso_bacco


def f(param, a = 0.05, b = 0.05):
    time.sleep(0.01) # To test MPI performance only
    x,y = param
    return (x-a)**2+(b-y)**2


bounds = np.array([[0,2],[0,2]])
params = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

pso = pso_bacco.pso_bacco.global_PSO(bounds = bounds, initial_pos = 'random',  params = params, npoints=4)
pso.run(f, func_argv = (0.05,0.05), niter = 50, backup_name = 'backup.hdf5')
pso.run(f, func_argv = (0.05,0.05), niter = 10, backup_name = 'backup.hdf5')
pso2 = pso_bacco.pso_bacco.global_PSO(bounds = bounds, params = params, npoints=4, backup_name='backup.hdf5')
pso2.run(f, func_argv = (0.05,0.05), niter = 40, backup_name = 'backup.hdf5')
pso2.plot_history('pso_history.png')

# > mpiexec -n 4 python other_example.py
