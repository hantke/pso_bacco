#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the 'PSO' class and helper functions

"""

import numpy as np
import copy
import os
from .utils import latinhypercube
class global_PSO():
    def __init__(self,
                 bounds = None,
                 params = None,
                 npoints = 10,
                 initial_pos = 'latin_hypercube',
                 initial_vel = 'random',
                 max_speed   = 0.5,
                 par_names = None,
                 backup_name = None,
                 verbose = True):
        """
        Global PSO methid

        Args:
            bounds (array, optional): Boundaries where the PSO is run. Defaults to None.
            params (dictionary, optional): main parameter of the PSO:
                                            c1: cognitive parameter. How attracted the particle is to it best personal value.
                                            c2: social parameter. How attracted the particle is to the best global value.
                                            w:  Inertia parameter.
                                            Defaults to None.
            npoints (int, optional): Number of points of the PSO. Defaults to 10.
            niter (int, optional): Number of steps of the PSO. Defaults to 100.
            initial_pos (array, optional): Initial position of the particles. Options are 'random',
            'latin_hypercube', or an array with the positions. Defaults to 'latin_hypercube'.
            initial_vel (array, optional): Initial position of the particles. Options are 'random',
            'latin_hypercube', or an array with the positions. Defaults to 'random'.
            max_speed (float, optional): maximum speed per axis: 1 means 1 box lenght in that axis. Defaults to 0.5.
            backup_name (str optional): Name of the file to initialisate the PSO. Defaults to None.
        """

        from mpi4py import MPI
        self.mpi  = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.verbose = verbose
        
        if backup_name is not None:
            if os.path.isfile(backup_name):
                self.load(backup_name)
                return
            elif verbose: print('Backup file {} does not exist.'.format(backup_name))
                
        self.ndim    = len(bounds)
        self.npoints = npoints
        self.swarm   = self.init_swarm()
        self.bounds  = bounds
        self.par_names = par_names

        if max_speed is None:
            #The max speed should not be more than 1 box length
            self.max_speed = bounds[:,1]-bounds[:,0]
        else:
            #The max speed is on unites of the boundaries
            self.max_speed = (bounds[:,1]-bounds[:,0])*max_speed

        self.params = params if params is not None else {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        self.init_particles(initial_pos = initial_pos, initial_vel = initial_vel)


    def save(self, filename):
        """
        Save main information of the PSO

        Args:
            filename (str): Name of the saving file
        """
        if self.rank != 0: return
        import deepdish as dd
        d                        = {}
        d['swarm']               = self.swarm
        d['header']              = {}
        d['header']['ndim']      = self.ndim
        d['header']['npoints']   = self.npoints
        d['header']['bounds']    = self.bounds
        d['header']['max_speed'] = self.max_speed
        d['header']['params']    = self.params
        d['header']['par_names']    = self.par_names
        dd.io.save(filename, d)
        if self.verbose: print('Saving a backup in {}.'.format(filename))

    def load(self, filename):
        """
        Load information of the PSO

        Args:
            filename (str): Name of the loading file
        """
        import deepdish as dd
        if self.rank == 0:
            d = dd.io.load(filename)
            if self.verbose: print('Loading a backup in {}.'.format(filename))

        swarm     = d['swarm'] if self.rank == 0 else None
        ndim      = d['header']['ndim'] if self.rank == 0 else None
        npoints   = d['header']['npoints'] if self.rank == 0 else None
        bounds    = d['header']['bounds'] if self.rank == 0 else None
        max_speed = d['header']['max_speed'] if self.rank == 0 else None
        params    = d['header']['params'] if self.rank == 0 else None
        par_names = d['header']['par_names'] if self.rank == 0 else None

        self.swarm     = self.comm.bcast(swarm, root=0)
        self.ndim      = self.comm.bcast(ndim, root=0)
        self.npoints   = self.comm.bcast(npoints, root=0)
        self.bounds    = self.comm.bcast(bounds, root=0)
        self.max_speed = self.comm.bcast(max_speed, root=0)
        self.params    = self.comm.bcast(params, root=0)
        self.par_names = self.comm.bcast(par_names, root=0)


    def init_swarm(self):
        """
        Init. the swarm information

        Returns:
            dictionary: Swarm information
        """
        swarm = {}
        swarm['pos'] = np.zeros((self.npoints, self.ndim))
        swarm['vel'] = np.zeros((self.npoints, self.ndim))
        swarm['val'] = np.zeros((self.npoints))

        swarm['pos_histo'] = np.zeros((1,self.npoints, self.ndim))
        swarm['val_histo'] = np.zeros(self.npoints)
        swarm['cost_histo'] = []

        swarm['pos_blocal']  = np.zeros((self.npoints, self.ndim))
        swarm['pos_bglobal'] = np.zeros((self.ndim))
        swarm['val_blocal']  = np.zeros((self.npoints))
        swarm['val_bglobal'] = 0
        return swarm

    def update_velocity(self):
        """
        Update velocity of the particles
        """
        #Based on the function from pyswarms
        #https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/backend/operators.py
        def _apply_clamp(self):
            for i in range(self.ndim):
                self.swarm['vel'][:,i][self.swarm['vel'][:,i] >  self.max_speed[i]] =  self.max_speed[i]
                self.swarm['vel'][:,i][self.swarm['vel'][:,i] < -self.max_speed[i]] = -self.max_speed[i]

        c1 = self.params["c1"]
        c2 = self.params["c2"]
        w  = self.params["w"]

        for i in range(self.ndim):
            cognitive = c1 * np.random.uniform(0, 1, self.npoints) * (self.swarm['pos_blocal'][:,i] - self.swarm['pos'][:,i])
            social    = c2 * np.random.uniform(0, 1, self.npoints) * (self.swarm['pos_bglobal'][i]  - self.swarm['pos'][:,i])
            self.swarm['vel'][:,i] = (w * self.swarm['vel'][:,i]) + cognitive + social
            self.swarm['vel'][:,i][self.swarm['vel'][:,i] >  self.max_speed[i]] =  self.max_speed[i]
            self.swarm['vel'][:,i][self.swarm['vel'][:,i] < -self.max_speed[i]] = -self.max_speed[i]

    def update_position(self):
        """
        Update position of the particles
        """
        self.swarm['pos'] += self.swarm['vel']

    def correct_bound(self, method, reflect_param = 0.5):
        """
        Correct boundaries

        Args:
            method (str): Type of correction:
                          Reflect: reflect the particle and inverse it velocity
                          Border:  locate the particle in the closest border with velocity 0 in that axis
            reflect_param (float, optional): reduce the velocity when reflecting by this factor. Defaults to 0.5.
        """
        if method == 'reflect' or method == 'Reflect':
            # You see a wall, you go against it, you 'bounce', reflect_param slower
            for i in range(self.ndim):
                count = 0
                any = True
                #Maybe if the particle is too far away from the bounds,
                #the correction need to be aplied again.
                while any:
                    _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                    _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                    if np.any(_mask1):
                        self.swarm['pos'][:,i][_mask1] += 2*(self.bounds[i][0]-self.swarm['pos'][:,i][_mask1])
                        self.swarm['vel'][:,i][_mask1] *= -reflect_param
                    if np.any(_mask2):
                        self.swarm['pos'][:,i][_mask2] -= 2*(self.swarm['pos'][:,i][_mask2]-self.bounds[i][1])
                        self.swarm['vel'][:,i][_mask2] *= -reflect_param
                    any = np.any(_mask1) | np.any(_mask2)
                    count +=1
                    if any and (count > 10):
                        if self.verbose: print('WARNING! Particle corrected more than 10 times, probably a bug. seting it in a random position with no velocity')
                        _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                        _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                        self.swarm['pos'][:,i][_mask1] = self.bounds[i][0] + np.random.random()*(self.bounds[i][1]-self.bounds[i][0])
                        self.swarm['vel'][:,i][_mask1] = 0
                        self.swarm['pos'][:,i][_mask2] = self.bounds[i][0] + np.random.random()*(self.bounds[i][1]-self.bounds[i][0])
                        self.swarm['vel'][:,i][_mask2] = 0
                        any=False

        if method == 'border' or method == 'Border':
            # Stay in the edge you cross, no velocity
            for i in range(self.ndim):
                _mask1 = self.swarm['pos'][:,i] < self.bounds[i][0]
                _mask2 = self.swarm['pos'][:,i] > self.bounds[i][1]
                self.swarm['pos'][:,i][_mask1] = self.bounds[i][0]
                self.swarm['vel'][:,i][_mask1] = 0
                self.swarm['pos'][:,i][_mask2] = self.bounds[i][1]
                self.swarm['vel'][:,i][_mask2] = 0

        #Other methods can be implemented if needed

    def init_particles(self, initial_pos = None, initial_vel = None):
        """
        Initial location and velocities of the particles

        Args:
            initial_pos (array or None, optional): Initial position of the particles. If None or 'random' they are located randomly inside the bondaries. If 'latin_hypercube' as a latin hypercube. Defaults to None.
            initial_vel (array or None, optional): Initial velocities of the particles. If None or 'random' they are located randomly between 0 and the maximum velocity.  Another option is 'latin_hypercube'. Defaults to None.
        """
        if (initial_pos is None) | (initial_pos == 'random'):
            self.swarm['pos'] = self.bounds[:,0]+(self.bounds[:,1]-self.bounds[:,0])*np.random.random((self.npoints, self.ndim))
        elif initial_pos == 'latin_hypercube':
            self.swarm['pos'] = self.bounds[:,0]+(self.bounds[:,1]-self.bounds[:,0])* latinhypercube(self.ndim, self.npoints, spread=False)
        else:
            self.swarm['pos'] = initial_pos

        if (initial_vel is None) | (initial_vel == 'random'):
            self.swarm['vel'] = (1-2*np.random.random((self.npoints, self.ndim)))*self.max_speed
        elif initial_pos == 'latin_hypercube':
            self.swarm['vel'] = (1-2*latinhypercube(self.ndim, self.npoints, spread=False))*self.max_speed
        else:
            self.swarm['vel'] = initial_vel

    def update_best_pos(self):
        """
        Update the best local and global position of the particles
        """
        if self.swarm['pos_histo'].shape[0] == 1:
            self.swarm['pos_histo'] =  copy.deepcopy(np.reshape(self.swarm['pos'],(1,self.npoints,self.ndim)))
            self.swarm['val_histo'] = copy.deepcopy(np.reshape(self.swarm['val'],(1,self.npoints) ))

        self.swarm['pos_histo'] = np.vstack((self.swarm['pos_histo'], np.reshape(self.swarm['pos'],(1,self.npoints,self.ndim)) ))
        self.swarm['val_histo'] = np.vstack((self.swarm['val_histo'], self.swarm['val']))

        for i in range(self.npoints):
            _ival_local   = np.argmin(self.swarm['val_histo'][:,i],axis=0)
            self.swarm['pos_blocal'][i]  = self.swarm['pos_histo'][_ival_local][i]
            self.swarm['val_blocal'][i]  = self.swarm['val_histo'][_ival_local][i]
        _ival_global              = np.argmin(self.swarm['val_blocal'])
        self.swarm['pos_bglobal'] = self.swarm['pos_blocal'][_ival_global]
        self.swarm['val_bglobal'] = self.swarm['val_blocal'][_ival_global]
        self.swarm['cost_histo'].append(self.swarm['val_bglobal'])

    def print_info(self,i, niter):
        """
        Print summary information of the status of the PSO

        Args:
            i (int): Step of the PSO cicle
            niter (int): Number of steps of the PSO run.
        """
        print('{}/ {}    lower cost: {} best pos: {}'.format(i+1,niter, self.swarm['val_bglobal'],self.swarm['pos_bglobal']))

    def plot_history(self, filename = None, show=True):
        """
        Make an plot of the cost history of the particles

        Args:
            filename (str, optional): Name of the plot file. Defaults to None.
            show (bool, optional): Wheter to show the plot or not. Default to True.
        """
        if self.rank != 0: return
        import matplotlib.pyplot as plt
        plt.semilogy(range(len(self.swarm['val_histo'])),self.swarm['val_histo'])
        cost = self.swarm['cost_histo']
        plt.semilogy(range(len(cost)),cost,c='k',ls='--')
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()

    def mpi_scatter_swarm(self):
        """
        Share the position of the particles from CPU 0 to the rest of the CPUs

        Returns:
            array: position of the particles for each CPU
        """
        _pos = self.swarm['pos'] if self.rank == 0 else None
        _pos = self.comm.bcast(_pos, root=0)
        return _pos

    def run(self, f, func_argv = (), niter=100, bound_correct_method = 'reflect', reflect_param = 0.8,
            backup_name=None, backup_frequency=10, verbose = None):
        """
        Run the PSO.

        Args:
            f (function): Function to evaluate the PSO
            func_argv (tuple, optional): aditional argument of function f. Defaults to ().
            bound_correct_method (str, optional): Method to correct boundaries. Defaults to 'reflect'.
            niter (int, optional): Number of steps of the PSO. Defaults to 100.
            reflect_param (float, optional): Speed decrease factor when particles are reflected when crossing the boundaries. Defaults to 0.5.
            backup_name: name of the file in which to save a backup. In case is None, no file will be saved. Default:None
            backup_frequency: every backup_frequency iterations it will be produced a backup file.
            verbose (bool, optional): Old Classic Verbose. If None, it is set to the one of the main class. Defaults to None.
        """

        _pos = self.mpi_scatter_swarm()
        _val = np.zeros(np.size(self.swarm['val']))
        if verbose is not None: self.verbose = verbose
        for i,p in enumerate(_pos):
            if i%self.size == self.rank:
                _val[i] = f(p,*func_argv)
        self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
        self.update_best_pos()
        if self.verbose and self.rank == 0: self.print_info(0, niter)
        for i in range(1,niter):
            if self.rank == 0:
                self.update_velocity()
                self.update_position()
                self.correct_bound(bound_correct_method, reflect_param = reflect_param)
            _pos = self.mpi_scatter_swarm()
            _val = np.zeros(np.size(self.swarm['val']))
            for j in range(self.npoints):
                p = _pos[j]
                if j%self.size == self.rank: _val[j] = f(p,*func_argv)
            self.comm.Reduce(_val,self.swarm['val'], self.mpi.SUM, 0)
            self.update_best_pos()
            if self.verbose and self.rank == 0: self.print_info(i,niter)

            if backup_name is not None:
                if (i%backup_frequency==0) or (i == niter-1):
                    self.save(filename=backup_name)
