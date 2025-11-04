""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import shutil
import inspect
import copy
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from multiprocessing.dummy import Pool as ThreadPool

from lumopt.utilities.base_script import BaseScript
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.fields import FieldsNoInterp
from lumopt.utilities.gradients import GradientFields

from lumopt.utilities.plotter import Plotter
from lumopt.utilities.configuration import beta_feature_required
from lumopt.lumerical_methods.lumerical_scripts import get_fields, get_fields_on_cad, get_lambda_from_cad


class SuperOptimization(object):
    """
        Optimization super class to run two or more co-optimizations targeting different figures of merit that take the 
        same parameters. The addition operator can be used to aggregate multiple optimizations. All the figures of merit 
        are simply added to generate an overall figure of merit that is passed to the chosen optimizer.

        Parameters
        ----------
        :param optimizations: list of co-optimizations (each of class Optimization). 
        :param plot_history:  A flag indicating if we should plot the history of the parameters (and gradients)
        :param fields_on_cad_only: Process field data on the CAD only to reduce data transfer and memory consumption. 
         Disables plotting of field information. (default is False)
        :use_advanced_fom: enables the use of a FOM 
        :difference_weight: weight for the penalty term for the difference between the figures of merit from the each 
         co-optimization. Roughly corresponds to zeta/1000 from the HPE paper.
    """

    def __init__(self, optimizations=None, plot_history=False, fields_on_cad_only=False, weights=None, 
                 use_advanced_fom=False, difference_weight=0):
        self.plot_history = bool(plot_history)
        self.fields_on_cad_only = fields_on_cad_only
        self.plotter = None                      #< Initialize later, when we are done with adding FOMs
        self.optimizations = optimizations
        self.use_advanced_fom = use_advanced_fom
        self.difference_weight = difference_weight

        if optimizations is not None:
            for opt in optimizations:
                if opt.fom.fom_type == 'intensity_volume':
                    self.concurrent_adjoint_solves = False

            if weights is None:
                self.weights= [1.0]*len(self.optimizations)
            else:
                assert len(weights) == len(optimizations), "weights and optimization objectives must be of the same length." 
                self.weights = weights     
         
        self.old_dir = os.getcwd()
        self.full_fom_hist = [] #< Stores the result of every FOM evaluation
        self.fom_hist = []      #< Only stores the results of every iteration of the optimizer (i.e. not intermediate results from line-searches etc.)
        self.params_hist=[]     #< List of parameters after iterations
        self.grad_hist =[]
        self.continuation_max_iter = 20
        
    def __add__(self,other):
        if self.optimizations is not None:
            opt_list = self.optimizations
            opt_list.append(other)
            opt_weights_list = self.weights
            opt_weights_list.append(1.0)
            
            return SuperOptimization(opt_list, self.plot_history, self.fields_on_cad_only, opt_weights_list)
        else:
            
            return SuperOptimization([self,other],  self.plot_history, self.fields_on_cad_only, [1.0,1.0])

    def __del__(self):
        os.chdir(self.old_dir)

    def initialize(self, start_params=None, bounds=None, working_dir=None):

        print('Initializing super optimization')
        working_dir = 'superopt' if working_dir is None else working_dir
        self.prepare_working_dir(working_dir)
        
        def check_one_forward_sim(co_opt):
            one_forward = self.one_forward
            ## Here, we check that FOMs have only one source
            co_opt.sim = Simulation(self.workingDir, co_opt.use_var_fdtd, co_opt.hide_fdtd_cad)
            co_opt.base_script(co_opt.sim.fdtd)
            source_list = list()
            co_opt.sim.fdtd.selectall()
            numElements = int(co_opt.sim.fdtd.getnumber())
            for i in range(numElements):
                objType = co_opt.sim.fdtd.get("type",i+1)
                if "Source" in objType:
                    source_list.append(objType)
            if len(source_list) > 1:
                print('Simulation file has', len(source_list),'sources (more than One Source), one forward simulation is not possible')
            else: 
                wavelengths_ref = co_opt.wavelengths
                base_script_file_ref = co_opt.base_script
                geometry_ref = co_opt.geometry
                optimizer_ref = co_opt.optimizer
                for k in range(1,len(self.optimizations)):
                    if np.any(self.optimizations[k].wavelengths.asarray() != wavelengths_ref.asarray()):
                        print('More than One Wavelengths range, one forward simulation is not possible')
                        break
                    elif self.optimizations[k].base_script != base_script_file_ref:
                        print('More than One base_script file, one forward simulation is not possible')
                        break
                    elif self.optimizations[k].geometry != geometry_ref:
                        print('More than One Geometry, one forward simulation is not possible')
                        break
                    elif self.optimizations[k].optimizer != optimizer_ref:
                        print('More than One Optimizer, one forward simulation is not possible')
                        break
                    else: 
                        print('One forward simulation is in progress ')
                        one_forward = True
            return one_forward
        
        self.one_forward = False
        if len(self.optimizations) > 1:
            print('Checking for one forward simulation :',end= '\t') 
            self.one_forward = check_one_forward_sim(self.optimizations[0])
        
        ## Generate a super-optimizer by making a deep copy of the first optimizer and then modifying it
        self.optimizer = copy.deepcopy(self.optimizations[0].optimizer)
        self.optimizer.logging_path = self.workingDir #< Logging from the main optimizer should land in the common paths for super-optimizations
        target_fom_list = [o.fom.target_fom for o in self.optimizations]
        self.target_fom = np.dot(self.weights, target_fom_list)  #< Weighted Sum of all target_foms 

        self.fom_names = ['total']

        fom_plot_style = [o.plot_fom_on_log_scale for o in self.optimizations]
        self.plot_fom_on_log_scale = all(fom_plot_style)

        ## Initialize the individual sub-optimizations
        def init_suboptimization(cur_optimization):
            cur_optimization.optimizer.scale_initial_gradient_to = 0 #< Disable automatic gradient scaling for individual FOMs. Can still be applied on combined FOM
            local_working_dir = os.path.join(self.workingDir,'opts')
            if (self.one_forward and cur_optimization == self.optimizations[0]) or not self.one_forward:
                cur_optimization.initialize(local_working_dir)
            legend_entry = cur_optimization.label if cur_optimization.label else cur_optimization.fom.monitor_name
            self.fom_names.append(legend_entry)
               
        list(map(init_suboptimization, self.optimizations))
        # with ThreadPool(self.num_threads) as pool:
        #     pool.map(func = init_suboptimization, iterable = self.optimizations)

        if start_params is None:
            start_params = self.optimizations[0].geometry.get_current_params()
        if bounds is None:
            bounds = np.array(self.optimizations[0].geometry.bounds)
        
        def callable_fom(params):
            Opt_list = [self.optimizations[0]] if self.one_forward else self.optimizations
            self.optimizations[0].sim.fdtd.clearjobs()
            print('Making forward solves')
            
            def make_ONE_forward_solve(optimization, opt_iter = self.optimizer.iteration, params = params):
                jobs = list()
                iter = opt_iter if optimization.store_all_simulations else 0
                if optimization.concurrent_adjoint_solves:
                    adjoint_job_name = optimization.make_adjoint_sim(params, iter, self.optimizations,self.one_forward)
                    jobs.append(adjoint_job_name)
                return jobs

            def make_forward_solve(optimization, opt_iter = self.optimizer.iteration, params = params):
                jobs = list()
                iter = opt_iter if optimization.store_all_simulations else 0
                forward_job_name = optimization.make_forward_sim(params, iter)

                jobs.append(forward_job_name)

                if optimization.concurrent_adjoint_solves:
                    adjoint_job_name = optimization.make_adjoint_sim(params, iter)
                    jobs.append(adjoint_job_name)
                return jobs
            
            if self.one_forward:
                if self.optimizer.iteration == 0 and self.optimizer.scale_initial_gradient_to == 0.0:
                    for co_opt in self.optimizations[1:]:
                        co_opt.fom.initialize(self.optimizations[0].sim)
                        co_opt.sim = self.optimizations[0].sim
                iter = self.optimizer.iteration if self.optimizations[0].store_all_simulations else 0
                forward_job_name = self.optimizations[0].make_forward_sim(params, iter,self.optimizations,self.one_forward)
                nested_job_list =list(map(make_ONE_forward_solve, self.optimizations))
                nested_job_list[0].insert(0,forward_job_name)
            else:
                with ThreadPool(self.num_threads) as pool:
                    nested_job_list = pool.map(make_forward_solve , iterable = self.optimizations)
     
            for job_list in nested_job_list:
                for job in job_list:
                    self.optimizations[0].sim.fdtd.addjob(job)

            self.optimizations[0].sim.fdtd.runjobs()
            print('Processing forward solves')

            def process_forward_solve(optimization, opt_iter = self.optimizer.iteration):
                iter = self.optimizer.iteration if optimization.store_all_simulations else 0
                return optimization.process_forward_sim(iter,self.optimizations, self.one_forward)
                       
            results = list(map(process_forward_solve, self.optimizations))

            # Separate the results into fom_list and fom_list_wavelength
            fom_list = [result[0] for result in results]
            fom_list_wavelength = [result[1] for result in results]


            self.fom_list = fom_list
            self.fom_list_wavelength = fom_list_wavelength
            
            if self.use_advanced_fom:
                # FOM is using soft-min (sum-log-exp) as in HPE Labs paper https://arxiv.org/abs/2210.05655
                # factors of 100 used to ensure soft-min is behaving as expected. self.difference corresponds to zeta/1000
                self.combined_fom_wavelength = -2*np.log(np.exp(-100*fom_list_wavelength[0]) + np.exp(-100*fom_list_wavelength[1]))/100 - self.difference_weight * 100**2*(fom_list_wavelength[0]-fom_list_wavelength[1])**2

                ## Integrate AFTER combined_fom
                wavelengths = self.optimizations[0].fom.wavelengths
                if len(wavelengths) > 1:
                    wavelength_range = wavelengths.max() - wavelengths.min()
                    T_fwd_weights = self.optimizations[0].fom.target_T_fwd_weights(wavelengths)
                    target_T_fwd = len(results)*self.optimizations[0].fom.target_T_fwd(wavelengths) # multiply by the # of co-optimizations
                    fom_integrand = T_fwd_weights*(np.abs(target_T_fwd) - np.abs(self.combined_fom_wavelength - target_T_fwd))/wavelength_range
                    combined_fom = np.trapz(fom_integrand,wavelengths)
                else:
                    combined_fom = self.combined_fom_wavelength
            else:
                combined_fom = np.dot(self.weights, fom_list)

            self.combined_fom = combined_fom

            ## Save field information (if global monitor is available)
            for optimization in Opt_list:
                optimization.save_fields_to_vtk(self.optimizer.iteration)

            dist_to_target_fom = self.target_fom - combined_fom  #< For plotting/logging we store the distance to a target
            self.full_fom_hist.append(dist_to_target_fom) 
            print('=> Combined FOM = {} ({})'.format(dist_to_target_fom, combined_fom))

            ## If the geometry class has an additional penalty term (e.g. min feature size for topology)
            if hasattr(self.optimizations[0].geometry,'calc_penalty_term'):
                fom_penalty = self.optimizations[0].geometry.calc_penalty_term(self.optimizations[0].sim, params)
                print('Actual fom: {}, Penalty term: {}, Total fom: {}'.format(combined_fom, fom_penalty,(combined_fom + fom_penalty)))
                combined_fom += fom_penalty
            return combined_fom

        def callable_jac(params):
            canMakeAdjointSim = True
            for opt in self.optimizations:
                canMakeAdjointSim = canMakeAdjointSim & opt.fom.can_make_adjoint_sim(opt.sim)
            if not canMakeAdjointSim:
                callable_fom(params) # IntensityVolume requires forward simulation to initialize adjoint source
                canMakeAdjointSim = True
                for opt in self.optimizations:
                    canMakeAdjointSim = canMakeAdjointSim & opt.fom.can_make_adjoint_sim(opt.sim)

            assert canMakeAdjointSim

            self.optimizations[0].sim.fdtd.clearjobs()
            if self.one_forward:
                for co_opt in self.optimizations[1:]:
                    co_opt.sim = self.optimizations[0].sim
            print('Making adjoint solves')
                
            def make_adjoint_solves_One_Forward(arg_pair, opt_iter = self.optimizer.iteration, params = params):
                idx, optimization = arg_pair[0], arg_pair[1]
                jobs = list()
                iter = opt_iter if optimization.store_all_simulations else 0
                no_forward_fields = not hasattr(optimization,'forward_fields')
                params_changed = not np.allclose(params, optimization.geometry.get_current_params())
                redo_forward_sim_One = (no_forward_fields or params_changed) if (optimization == self.optimizations[0]) else False 
                redo_forward_sim = no_forward_fields or params_changed
                do_adjoint_sim = redo_forward_sim or not optimization.concurrent_adjoint_solves or optimization.forward_fields_iter != iter
                if redo_forward_sim_One:
                    forward_job_name = optimization.make_forward_sim(params, iter,self.optimizations,self.one_forward)
                    jobs.append(forward_job_name)
                if do_adjoint_sim:
                    adjoint_job_name = optimization.make_adjoint_sim(params, iter,self.optimizations,self.one_forward)
                    jobs.append(adjoint_job_name)
                return idx, redo_forward_sim, jobs
            
            def make_adjoint_solves(arg_pair, opt_iter = self.optimizer.iteration, params = params):
                idx, optimization = arg_pair[0], arg_pair[1]
                jobs = list()
                iter = opt_iter if optimization.store_all_simulations else 0
                no_forward_fields = not hasattr(optimization,'forward_fields')
                params_changed = not np.allclose(params, optimization.geometry.get_current_params())
                redo_forward_sim = no_forward_fields or params_changed
                do_adjoint_sim = redo_forward_sim or not optimization.concurrent_adjoint_solves or optimization.forward_fields_iter != iter
                if redo_forward_sim:
                    forward_job_name = optimization.make_forward_sim(params, iter,self.optimizations,self.one_forward)
                    jobs.append(forward_job_name)
                if do_adjoint_sim:
                    adjoint_job_name = optimization.make_adjoint_sim(params, iter,self.optimizations,self.one_forward)
                    jobs.append(adjoint_job_name)
                return idx, redo_forward_sim, jobs
            
            if self.one_forward:
                no_forward_fields = not hasattr(self.optimizations[0],'forward_fields')
                params_changed = not np.allclose(params, self.optimizations[0].geometry.get_current_params())
                redo_forward_sim = no_forward_fields or params_changed
                if redo_forward_sim:
                    if self.optimizer.iteration == 0 and self.optimizer.scale_initial_gradient_to != 0.0:
                        for co_opt in self.optimizations[1:]:
                            co_opt.fom.initialize(self.optimizations[0].sim)
                nested_job_list = list(map(make_adjoint_solves_One_Forward, enumerate(self.optimizations )))
            else:
                with ThreadPool(self.num_threads) as pool:
                    nested_job_list = pool.map(func = make_adjoint_solves, iterable = enumerate(self.optimizations))   
            
            redo_forward_sim_dict = dict()
            for idx,redo_fwd,job_list in nested_job_list:
                redo_forward_sim_dict[idx] = redo_fwd
                for job in job_list:
                    self.optimizations[0].sim.fdtd.addjob(job)
                  
            if len(self.optimizations[0].sim.fdtd.listjobs()) > 0:
                print('Running solves')
                self.optimizations[0].sim.fdtd.runjobs()
            print('Processing adjoint solves')
            def process_adjoint_solves(arg_pair, redo_fwd = redo_forward_sim_dict, opt_iter = self.optimizer.iteration):
                idx, optimization = arg_pair[0], arg_pair[1]
                iter = opt_iter if optimization.store_all_simulations else 0
                if redo_fwd[idx]:
                    if not self.one_forward:
                        optimization.process_forward_sim(iter,self.optimizations, self.one_forward)
                    elif optimization == self.optimizations[0]:
                        for co_opt in self.optimizations:
                            co_opt.process_forward_sim(iter,self.optimizations, self.one_forward)
                optimization.process_adjoint_sim(iter,self.optimizations,self.one_forward)
                jac, jac_wavelength = optimization.calculate_gradients()
                return np.array(jac), np.array(jac_wavelength)  
            
            results = list(map(process_adjoint_solves, enumerate(self.optimizations)))    

            # Separate the results into fom_list and fom_list_wavelength
            jac_list = [result[0] for result in results]
            jac_list_wavelength = [result[1] for result in results]

            # Now integrate weighted_jac_list
            if self.use_advanced_fom:
                weighted_jac_list_integrand = 2*( np.exp(100*self.fom_list_wavelength[0])*jac_list_wavelength[1] + np.exp(100*self.fom_list_wavelength[1])*jac_list_wavelength[0])/(np.exp(100*self.fom_list_wavelength[0]) + np.exp(100*self.fom_list_wavelength[1])) - self.difference_weight * 2 * 100**2 * (self.fom_list_wavelength[0] - self.fom_list_wavelength[1]) * (jac_list_wavelength[0] - jac_list_wavelength[1])
                
                num_opt_param = weighted_jac_list_integrand.shape[0]
                weighted_jac_list = np.zeros(num_opt_param, dtype = 'complex')

                wavelengths = self.optimizations[0].fom.wavelengths
                wavelength_range = wavelengths.max() - wavelengths.min()
                T_fwd_weights = self.optimizations[0].fom.target_T_fwd_weights(wavelengths)
                target_T_fwd = 2*self.optimizations[0].fom.target_T_fwd(wavelengths) # since it's for the sum of both polarizations
                norm_p = self.optimizations[0].fom.norm_p

                T_fwd_error = self.combined_fom_wavelength - target_T_fwd
                T_fwd_error = np.multiply(T_fwd_weights, T_fwd_error)
                T_fwd_error_integrand = np.power(np.abs(T_fwd_error), norm_p) / wavelength_range            
                const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wavelengths), 1.0 / norm_p - 1.0)
                integral_kernel = np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / wavelength_range

                for i in range(num_opt_param):
                    weighted_jac_list[i] = const_factor * np.trapz(y = integral_kernel * weighted_jac_list_integrand[i], x = wavelengths)
                weighted_jac_list = weighted_jac_list.real
            else:
                weighted_jac_list = np.dot(self.weights, jac_list)
            self.last_grad = weighted_jac_list
            return self.last_grad

        def plotting_function(params):
            ''' This function is called after each iteration'''
            
            ## Add the last FOM evaluation to the list of FOMs that we wish to plot. This removes entries caused by linesearches etc.
            self.params_hist.append(params)
            self.fom_hist.append(self.full_fom_hist[-1])
            self.grad_hist.append(self.last_grad / self.optimizer.scaling_factor)

            for optimization in self.optimizations:
                optimization.plotting_function(params)
            
            ## == Forward the calls to the individual optimizations to show each individual FOM ==
            self.plotter.clear()
            self.plotter.update_fom(self)                   #< Plot the total FOM first so it always has the same line color
            for optimization in self.optimizations:
                self.plotter.update_fom(optimization)       #< Plot individual FOMs as well
                self.plotter.update_geometry(optimization)  #< Also plot geometry, e.g. to show over-etch/under-etch
            self.plotter.set_legend(self.fom_names)

            self.plotter.update_gradient(self)
            self.plotter.draw_and_save()                    #< Finally, refresh the screen and save the image

            for optimization in self.optimizations:
                optimization.save_index_to_vtk(self.optimizer.iteration)

            if hasattr(self.optimizations[0].geometry,'to_file'):
                self.optimizations[0].geometry.to_file(os.path.join(self.workingDir,'parameters_{}.npz').format(self.optimizer.iteration))

            with open(os.path.join(self.workingDir,'convergence_report.txt'),'a') as f:
                f.write('{}, {}'.format(self.optimizer.iteration,self.fom_hist[-1]))
  
                ## Log all the individual FOMs as well
                for optimization in self.optimizations:
                    f.write(', {}'.format(optimization.fom_hist[-1]))
                    self.plotter.update_fom(optimization)     

                if hasattr(self.optimizations[0].geometry,'write_status'):
                    self.optimizations[0].geometry.write_status(f) 
                if len(self.params_hist[-1])<2500:
                    f.write(', {}'.format(np.array2string(self.params_hist[-1], separator=', ', max_line_width=10000)))
                if len(self.grad_hist[-1])<2500:
                    f.write(', {}'.format(np.array2string(self.grad_hist[-1], separator=', ', max_line_width=10000)))
                f.write('\n')
   
        if hasattr(self.optimizer,'initialize'):
            self.optimizer.initialize(start_params=start_params,
                                      callable_fom=callable_fom,
                                      callable_jac=callable_jac,
                                      bounds=bounds,
                                      plotting_function=plotting_function)
        
    
    
    def init_plotter(self):
        if self.plotter is None:
            self.plotter = Plotter(movie = True, plot_history = self.plot_history, plot_fields = not self.fields_on_cad_only)

    def plot_fom(self, fomax, paramsax, gradients_ax):

        if self.plot_fom_on_log_scale:
            fomax.semilogy(np.abs(self.fom_hist))
        else:
            fomax.plot(np.abs(self.fom_hist))
        
        fomax.set_xlabel('Iteration')
        fomax.set_title('Figure of Merit')
        fomax.set_ylabel('FOM')

        if paramsax is not None:
            paramsax.clear()
            paramsax.semilogy(np.abs(self.params_hist))
            paramsax.set_xlabel('Iteration')
            paramsax.set_ylabel('Parameters')
            paramsax.set_title("Parameter evolution")
    
        if (gradients_ax is not None) and hasattr(self, 'grad_hist'):
            gradients_ax.clear()
            gradients_ax.semilogy(np.abs(self.grad_hist))
            gradients_ax.set_xlabel('Iteration')
            gradients_ax.set_ylabel('Gradient Magnitude')
            gradients_ax.set_title("Gradient evolution")

    def plot_gradient(self, fig, ax_fields, ax_gradients):
        
        ## If we process the fields on the CAD, we don't have the data to plot, so do nothing
        if self.fields_on_cad_only:
            return

        self.optimizations[0].gradient_fields.forward_fields.plot(ax_fields, title = 'Forward Fields', cmap = 'Blues')

        ax_gradients.clear()
        main_grad = self.optimizations[0].gradient_fields
        x = main_grad.forward_fields.x
        y = main_grad.forward_fields.y
        xx, yy = np.meshgrid(x, y)
        
        combined_gradients = 0
        for optimization in self.optimizations:
            combined_gradients = combined_gradients + optimization.gradient_fields.get_forward_dot_adjoint_center()

        ax_gradients.pcolormesh(xx*1e6, yy*1e6, combined_gradients, cmap = plt.get_cmap('bwr'), shading='auto')
        ax_gradients.set_title('Sparse perturbation gradient fields')
        ax_gradients.set_xlabel('x(um)')
        ax_gradients.set_ylabel('y(um)')


    def prepare_working_dir(self, working_dir):

        ## Check if we have an absolute path
        if not os.path.isabs(working_dir):
            ## If not, we assume it is meant relative to the path of the script which called this script
            working_dir = os.path.abspath(os.path.join(self.base_file_path,working_dir))
        
        ## Check if the provided path already ends with _xxxx (where xxxx is a number)
        result = re.match(r'_\d+$', working_dir)
        without_suffix = re.sub(r'_\d+$', '', working_dir)
        suffix_num = int(result[1:]) if result else 0
        working_dir = without_suffix+'_{}'.format(suffix_num)

        ## Check if path already exists. If so, keep increasing the number until it does not exist
        while os.path.exists(working_dir):
            suffix_num += 1
            working_dir = without_suffix+'_{}'.format(suffix_num)

        os.makedirs(working_dir)
        os.chdir(working_dir)
        
        ## Copy the calling script over
        if os.path.isfile(self.calling_file_name):
            shutil.copy(self.calling_file_name, working_dir)

        self.workingDir = working_dir

    def run(self, working_dir = None, num_threads = None):
        """ Runs the co-optimization concurrently. The number of threads sets the size of the thread pool used to schedule 
            CAD operations (e.g. loading and saving files). One CAD is launched for every added optimization object, and the 
            number of threads determines how many of them will be active at the same time. All the simulations required for 
            every fom/jac evaluation are scheduled through the job manager (part of the CAD), so the resource configuration
            set there is what will determine how many simulations are run in parallel.

        Parameters
        ----------
        :param num_threads: sets the number of concurrent CAD operations; defaults to the number of added optimization objects.
        :param working_dir: directory where the simulation files and other output is stored
        """

        self.num_threads = int(num_threads) if num_threads else len(self.optimizations)
        if self.num_threads < 1:
            raise UserWarning('number of threads must be positive.')

        ## Figure out from which file this method was called (most likely the driver script)
        frame = inspect.stack()[1]
        self.calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        self.base_file_path = os.path.dirname(self.calling_file_name)

        self.initialize(working_dir=working_dir)
        self.init_plotter()
        
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, os.path.join(self.workingDir,'optimization.png'), 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        ## For topology optimization we are not done yet ... 
        if hasattr(self.optimizations[0].geometry,'progress_continuation'):
            print(' === Starting Binarization Phase === ')
            self.optimizer.max_iter=self.continuation_max_iter

            ## We only want a list of unique separate instances (not references to the same geometry)
            geo_list = [cur_opt.geometry for cur_opt in self.optimizations]
            unique_geometries = {id(x): x for x in geo_list}.values()

            ## Check if any one of the gemetries still needs to continue. If so ... keep at it.
            while any( [cur_geo.progress_continuation() for cur_geo in unique_geometries]):
                for optimization in self.optimizations:
                    optimization.optimizer.reset_start_params(self.params_hist[-1], 0)
                new_scaling = self.optimizer.scale_initial_gradient_to / 1.1 #< Slowly reduce the scaling. TODO: Could make sense to limit this somehow
                self.optimizer.reset_start_params(self.params_hist[-1], new_scaling)

                self.optimizer.run()
                
        final_fom = np.abs(self.fom_hist[-1])
        return final_fom,self.params_hist[-1]


class Optimization(SuperOptimization):
    """ Acts as orchestrator for all the optimization pieces. Calling the member function run will perform the optimization,
        which requires four key pieces: 
            1) a script to generate the base simulation,
            2) an object that defines and collects the figure of merit,
            3) an object that generates the shape under optimization for a given set of optimization parameters and
            4) a gradient based optimizer.

        Parameters
        ----------
        :param base_script:    callable, file name or plain string with script to generate the base simulation.
        :param wavelengths:    wavelength value (float) or range (class Wavelengths) with the spectral range for all simulations.
        :param fom:            figure of merit (class ModeMatch).
        :param geometry:       optimizable geometry (class FunctionDefinedPolygon).
        :param optimizer:      SciyPy minimizer wrapper (class ScipyOptimizers).
        :param hide_fdtd_cad:  flag run FDTD CAD in the background.
        :param use_deps:       flag to use the numerical derivatives calculated directly from FDTD.
        :param plot_history:   plot the history of all parameters (and gradients)
        :param store_all_simulations: Indicates if the project file for each iteration should be stored or not 
        :param save_global_index: Should the project save the result of a global index monitor to file after each iteration (for visualization purposes)
        :param label:          If the optimization is part of a super-optimization, this string is used for the legend of the corresponding FOM plot 
        :param source_name:    Name of the source object in the simulation project (default is "source")
        :param fields_on_cad_only: Process all field data on the CAD only and don't transfer to Python. Reduces memory and improves performance but disables plotting of field/gradient information.
    """

    def __init__(self, base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd = False, hide_fdtd_cad = False, use_deps = True, plot_history = True, store_all_simulations = True, save_global_index = False, label=None, source_name = 'source', fields_on_cad_only = False):
        super().__init__(plot_history=plot_history, fields_on_cad_only=fields_on_cad_only)
        self.base_script = base_script if isinstance(base_script, BaseScript) else BaseScript(base_script)
        self.wavelengths = wavelengths if isinstance(wavelengths, Wavelengths) else Wavelengths(wavelengths)
        self.fom = fom
        self.geometry = geometry 
        self.optimizer = optimizer 
        self.use_var_fdtd = bool(use_var_fdtd) 
        self.hide_fdtd_cad = bool(hide_fdtd_cad) 
        self.source_name = source_name
        if self.fom.fom_type == 'intensity_volume':
            self.concurrent_adjoint_solves = False
        else:
            self.concurrent_adjoint_solves = True

        if callable(use_deps):
            self.use_deps = True                  
            self.custom_deps = use_deps          
        else:
            self.use_deps = bool(use_deps) 
            self.custom_deps = None 

        self.store_all_simulations = store_all_simulations
        self.save_global_index  = save_global_index
        self.unfold_symmetry = geometry.unfold_symmetry
        self.label = label
        self.plot_fom_on_log_scale = (float(fom.target_fom) != 0.0) 

        if self.use_deps:  
            print("Accurate interface detection enabled") 

        ## Figure out from which file this method was called (most likely the driver script)
        frame = inspect.stack()[1]
        self.calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        self.base_file_path = os.path.dirname(self.calling_file_name)

    def check_gradient(self, test_params, dx, working_dir = None):
        self.initialize(working_dir)

        ## Calculate the gradient using the adjoint method:
        adj_grad = self.callable_jac(test_params)
        fd_grad = np.zeros_like(adj_grad)
       
        ## Calculate the gradient using finite differences
        cur_dx = dx/2. 
        for i,param in enumerate(test_params):
            
            d_params = test_params.copy()
            d_params[i] = param + cur_dx
            f1 = self.callable_fom(d_params)
            d_params[i] = param - cur_dx
            f2 = self.callable_fom(d_params)
        
            fd_grad[i] = (f1-f2)/dx

            if (fd_grad[i] == 0.) & (adj_grad[i] == 0.):
                print("Checking gradient #{} : Adjoint={:.4f}, FD={:.4f}, Rel. Diff=0".format(i,adj_grad[i],fd_grad[i]))
            else:
                print("Checking gradient #{} : Adjoint={:.4f}, FD={:.4f}, Rel. Diff={:.4f}".format(i,adj_grad[i],fd_grad[i], 2.*abs(adj_grad[i]-fd_grad[i])/abs(adj_grad[i]+fd_grad[i]) ))

        ## More meaningful comparison of the vector norm:
        np.set_printoptions(threshold=np.inf)

        print(" fd_grad: {}".format(np.array2string(fd_grad, separator=', ', max_line_width=10000)))
        print("adj_grad: {}".format(np.array2string(adj_grad, separator=', ', max_line_width=10000)))
        vec_error = np.linalg.norm(fd_grad-adj_grad)/np.linalg.norm(fd_grad)
        print("norm of vec. diff: {:.4f}".format(vec_error))
        return fd_grad, adj_grad, vec_error

    def run(self, working_dir = None): 
            
        self.initialize(working_dir) 
        self.init_plotter()  
        if self.plotter.movie:  
            with self.plotter.writer.saving(self.plotter.fig, os.path.join(self.workingDir,'optimization.png'), 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        ## For topology optimization we are not done yet ... 
        if hasattr(self.geometry,'progress_continuation'): 
            print(' === Starting Binarization Phase === ')
            self.optimizer.max_iter=self.continuation_max_iter 
            while self.geometry.progress_continuation(): 
                self.optimizer.reset_start_params(self.params_hist[-1], 0.05) #< Run the scaling analysis again
                self.optimizer.run()

        final_fom = np.abs(self.fom_hist[-1]) 
        return final_fom,self.params_hist[-1]

    def plotting_function(self, params):
        ## Add the last FOM evaluation to the list of FOMs that we wish to plot. This removes entries caused by linesearches etc.
        self.fom_hist.append(self.full_fom_hist[-1])

        ## In a multi-FOM optimization, only the first optimization has a plotter
        if self.plotter is not None:

            self.params_hist.append(params)
            self.grad_hist.append(self.last_grad / self.optimizer.scaling_factor)

            self.plotter.clear()
            self.plotter.update_fom(self)
            self.plotter.update_gradient(self) 
            self.plotter.update_geometry(self)     
            self.plotter.draw_and_save()

            self.save_index_to_vtk(self.optimizer.iteration)

            if hasattr(self.geometry,'to_file'):
                self.geometry.to_file(os.path.join(self.workingDir,'parameters_{}.npz').format(self.optimizer.iteration)) 

            with open(os.path.join(self.workingDir,'convergence_report.txt'),'a') as f:
                f.write('{}, {}'.format(self.optimizer.iteration,self.fom_hist[-1]))
                
                if hasattr(self.geometry,'write_status'):
                    self.geometry.write_status(f) 

                if len(self.params_hist[-1])<2500:  
                    f.write(', {}'.format(np.array2string(self.params_hist[-1], separator=', ', max_line_width=10000)))

                if len(self.grad_hist[-1])<2500: 
                    f.write(', {}'.format(np.array2string(self.grad_hist[-1], separator=', ', max_line_width=10000)))

                f.write('\n')

    def initialize(self, working_dir):    
        """ 
            Performs all steps that need to be carried only once at the beginning of the optimization. 
        """
        if self.use_var_fdtd and self.geometry.num_jobs > 1:
            raise ValueError("Parallel meshing not supported with varFDTD")
        
        working_dir = 'opts' if working_dir is None else working_dir
        self.prepare_working_dir(working_dir)   

        ## Store a copy of the script file
        if hasattr(self.base_script, 'script_str'):  
            with open('script_file.lsf','a') as file:  
                file.write(self.base_script.script_str.replace(';',';\n')) 

        # FDTD CAD
        # WARNING: NOT THREAD SAFE, ADD LOCK
        #lock.acquire()
        self.sim = Simulation(self.workingDir, self.use_var_fdtd, self.hide_fdtd_cad)
        self.geometry.check_license_requirements(self.sim)

        # FDTD model
        self.base_script(self.sim.fdtd)  
        Optimization.set_global_wavelength(self.sim, self.wavelengths)
        Optimization.set_source_wavelength(self.sim, self.source_name, self.fom.multi_freq_src, len(self.wavelengths))  

        self.sim.fdtd.setnamed('opt_fields', 'override global monitor settings', False)
        self.sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'none')
        Optimization.add_index_monitor(self.sim, 'opt_fields', self.wavelengths) 
        
        # Optimizer
        start_params = self.geometry.get_current_params() 

        # We need to add the geometry first because it adds the mesh override region
        self.geometry.add_geo(self.sim, start_params, only_update = False, param_id = None) 

        # If we don't have initial parameters yet, try to extract them from the simulation (this is mostly for topology optimization)
        if start_params is None:     
            self.geometry.extract_parameters_from_simulation(self.sim) 
            start_params = self.geometry.get_current_params()  
        self.geometry.num_params = len(start_params)

        callable_fom = self.callable_fom 
        callable_jac = self.callable_jac  
        bounds = self.geometry.bounds

        self.fom.initialize(self.sim) 

        def plotting_function_fwd(params):
             self.plotting_function(params) 
        
        self.optimizer.initialize(start_params = start_params, callable_fom = callable_fom, callable_jac = callable_jac, bounds = bounds, plotting_function = plotting_function_fwd)
        
        self.fom_hist = []
      
    def save_fields_to_vtk(self, cur_iteration):
        if self.save_global_index:
            self.sim.save_fields_to_vtk(os.path.join(self.workingDir,'global_fields_{}').format(cur_iteration))

    def save_index_to_vtk(self, cur_iteration): 
        if self.save_global_index:
            self.sim.save_index_to_vtk(os.path.join(self.workingDir,'global_index_{}').format(cur_iteration))
            
    def make_forward_sim(self, params, iter, co_optimizations = None, one_forward = False): 
        self.sim.fdtd.switchtolayout()
        self.geometry.update_geometry(params, self.sim) 
        self.geometry.add_geo(self.sim, params = None, only_update = True, param_id = 0) 
        Optimization.deactivate_all_sources(self.sim) 
        self.sim.fdtd.setnamed(self.source_name, 'enabled', True) 
        if co_optimizations is not None and len(co_optimizations) > 1 and one_forward: 
            for co_opt in co_optimizations:
                co_opt.fom.make_forward_sim(co_optimizations[0].sim) 
        else:
            self.fom.make_forward_sim(self.sim) 
        forward_name = 'forward_{}'.format(iter) 
        return self.sim.save(forward_name) 
    
    def process_forward_sim(self, iter,co_optimizations = None,one_forward = False):
        if not one_forward or (self == co_optimizations[0] and one_forward): 
            forward_name = 'forward_{}'.format(iter)    
            self.sim.load(forward_name) 
            
            self.sim.fdtd.run()  

            Optimization.check_simulation_was_successful(self.sim) 
            if self.fields_on_cad_only: 
                get_fields_on_cad(  self.sim.fdtd,
                                    monitor_name = 'opt_fields',
                                    field_result_name = 'forward_fields', 
                                    get_eps = True,
                                    get_D = not self.use_deps,
                                    get_H = False,
                                    nointerpolation = not self.geometry.use_interpolation(),
                                    unfold_symmetry = self.unfold_symmetry)
                self.forward_fields_wl = get_lambda_from_cad(self.sim.fdtd, field_result_name = 'forward_fields')
            else:   
                self.forward_fields = get_fields(self.sim.fdtd,
                                            monitor_name = 'opt_fields',
                                            field_result_name = 'forward_fields',
                                            get_eps = True,
                                            get_D = not self.use_deps,
                                            get_H = False,
                                            nointerpolation = not self.geometry.use_interpolation(),
                                            unfold_symmetry = self.unfold_symmetry)
                assert hasattr(self.forward_fields, 'E') 
                self.forward_fields_wl = self.forward_fields.wl

        self.forward_fields_iter = int(iter) 
        
        if one_forward and self != co_optimizations[0]:
            self.forward_fields = co_optimizations[0].forward_fields         
            self.forward_fields_wl = co_optimizations[0].forward_fields.wl   
                
        if not one_forward: 
            fom, fom_wavelength = self.fom.get_fom(self.sim) 
        else:
            fom, fom_wavelength = self.fom.get_fom(co_optimizations[0].sim)

        if self.store_all_simulations: 
            if not one_forward or self != co_optimizations[0]:
                self.sim.remove_data_and_save() #< Remove the data from the file to save disk space.
            if one_forward and self == co_optimizations[-1]:
                co_optimizations[0].sim.remove_data_and_save()
        
        dist_to_target_fom = self.fom.target_fom - fom #< For plotting/logging we store the distance to a target
        self.full_fom_hist.append(dist_to_target_fom) 
        if self.fom.target_fom == 0.0:
            print('FOM = {}'.format(fom)) 
        else:
            print('FOM = {} ({} - {})'.format(dist_to_target_fom, self.fom.target_fom, fom)) 
        return fom, fom_wavelength

    def callable_fom(self, params): 
        """ Function for the optimizers to retrieve the figure of merit.
            :param params:  optimization parameters.
            :param returns: figure of merit.
        """
    
        self.sim.fdtd.clearjobs() 
        iter = self.optimizer.iteration if self.store_all_simulations else 0 
        self.sim.iteration = iter
        print('Making forward solve')
        forward_job_name = self.make_forward_sim(params, iter) 
        self.sim.fdtd.addjob(forward_job_name)

        if self.concurrent_adjoint_solves: 
            print('Making adjoint solve')  
            adjoint_job_name = self.make_adjoint_sim(params, iter) 
            self.sim.fdtd.addjob(adjoint_job_name)
        print('Running solves')
        self.sim.fdtd.runjobs()

        print('Processing forward solve')
        fom, fom_wavelength = self.process_forward_sim(iter)

        ## If the geometry class has an additional penalty term (e.g. min feature size for topology)
        if hasattr(self.geometry,'calc_penalty_term'):
            fom_penalty = self.geometry.calc_penalty_term(self.sim, params)
            print('Actual fom: {}, Penalty term: {}, Total fom: {}'.format(fom, fom_penalty,(fom + fom_penalty)))
            fom += fom_penalty

        return fom

    def make_adjoint_sim(self, params, iter,co_optimizations = None, one_forward = False):
        assert np.allclose(params, self.geometry.get_current_params())   
        if one_forward:   
            adjoint_name = 'adjoint_{0}_{1}'.format(co_optimizations.index(self),iter)
            self.sim = co_optimizations[0].sim    
        else:
            adjoint_name = 'adjoint_{}'.format(iter)  
        self.sim.fdtd.switchtolayout()            
        self.geometry.add_geo(self.sim, params = None, only_update = True, param_id = 0)  
        self.sim.fdtd.setnamed(self.source_name, 'enabled', False)   
        
        self.fom.make_adjoint_sim(self.sim)    
        
        if  co_optimizations is not None and len(co_optimizations) > 1 and one_forward: 
            for co_opt in co_optimizations:
                if co_opt != self:
                    self.sim.fdtd.setnamed(co_opt.fom.adjoint_source_name, 'enabled', False) 
        return self.sim.save(adjoint_name)  

    def process_adjoint_sim(self, iter,co_optimizations = None, one_forward = False):
        if one_forward:   
            adjoint_name = 'adjoint_{0}_{1}'.format(co_optimizations.index(self),iter)
            self.sim = co_optimizations[0].sim  
        else:    
            adjoint_name = 'adjoint_{}'.format(iter)    
        
        self.sim.load(adjoint_name)  
        self.sim.fdtd.run()

        Optimization.check_simulation_was_successful(self.sim) 

        if self.fields_on_cad_only:  
            get_fields_on_cad(  self.sim.fdtd,
                                monitor_name = 'opt_fields',
                                field_result_name = 'adjoint_fields',
                                get_eps = not self.use_deps,
                                get_D = not self.use_deps,
                                get_H = False,
                                nointerpolation = not self.geometry.use_interpolation(),
                                unfold_symmetry = self.unfold_symmetry)
        else:                   
            self.adjoint_fields = get_fields(self.sim.fdtd,
                                         monitor_name = 'opt_fields',
                                         field_result_name = 'adjoint_fields',
                                         get_eps = not self.use_deps,
                                         get_D = not self.use_deps,
                                         get_H = False,
                                         nointerpolation = not self.geometry.use_interpolation(),
                                         unfold_symmetry = self.unfold_symmetry)
            assert hasattr(self.adjoint_fields, 'E')      
            self.adjoint_fields.iter = int(iter)    
     
        self.scaling_factor = self.fom.get_adjoint_field_scaling(self.sim)  
                       
        if not self.fields_on_cad_only:                 
            self.adjoint_fields.scale(3, self.scaling_factor)  

        if self.store_all_simulations:    
            self.sim.remove_data_and_save() #< Remove the data from the file to save disk space.
              
        
    def callable_jac(self, params):  
        """ Function for the optimizer to extract the figure of merit gradient.
            :param params:  optimization paramaters.
            :param returns: partial derivative of the figure of merit with respect to each optimization parameter.
        """

        self.sim.fdtd.clearjobs()                                               
        iter = self.optimizer.iteration if self.store_all_simulations else 0
        no_forward_fields = not hasattr(self,'forward_fields')          
        params_changed = not np.allclose(params, self.geometry.get_current_params()) 
        redo_forward_sim = no_forward_fields or params_changed                 
        do_adjoint_sim = redo_forward_sim or not self.concurrent_adjoint_solves or self.forward_fields_iter != iter  


        if redo_forward_sim:                                            
            print('Making forward solve')                               
            forward_job_name = self.make_forward_sim(params, iter)      

            print('Processing forward solve in sequence')
            fom = self.process_forward_sim(iter)    

        if do_adjoint_sim:                                           
            print('Making adjoint solve')                           
            adjoint_job_name = self.make_adjoint_sim(params, iter)   
            
        print('Processing adjoint solve in sequence')
        self.process_adjoint_sim(iter)

        grad, grad_wavelength = self.calculate_gradients()  
        self.last_grad = grad      

        if hasattr(self.geometry,'calc_penalty_term'):  
            print('Calculating Penalty Terms')    
            penalty_grad = self.geometry.calc_penalty_gradient(self.sim, params) 
            grad += penalty_grad

        return grad 

    def calculate_gradients(self): 
        """ Calculates the gradient of the figure of merit (FOM) with respect to each of the optimization parameters.
            It assumes that both the forward and adjoint solves have been run so that all the necessary field results
            have been collected. There are currently two methods to compute the gradient:
                1) using the permittivity derivatives calculated directly from meshing (use_deps == True) and
                2) using the shape derivative approximation described in Owen Miller's thesis (use_deps == False).
        """
        if not self.fields_on_cad_only:  
            self.gradient_fields = GradientFields(forward_fields = self.forward_fields, adjoint_fields = self.adjoint_fields) 

        self.sim.fdtd.switchtolayout() 
        if self.use_deps:       
            if self.custom_deps:
                self.custom_deps(self.sim,self.geometry)
            else:
                self.geometry.d_eps_on_cad(self.sim)
            fom_partial_derivs_vs_wl = GradientFields.spatial_gradient_integral_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.scaling_factor, self.geometry.num_params)
            self.gradients, self.gradients_wavelength = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl.transpose(), self.forward_fields_wl) 
        else:   
            if hasattr(self.geometry,'calculate_gradients_on_cad'): 
                grad_name = self.geometry.calculate_gradients_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.scaling_factor)          
                self.gradients = self.fom.fom_gradient_wavelength_integral_on_cad(self.sim, grad_name, self.forward_fields_wl)
                self.gradients_wavelength = None
            else:
                fom_partial_derivs_vs_wl = self.geometry.calculate_gradients(self.gradient_fields) 
                self.gradients, self.gradients_wavelength = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, self.forward_fields_wl) 
            
        return self.gradients, self.gradients_wavelength
    
    def plot_gradient(self, fig, ax1, ax2):  
        self.gradient_fields.plot(fig, ax1, ax2)

    @staticmethod
    def add_index_monitor(sim, monitor_name, wavelengths):
        sim.fdtd.select(monitor_name)
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(monitor_name))
        index_monitor_name = monitor_name + '_index'
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addindex()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addeffectiveindex()
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        sim.fdtd.set('name', index_monitor_name)
        sim.fdtd.setnamed(index_monitor_name, 'override global monitor settings', True)
        if wavelengths.custom:
            sim.fdtd.setnamed(index_monitor_name, 'sample spacing', 'uniform')
            sim.fdtd.setnamed(index_monitor_name, 'use wavelength spacing', True)                  
            sim.fdtd.setnamed(index_monitor_name, 'frequency points', 1)
        else:    
            sim.fdtd.setnamed(index_monitor_name, 'frequency points', 1)
        sim.fdtd.setnamed(index_monitor_name, 'record conformal mesh when possible', True)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geometric_props = ['monitor type']
        geometric_props.extend(Optimization.cross_section_monitor_props(monitor_type))
        for prop_name in geometric_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(index_monitor_name, prop_name, prop_val)
        sim.fdtd.setnamed(index_monitor_name, 'spatial interpolation', 'none')
    
    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        if monitor_type == '3D':
            geometric_props.extend(['x span','y span','z span'])
        elif monitor_type == '2D X-normal':
            geometric_props.extend(['y span','z span'])
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span','z span'])
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span','y span'])
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props

    @staticmethod
    def set_global_wavelength(sim, wavelengths):
        if not wavelengths.custom:  
            sim.fdtd.setglobalmonitor('use source limits', True)
            sim.fdtd.setglobalmonitor('use wavelength spacing', True)
            sim.fdtd.setglobalmonitor('frequency points', len(wavelengths))
            sim.fdtd.setglobalsource('set wavelength', True)
            sim.fdtd.setglobalsource('wavelength start', wavelengths.min())
            sim.fdtd.setglobalsource('wavelength stop', wavelengths.max())
        else:
            sim.fdtd.setglobalmonitor('sample spacing', 'custom')
            sim.fdtd.setglobalmonitor('custom frequency samples', sp.constants.speed_of_light/wavelengths.asarray())
            sim.fdtd.setglobalsource('wavelength start', wavelengths.min())
            sim.fdtd.setglobalsource('wavelength stop', wavelengths.max())
            
            
    @staticmethod
    def set_source_wavelength(sim, source_name, multi_freq_src, freq_pts):
        if sim.fdtd.getnamednumber(source_name) < 1:   
            raise UserWarning("At least one source with the name '{}' must be defined in the base simulation.".format(source_name))
        if sim.fdtd.getnamed(source_name, 'override global source settings'):
            print('Wavelength range of source object will be superseded by the global settings.')
        sim.fdtd.setnamed(source_name, 'override global source settings', False)  
        sim.fdtd.select(source_name)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'frequency points', freq_pts)
        elif sim.fdtd.haveproperty('multifrequency beam calculation'):  
            sim.fdtd.setnamed(source_name, 'multifrequency beam calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'number of frequency points', freq_pts)

    @staticmethod
    def set_use_legacy_conformal_interface_detection(sim, flagVal):
        # should not be called, warning kept as a guard
        raise DeprecationWarning('you are using a deprecated mesh refinement option.')
            
    @staticmethod
    def check_simulation_was_successful(sim):
        if sim.fdtd.getnamednumber('FDTD') == 1:
            simulation_status = sim.fdtd.getresult('FDTD','status') 
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            simulation_status = sim.fdtd.getresult('varFDTD','status')
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        if simulation_status != 1 and simulation_status != 2: # run to full simulation time (1) or autoshutoff triggered (2)
            raise UserWarning('FDTD simulation did not complete successfully: status {0}'.format(simulation_status))
        return simulation_status

    @staticmethod
    def deactivate_all_sources(sim):
        sim.fdtd.selectall()  
        numElements = int(sim.fdtd.getnumber())  
        for i in range(numElements):    
            objType = sim.fdtd.get("type",i+1)
            if "Source" in objType:
                sim.fdtd.set("enabled",False,i+1) 




class FAIDOptimization(Optimization):
    @beta_feature_required("enable_FAID_beta", "FAID Optimization")
    def __init__(self, continuation_max_iter=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuation_max_iter = continuation_max_iter
        self.script_dir = os.getcwd()

    def _is_matplotlib_object(self, obj):
        return isinstance(
            obj, (mpl.transforms.TransformNode, mpl.figure.Figure, mpl.axes.Axes)
        )

    def _get_copyable_attributes(self):
        attrs = {}
        for name, value in self.__dict__.items():
            if name in ("plotter", "sim"):
                continue

            if self._is_matplotlib_object(value):
                continue

            if isinstance(value, dict):
                filtered_dict = {
                    k: v for k, v in value.items() if not self._is_matplotlib_object(v)
                }
                attrs[name] = filtered_dict
            else:
                attrs[name] = value
        return attrs

    def copy(self):
        new_opt = FAIDOptimization.__new__(FAIDOptimization)
        attrs = self._get_copyable_attributes()

        for name, value in attrs.items():
            try:
                setattr(new_opt, name, copy.deepcopy(value))
            except Exception:
                try:
                    setattr(new_opt, name, copy.copy(value))
                except Exception:
                    setattr(new_opt, name, value)

        return new_opt

    def _setup_working_dir(self, working_dir=None):
        if working_dir is None:
            results_dir = os.path.join(self.script_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

            existing_runs = [
                d
                for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
            ]
            next_run = len(existing_runs)

            working_dir = os.path.join(results_dir, f"run_{next_run}")

        return working_dir

    def get_fom(self, params=None, working_dir=None):
        if working_dir is None:
            working_dir = self._setup_working_dir(
                os.path.join(self.script_dir, "results", "single_fom")
            )

        self.initialize(working_dir)

        if params is None:
            params = self.geometry.get_current_params()

        forward_job_name = self.make_forward_sim(params, 0)
        self.sim.fdtd.addjob(forward_job_name)
        self.sim.fdtd.runjobs()
        fom = self.process_forward_sim(0)

        return fom

    def run(self, working_dir=None):
        working_dir = self._setup_working_dir(working_dir)

        self.initialize(working_dir)
        self.init_plotter()

        if self.plotter.movie:
            with self.plotter.writer.saving(
                self.plotter.fig, os.path.join(self.workingDir, "optimization.png"), 100
            ):
                self.optimizer.run()
        else:
            self.optimizer.run()

        if hasattr(self.geometry, "progress_continuation"):
            print(" === Starting Binarization Phase === ")
            self.optimizer.max_iter = self.continuation_max_iter

            while self.geometry.progress_continuation():
                self.optimizer.reset_start_params(self.params_hist[-1], 0)
                new_scaling = self.optimizer.scale_initial_gradient_to / 1.1
                self.optimizer.reset_start_params(self.params_hist[-1], new_scaling)
                self.optimizer.run()

        final_fom = abs(self.fom_hist[-1])
        return final_fom, self.params_hist[-1]

    def plotting_function(self, params):
        super().plotting_function(params)
        if (
            hasattr(self.geometry, "fabrication_model")
            and self.geometry.fabrication_model is not None
            and hasattr(self.geometry.fabrication_model, "plot_fabrication_effects")
        ):
            self.geometry.fabrication_model.plot_fabrication_effects()
