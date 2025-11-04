""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
import numpy as np
import lumapi
import copy
import os
import subprocess
import platform

class Geometry(object):

    self_update=False
    unfold_symmetry = True #< By default, we do want monitors to unfold symmetry
    use_central_differences = False
    obj_bounds = None
    serial = 1

    def use_interpolation(self):
        return False
        
    def check_license_requirements(self, sim):
        return True

    def __init__(self,geometries,operation):
        self.geometries=geometries
        self.operation=operation
        if self.operation=='mul':
            self.bounds=geometries[0].bounds
        if self.operation=='add':
            self.bounds=np.concatenate((np.array(geometries[0].bounds),np.array(geometries[1].bounds)))
        self.dx=max([geo.dx for geo in self.geometries])

        return

    def __add__(self,other):
        '''Two geometries with independent parameters'''
        geometries=[self,other]
        return Geometry(geometries,'add')

    def __mul__(self,other):
        '''Two geometries with common parameters'''
        geometries = [self, other]
        return Geometry(geometries, 'mul')

    def add_geo(self, sim, params, only_update, param_id=None):
        for geometry in self.geometries:
            geometry.add_geo(sim, params, only_update, param_id)

    def initialize(self,wavelengths,opt):
        for geometry in self.geometries:
            geometry.initialize(wavelengths,opt)
        self.opt=opt

    def update_geometry(self, params, sim = None):
        if self.operation=='mul':
            for geometry in self.geometries:
                geometry.update_geometry(params,sim)

        if self.operation=='add':
            n1=len(self.geometries[0].get_current_params())
            self.geometries[0].update_geometry(params[:n1],sim)
            self.geometries[1].update_geometry(params[n1:],sim)

    def calculate_gradients(self, gradient_fields):
        derivs1 = np.array(self.geometries[0].calculate_gradients(gradient_fields))
        derivs2 = np.array(self.geometries[1].calculate_gradients(gradient_fields))

        if self.operation=='mul':
            return derivs1+derivs2
        if self.operation=='add':
            np.concatenate(derivs1,derivs2)

    def get_current_params(self):
        params1=np.array(self.geometries[0].get_current_params())
        if self.operation=='mul':
            return params1
        if self.operation=='add':
            return params1+np.array(self.geometries[1].get_current_params())

    def plot(self,*args):
        return False

    @staticmethod
    def get_eps_from_index_monitor(fdtd, eps_result_name, current_params=None, param_index=None, get_bounds=None, monitor_name='opt_fields'):
        """
        Retrieves the permittivity (epsilon) tensor components from the index monitor in the simulation.

        This method processes the index data from the specified monitor and calculates the squared components 
        of the refractive index (index_x^2, index_y^2, index_z^2) to populate the epsilon tensor. Optionally, 
        it can restrict the data extraction to specific bounds in the simulation grid. 

        Parameters
        ----------
        :fdtd: The simulation object used to execute Lumerical FDTD commands.
        :eps_result_name (str): The name of the variable to store the epsilon tensor result.
        :current_params (list, optional): Current parameter values for the geometry. Defaults to None.
        :param_index (int, optional): Index of the parameter being optimized. Defaults to None.
        :get_bounds (callable, optional): A function that returns bounds for the simulation grid 
            based on current_params and param_index. Defaults to None.
        :monitor_name (str, optional): The name of the monitor to retrieve index data from. Defaults to 'opt_fields'.

        Notes:
            - If symmetry boundaries are applied, the symmetry region must lie on the negative plane for the unfolding
              of the index monitor data to work correctly. 
            - If `get_bounds` is not provided, the entire simulation grid is used.
            - If bounds are provided, the method ensures that the indices are adjusted to include the specified range 
              and remain within the grid limits.    

        Returns:
            None: The epsilon tensor is stored in the variable specified by `eps_result_name` within the simulation.
        """

        index_monitor_name = monitor_name + '_index'
        
        if not get_bounds:
            fdtd.eval("{0}_data_set = getresult('{0}','index');".format(index_monitor_name) +
                    "{0} = matrix(length({1}_data_set.x), length({1}_data_set.y), length({1}_data_set.z), length({1}_data_set.f), 3);".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 1) = {1}_data_set.index_x^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 2) = {1}_data_set.index_y^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 3) = {1}_data_set.index_z^2;".format(eps_result_name, index_monitor_name) +
                    "clear({0}_data_set);".format(index_monitor_name))
        
        else:
            bounds = get_bounds(current_params, param_index)
            assert bounds is None or len(bounds) in [2, 3], "bounds should be None or have 2 or 3 pairs of values"
            if bounds is not None:
                for pair in bounds:
                    assert len(pair) == 2, "Each pair in bounds should have exactly two elements"

            if len(bounds) == 2:
                fdtd.eval(
                    "x_min_index = find(x_grid,{0})-1;".format(bounds[0][0])+ # decrement the index by 1 to ensure our grid includes the value
                    "x_max_index = find(x_grid,{0})+1;".format(bounds[0][1])+
                    "y_min_index = find(y_grid,{0})-1;".format(bounds[1][0])+ 
                    "y_max_index = find(y_grid,{0})+1;".format(bounds[1][1])+

                    "if (x_min_index==0) {x_min_index=1;}"+ #ensure inc/decremented indices are within bounds of x_grid
                    "if (y_min_index==0) {y_min_index=1;}"+
                    "if (x_max_index>length(x_grid)) {x_max_index=length(x_grid);}"+
                    "if (y_max_index>length(y_grid)) {y_max_index=length(y_grid);}"+
            
                    "original_opt_fields_index=getnamed('opt_fields_index',{'x','x span','y','y span'},1);" +
                    "setnamed('opt_fields_index',{'x' : 0.5*(x_grid(x_min_index) + x_grid(x_max_index)),'x span' : x_grid(x_max_index) - x_grid(x_min_index), 'y' : 0.5*(y_grid(y_min_index) + y_grid(y_max_index)),'y span' : y_grid(y_max_index) - y_grid(y_min_index)});" +

                    "{0}_data_set = getresult('{0}','index');".format(index_monitor_name) +
                    "unfolded_x = size(opt_fields_index_data_set.index_x,1);"+
                    "unfolded_y = size(opt_fields_index_data_set.index_x,2);"+
                    "initial_x = 1 + x_max_index - x_min_index;"+
                    "initial_y = 1 + y_max_index - y_min_index;"+
                    "if (unfolded_x >= initial_x) {"+ 
                    "    start_x = 1+unfolded_x-initial_x;"+
                    "    end_x = unfolded_x;"+ 
                    "} else {"+ #bounding box is inside symmetry region
                    "    start_x = 1;"+
                    "    end_x = unfolded_x;"+
                    "}"+
                    "if (unfolded_y >= initial_y) {"+
                    "    start_y = 1+unfolded_y-initial_y;"+
                    "    end_y = unfolded_y;"+
                    "} else {"+
                    "    start_y = 1;"+
                    "    end_y = unfolded_y;"+
                    "    y_min_index = y_max_index + start_y - end_y;"+
                    "}"+

                    "{0} = matrix(x_max_index - x_min_index + 1, y_max_index - y_min_index + 1, length(z_grid), 1, 3);".format(eps_result_name)+
                    "{0}(:, :, :, :, 1) = {1}_data_set.index_x(start_x:end_x, start_y:end_y, :, :)^2;".format(eps_result_name, index_monitor_name) + 
                    "{0}(:, :, :, :, 2) = {1}_data_set.index_y(start_x:end_x, start_y:end_y, :, :)^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 3) = {1}_data_set.index_z(start_x:end_x, start_y:end_y, :, :)^2;".format(eps_result_name, index_monitor_name) +
                    "setnamed('opt_fields_index',original_opt_fields_index);")

            elif len(bounds) == 3:
                fdtd.eval(
                    "x_min_index = find(x_grid,{0})-1;".format(bounds[0][0])+ # decrement the index by 1 to ensure our grid includes the value
                    "x_max_index = find(x_grid,{0})+1;".format(bounds[0][1])+
                    "y_min_index = find(y_grid,{0})-1;".format(bounds[1][0])+ 
                    "y_max_index = find(y_grid,{0})+1;".format(bounds[1][1])+
                    "z_min_index = find(z_grid,{0})-1;".format(bounds[2][0])+ 
                    "z_max_index = find(z_grid,{0})+1;".format(bounds[2][1])+

                    "if (x_min_index==0) {x_min_index=1;}"+ #ensure inc/decremented indices are within bounds of x_grid
                    "if (y_min_index==0) {y_min_index=1;}"+
                    "if (z_min_index==0) {z_min_index=1;}"+
                    "if (x_max_index > length(x_grid)) {x_max_index = length(x_grid);}"+
                    "if (y_max_index > length(y_grid)) {y_max_index = length(y_grid);}"+
                    "if (z_max_index > length(z_grid)) {z_max_index = length(z_grid);}"+
                    "original_opt_fields_index=getnamed('opt_fields_index', {'x', 'x span', 'y', 'y span', 'z', 'z span'}, 1);" +
                    "setnamed('opt_fields_index',{'x' : 0.5*(x_grid(x_min_index) + x_grid(x_max_index)),'x span' : x_grid(x_max_index) - x_grid(x_min_index), 'y' : 0.5*(y_grid(y_min_index) + y_grid(y_max_index)),'y span' : y_grid(y_max_index) - y_grid(y_min_index), 'z' : 0.5*(z_grid(z_min_index) + z_grid(z_max_index)),'z span' : z_grid(z_max_index) - z_grid(z_min_index)});" +

                    "{0}_data_set = getresult('{0}','index');".format(index_monitor_name) +
                    "unfolded_x = size(opt_fields_index_data_set.index_x, 1);"+
                    "unfolded_y = size(opt_fields_index_data_set.index_x, 2);"+
                    "unfolded_z = size(opt_fields_index_data_set.index_x, 3);"+
                    "initial_x = 1 + x_max_index - x_min_index;"+
                    "initial_y = 1 + y_max_index - y_min_index;"+
                    "initial_z = 1 + z_max_index - z_min_index;"+
                    "if (unfolded_x >= initial_x) {"+
                    "    start_x = 1+unfolded_x-initial_x;"+
                    "    end_x = unfolded_x;"+
                    "} else {"+
                    "    start_x = 1;"+
                    "    end_x = unfolded_x;"+
                    "}"+
                    "if (unfolded_y >= initial_y) {"+
                    "    start_y = 1+unfolded_y-initial_y;"+
                    "    end_y = unfolded_y;"+
                    "} else {"+
                    "    start_y = 1;"+
                    "    end_y = unfolded_y;"+
                    "    y_min_index = y_max_index + start_y - end_y;"+
                    "}"+
                    "if (unfolded_z >= initial_z) {"+
                    "    start_z = 1+unfolded_z-initial_z;"+
                    "    end_z = unfolded_z;"+
                    "} else {"+
                    "    start_z = 1;"+
                    "    end_z = unfolded_z;"+
                    "    z_min_index = z_max_index + start_z - end_z;"+
                    "}"+
                    "{0} = matrix(x_max_index - x_min_index + 1, y_max_index - y_min_index + 1, z_max_index - z_min_index + 1, 1, 3);".format(eps_result_name)+
                    "{0}(:, :, :, :, 1) = {1}_data_set.index_x(start_x:end_x, start_y:end_y, start_z:end_z, :)^2;".format(eps_result_name, index_monitor_name) + 
                    "{0}(:, :, :, :, 2) = {1}_data_set.index_y(start_x:end_x, start_y:end_y, start_z:end_z, :)^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 3) = {1}_data_set.index_z(start_x:end_x, start_y:end_y, start_z:end_z, :)^2;".format(eps_result_name, index_monitor_name) +
                    "setnamed('opt_fields_index', original_opt_fields_index);")


            else:   # if 'None' is returned from get_bounds -> use the full index monitor
                fdtd.eval("{0}_data_set = getresult('{0}','index');".format(index_monitor_name) +
                    "{0} = matrix(length({1}_data_set.x), length({1}_data_set.y), length({1}_data_set.z), length({1}_data_set.f), 3);".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 1) = {1}_data_set.index_x^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 2) = {1}_data_set.index_y^2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 3) = {1}_data_set.index_z^2;".format(eps_result_name, index_monitor_name) +
                    "clear({0}_data_set);".format(index_monitor_name) +
                    "index_data = getresult('{0}','index');".format(index_monitor_name))
            
    
    def d_eps_on_cad_parallel(self, sim):
        current_params = self.get_current_params()
        cur_dx = self.dx/2 if self.use_central_differences else self.dx
        lumapi.putDouble(sim.fdtd.handle, "dx", cur_dx)

        if self.obj_bounds:
            script_base = (
                "i = {0};"
                "load('TempFileMesh_{1}'+num2str(i)+'.fsp');"
                "index_monitor = '{2}';"
                "redrawoff;"
                "setnamed(index_monitor,{{'x': 0.5*({4} + {3}), 'x span': {4} - {3}, 'y': 0.5*({6} + {5}), 'y span': {6} - {5}}});"
                "D = getresult(index_monitor,'index');"
                "M = matrix(length(D.x), length(D.y), length(D.z), length(D.f), 3);"
                "M(:,:,:,:,1) = D.index_x^2;"   
                "M(:,:,:,:,2) = D.index_y^2;"
                "M(:,:,:,:,3) = D.index_z^2;"
                "h5write('TempFileEps_{1}'+num2str(i)+'.h5','eps',M,'overwrite');")
        else:
            script_base = (
                "i = {0};"
                "load('TempFileMesh_{1}'+num2str(i)+'.fsp');"
                "index_monitor = '{2}';"
                "redrawoff;"
                "D = getresult(index_monitor,'index');"
                "M = matrix(length(D.x), length(D.y), length(D.z), length(D.f), 3);"
                "M(:,:,:,:,1) = D.index_x^2;"  
                "M(:,:,:,:,2) = D.index_y^2;"
                "M(:,:,:,:,3) = D.index_z^2;"
                "h5write('TempFileEps_{1}'+num2str(i)+'.h5','eps',M,'overwrite');"
                )

        if not hasattr(self, 'sources'):
            self.sources = []
            sim.fdtd.selectall()
            num_elements = int(sim.fdtd.getnumber())
            for i in range(num_elements):
                obj_type = sim.fdtd.get("type",i+1)
                if "Source" in obj_type:
                    self.sources.append(sim.fdtd.get("name",i+1))

        if not hasattr(self,'x_grid'):
            sim.fdtd.eval("index_data = getresult('opt_fields_index','index');"
            "x_grid=index_data.x;" +
            "y_grid=index_data.y;")
            self.x_grid = [x[0] for x in sim.fdtd.getv('x_grid')]
            self.y_grid = [y[0] for y in sim.fdtd.getv('y_grid')]
        self.bound_indices=[]

        ## If we don't use central differences, we need to calculate the current mesh
        if not self.use_central_differences or self.use_central_differences and self.obj_bounds:
            Geometry.get_eps_from_index_monitor(sim.fdtd, 'original_eps_data')

        ## Generate the various files and add them to the queue
        for i,param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + cur_dx
            self.add_geo(sim, d_params, only_update = True, param_id=i)

            filename = 'TempFileMesh_p{}'.format(i)
            sim.fdtd.save(filename)

            if i == 0:
                for source in self.sources:
                    sim.fdtd.select(source)
                    sim.fdtd.delete()
                    sim.fdtd.save(filename)

            if self.use_central_differences:                
                d_params[i] = param - cur_dx
                self.add_geo(sim, d_params, only_update = True, param_id=i)
                filename = 'TempFileMesh_m{}'.format(i)
                sim.fdtd.save(filename)

        if not hasattr(self, 'solver_path'):
            engine_path = sim.fdtd.getresource("FDTD",1,"solver executable")
            folder_path = os.path.split(engine_path)[0]
            if platform.system() == "Windows":
                self.solver_path = os.path.join(folder_path,'fdtd-solutions.exe')
            else:
                self.solver_path = os.path.join(folder_path,'fdtd-solutions')



        num_batches = len(current_params) // self.num_jobs + 1
        for k in range(num_batches):
            plist = []
            print(f"Starting batch {k+1}")
            for i in range(min(self.num_jobs,len(current_params)-k*self.num_jobs)):
                sfname = f"ms{i+1}.lsf"
                with open(sfname,"w") as hf:
                    if self.obj_bounds:
                        get_bounds = self.obj_bounds.get_bounds
                        param_dict=self.obj_bounds.param_dicts[i+k*self.num_jobs]
                        object_name = next(iter(param_dict.keys()))
                        property_names= list(next(iter(param_dict.values()))[1][0].keys())    
                        for property in property_names:
                            param_dict[object_name][1][0][property]=sim.fdtd.getnamed(object_name,property)
                
                        property_being_changed = param_dict[object_name][0][0]
                        param_dict_orig=copy.deepcopy(param_dict)
                        param_dict_orig[object_name][1][0][property_being_changed]-=cur_dx*self.obj_bounds.param_scaling
                        bounds=get_bounds(param_dict, param_dict_orig)

                        #use x_grid, y_grid to find indices of bounds
                        found_xmin, found_ymin = False, False
                        for j,val in enumerate(self.x_grid):
                            if val > bounds[0][0] and not found_xmin: 
                                x_min_index = max(j-1,0) #decrement by one since we necessarily passed the bound but not below 0
                                found_xmin = True
                            elif val > bounds[0][1]:
                                x_max_index = j; break

                        for j,val in enumerate(self.y_grid):
                            if val > bounds[1][0] and not found_ymin: 
                                y_min_index = max(j-1,0)
                                found_ymin = True
                            elif val > bounds[1][1]:
                                y_max_index = j; break
                        self.bound_indices.append([x_min_index, x_max_index, y_min_index, y_max_index])

                        hf.write(script_base.format(i+k*self.num_jobs, 
                                                    'p',
                                                    'opt_fields_index',
                                                    self.x_grid[x_min_index], 
                                                    self.x_grid[x_max_index], 
                                                    self.y_grid[y_min_index], 
                                                    self.y_grid[y_max_index]))
                        if self.use_central_differences:
                            hf.write(script_base.format(i+k*self.num_jobs, 
                                                    'm',
                                                    'opt_fields_index',
                                                    self.x_grid[x_min_index], 
                                                    self.x_grid[x_max_index], 
                                                    self.y_grid[y_min_index], 
                                                    self.y_grid[y_max_index]))
                    else:
                        hf.write(script_base.format(i+k*self.num_jobs, 'p', 'opt_fields_index'))       # apply any variables to the script template here
                        if self.use_central_differences:
                            hf.write(script_base.format(i+k*self.num_jobs, 'm', 'opt_fields_index'))       # apply any variables to the script template here

                # override the design environment threads setting with the command line argument
                plist.append(subprocess.Popen([self.solver_path, "-threads", str(self.threads_per_job), "-hide", "-run", sfname, "-exit"]))
               # plist.append(subprocess.Popen([self.solver_path, "-threads", str(self.threads_per_job), "-run", sfname]))

                print(f"Started {i+1}")
            for process in plist:
                print(".", end="")
                process.wait()

        sim.fdtd.eval("d_epses = cell({});".format(current_params.size))

        ## Load the various files, extract the mesh data 
        for i,param in enumerate(current_params):
            h5data = 'TempFileEps_p{}.h5'.format(i)
            if self.obj_bounds:
                x_min_index = self.bound_indices[i][0]+1 # +1 for index convention in .lsf
                x_max_index = self.bound_indices[i][1]+1
                y_min_index = self.bound_indices[i][2]+1
                y_max_index = self.bound_indices[i][3]+1

                sim.fdtd.eval(f"eps_data_updated = h5read('{h5data}','eps');"
                                "eps_data1 = original_eps_data;"
                                f"eps_data1({x_min_index}:{x_max_index}, {y_min_index}:{y_max_index}, :, :, :) = eps_data_updated;")
                
                if self.use_central_differences:
                    h5data = 'TempFileEps_m{}.h5'.format(i)
                    sim.fdtd.eval(f"eps_data_updated = h5read('{h5data}','eps');"
                                "eps_data2 = original_eps_data;"
                                f"eps_data2({x_min_index}:{x_max_index}, {y_min_index}:{y_max_index}, :, :, :) = eps_data_updated;"
                                "d_epses{"+str(i+1)+"} = (eps_data1 - eps_data2) / (2*dx);")
                else:
                    sim.fdtd.eval("d_epses{"+str(i+1)+"} = (eps_data1 - original_eps_data) / dx;")

            else:
                sim.fdtd.eval(f"eps_data1 = h5read('{h5data}','eps');")

                if self.use_central_differences:
                    h5data = 'TempFileEps_m{}.h5'.format(i)
                    sim.fdtd.eval(f"eps_data2 = h5read('{h5data}','eps');"
                                  "d_epses{"+str(i+1)+"} = (eps_data1 - eps_data2) / (2*dx);")
                else:
                    sim.fdtd.eval("d_epses{"+str(i+1)+"} = (eps_data1 - original_eps_data) / dx;")
            
            sys.stdout.write('.'), sys.stdout.flush()

        filename = 'adjoint_{}'.format(sim.iteration)
        sim.fdtd.load(filename)

        sim.fdtd.eval("clear(eps_data1, dx);")
        print('')
        if self.use_central_differences:
            sim.fdtd.eval("clear(eps_data2);")
        else:
            sim.fdtd.eval("clear(original_eps_data);")
        sim.fdtd.redrawon()

    def d_eps_on_cad_serial(self, sim):
        """
        Computes the derivative of the permittivity (epsilon) tensor using finite difference. The computation is
        performed serially, one parameter at a time. The results are stored in the cell array 'd_eps' in the lumerical 
        workspace.

        Attributes
        ----------
        current_params: The current parameter values for the geometry.
        obj_bounds: Indicates whether the simulation is restricted to specific bounds.
        use_central_differences: Determines whether central differences are used for finite difference calculations.
        dx: The step size for finite difference calculations.
        x_grid: The x-coordinates of the simulation grid (if bounds are applied).
        y_grid: The y-coordinates of the simulation grid (if bounds are applied).
        bound_indices: Stores the indices of the bounds in the simulation grid (if bounds are applied).

        Notes
        -----
        - If 'obj_bounds' is True, the method adjusts the index monitor's position and span based on the bounds.
        - If 'use_central_differences' is True, the method calculates the epsilon tensor for both positive 
          and negative perturbations of the parameter.
        - The method uses the 'get_eps_from_index_monitor' function to retrieve the epsilon tensor from the 
          simulation's index monitor.
        """

        sim.fdtd.redrawoff()
        Geometry.get_eps_from_index_monitor(sim.fdtd, 'original_eps_data')

        current_params = self.get_current_params()
        if self.obj_bounds:
            sim.fdtd.eval(
                        "index_data = getresult('opt_fields_index','index');"
                        "x_grid = index_data.x;" +
                        "y_grid = index_data.y;" +
                        "z_grid = index_data.z;" +
                        "d_epses = cell({});".format(current_params.size) +
                        "x_min_indices = zeros({});".format(current_params.size) +
                        "x_max_indices = zeros({});".format(current_params.size) +
                        "y_min_indices = zeros({});".format(current_params.size) +
                        "y_max_indices = zeros({});".format(current_params.size)
                        )
            self.x_grid = [x[0] for x in sim.fdtd.getv('x_grid')]
            self.y_grid = [y[0] for y in sim.fdtd.getv('y_grid')]
            self.bound_indices=[]
        else:
            sim.fdtd.eval("d_epses = cell({});".format(current_params.size))
        cur_dx = self.dx/2 if self.use_central_differences else self.dx
        lumapi.putDouble(sim.fdtd.handle, "dx", cur_dx)
        print('Getting d eps: dx = ' + str(cur_dx))

        for i,param in enumerate(current_params):

            d_params = current_params.copy()
            d_params[i] = param + cur_dx

            self.add_geo(sim, d_params, only_update = True, param_id=i)

            if self.obj_bounds:
                Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data', current_params, i, self.obj_bounds.get_bounds)
            else:
                Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data')
            if self.use_central_differences:
                d_params[i] = param - cur_dx
                self.add_geo(sim, d_params, only_update = True, param_id=i)
                if self.obj_bounds:
                    Geometry.get_eps_from_index_monitor(sim.fdtd, 'eps_data2', current_params, i, self.obj_bounds.get_bounds)
                    sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - eps_data2) / (2*dx);"
                              "x_min_indices("+str(i+1)+") = x_min_index;"
                              "x_max_indices("+str(i+1)+") = x_max_index;"
                              "y_min_indices("+str(i+1)+") = y_min_index;"
                              "y_max_indices("+str(i+1)+") = y_max_index;"
                              )    
                else:
                    Geometry.get_eps_from_index_monitor(sim.fdtd, 'eps_data2')
                sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - eps_data2) / (2*dx);") 

            elif not self.obj_bounds:
                sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - original_eps_data) / dx;")
            else:
                sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - original_eps_data(x_min_index:x_max_index, y_min_index:y_max_index,:,:,:)) / dx;"
                              "x_min_indices("+str(i+1)+") = x_min_index;"
                              "x_max_indices("+str(i+1)+") = x_max_index;"
                              "y_min_indices("+str(i+1)+") = y_min_index;"
                              "y_max_indices("+str(i+1)+") = y_max_index;"
                              )

            sys.stdout.write('.'), sys.stdout.flush()

        sim.fdtd.eval("clear(original_eps_data, current_eps_data, dx);")

        if self.use_central_differences:
            sim.fdtd.eval("clear(eps_data2);")
        sim.fdtd.redrawon()

    def d_eps_on_cad(self,sim):
        if self.num_jobs > 1:
            self.d_eps_on_cad_parallel(sim)
        else:
            self.d_eps_on_cad_serial(sim)
