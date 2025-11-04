import numpy as np
import inspect

from lumopt.geometries.geometry import Geometry

class ParameterizedGeometry(Geometry):
    """ 
        Defines a parametrized geometry using any of the built-in geometric structures available in the FDTD CAD.
        Users must provide a Python function with the signature ('params', 'fdtd', 'only_update'). The function
        must take the optimization parameters and a handle to the FDTD CAD to build the geometry under optimization
        (material assignments included). The flag 'only_update' is used to avoid frequent recreations of the parameterized
        geometry: when the flag is true, it is assumed that the geometry was already added at least once to the CAD.

        Parameters
        ----------
        :param func:           function with the signature ('params', 'fdtd', 'only_update', **kwargs).
        :param initial_params: flat array with the initial optimization parameter values.
        :param bounds:         bounding ranges (min/max pairs) for each optimization parameter.
        :param dx:             step size for computing the figure of merit gradient using permittivity perturbations.
        :threads_per_job:      number of threads used for each job during parallel meshing
        :num_jobs:             number of individual parallel meshing jobs performed concurrently
        :obj_bounds:           an optional object that provides dictionary of strings 'obj_bounds.param_dicts' that contains 
                               information for each object property that is modified and a function 'get_bounds' with a
                               signature of dictionaries for the object parameter information for the original and modified object.
    """
    
    def __init__(self, func, initial_params, bounds, dx, obj_bounds=None, threads_per_job=1, num_jobs=1):
        self.func = func
        self.current_params = np.array(initial_params).flatten()
        self.bounds = bounds
        self.dx = float(dx)
        self.obj_bounds = obj_bounds                
        self.threads_per_job = threads_per_job
        self.num_jobs = num_jobs

        if inspect.isfunction(self.func):
            bound_args = inspect.signature(self.func).bind('params', 'fdtd', 'only_update')
            if bound_args.args != ('params', 'fdtd', 'only_update'):
                raise UserWarning("user defined function does not take three positional arguments.")
        else:
            raise UserWarning("argument 'func' must be a Python function.")
        if self.dx <= 0.0:
            raise UserWarning("step size must be positive.")

        self.params_hist = list(self.current_params)

    def update_geometry(self, params, sim):
        self.current_params = params
        self.params_hist.append(params)

    def get_current_params(self):
        return self.current_params

    def calculate_gradients(self, gradient_fields):
        raise UserWarning("unsupported gradient calculation method.")

    def add_geo(self, sim, params, only_update, param_id):
        """
        Adds or updates the geometry in the simulation based on the provided parameters.

        This method calls the user-defined geometry function (`self.func`) to add or update 
        the geometry in the simulation. It ensures that the simulation is in layout mode 
        if `only_update` is False and passes the required arguments to the geometry function.

        Parameters
        ----------
        sim: The simulation object used to execute Lumerical FDTD commands.
        params: The parameter values for the geometry. If None, the current parameters (`self.current_params`) are used.
        param_id: The ID of the parameter being updated, passed to the geometry function if it accepts `param_id`.
        only_update: If True, only updates the geometry without switching to layout mode.

        Notes
        -----
        - The user-defined geometry function (`self.func`) must accept at least three positional arguments: 
          `params`, `fdtd`, and `only_update`.
        - The geometry function accepts additional arguments `obj_bounds`, they are 
          passed as keyword arguments.
        - If `only_update` is False, the simulation is switched to layout mode before adding the geometry.
        
        """
        if not only_update:
            sim.fdtd.switchtolayout()

        func_signature = inspect.signature(self.func)
        func_params = func_signature.parameters

        args = [params if params is not None else self.current_params, sim.fdtd, only_update]
        kwargs = {}

        if 'param_id' in func_params:
            kwargs['param_id'] = param_id
        if 'obj_bounds' in func_params:
            kwargs['obj_bounds'] = self.obj_bounds

        return self.func(*args, **kwargs)