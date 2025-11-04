#(c) 2025 ANSYS, Inc. Unauthorized use, distribution, or duplication is prohibited.
import glob
import json
import math
import os
import random
import re
import socket
import subprocess
import time
import warnings

import lumapi

#default config
license_priority_hpc_pack = True
mpirun = 'mpirun'
mpilib = None
fdtd_engine = '/opt/lumerical/v242/bin/fdtd-engine-ompi-lcl'
fdtd_gui = '/opt/lumerical/v242/bin/fdtd-solutions'
pythonpath = '/opt/lumerical/v242/api/python'
python = '/opt/lumerical/v242/python/bin/python'

#read system and user config
config_files = [f'{os.path.dirname(__file__)}/lumslurm.config',
                os.path.expanduser('~/.lumslurm.config')]

for config_file in config_files:
    try:
        with open(config_file, 'r', encoding='utf-8') as cf:
            config = json.load(cf)
    except FileNotFoundError:
        config = None

    if config:
        for varname in ['mpirun', 'mpilib', 'fdtd_engine', 'fdtd_gui', 'pythonpath', 'python']:
            if varname in config:
                globals()[varname] = config[varname]

def fdtd_memory_estimate(filename):
    """
    Estimates the memory requirements for an FDTD simulation

    This runs the FDTD solver on a project file to perform
    the memory estimate.

    Parameters:
        filename (string): name of fsp file, possibly including a path

    Returns:
        dict: a dict with fields memory, gridpoints and time_steps.
        Memory field is the memory requirement in bytes
    """

    args = [fdtd_engine, '-mr', filename]
    try:
        p = subprocess.run(args, capture_output=True, encoding='utf-8', check=True)
        p.check_returncode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(fdtd_engine + ' failed with: ' + p.stderr) from e

    report = {}
    for line in p.stdout.split():
        x = line.split('=')
        if len(x) == 2:
            report[x[0]] = float(x[1])

    report['memory'] = report['memory']*1e6

    return report

def partition_info():
    """
    Query slurm for information about partitions

    Returns:
        dict: key is partition name, value is a dict with
        partition details
    """

    args = ['sinfo', '--json']
    try:
        p = subprocess.run(args, capture_output=True, encoding='utf-8', check=True)
        p.check_returncode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError('sinfo command failed with: ' + p.stderr) from e

    info = json.loads(p.stdout)
    partitions = dict()

    if 'sinfo' in info:
        parts = info['sinfo']
        for part in parts:
            specs = {'cpu': part['cpus']['maximum'], 'memory': part['memory']['maximum']*1e6}

            gres = part['gres']['total'].split(":", 2)
            if len(gres) >=2 and gres[0] == 'gpu':
                m = re.search('[0-9]+', gres[-1])
                if m:
                    specs['gpu'] = int(m[0])
                    if len(gres) > 2:
                        specs['gpu model'] = gres[1]
                else:
                    specs['gpu'] = 0
            else:
                specs['gpu'] = 0

            pname = part['partition']['name']
            if pname in partitions:
                for k in ['cpu', 'memory', 'gpu']:
                    if specs[k] > partitions[pname][k]:
                        partitions[pname][k] = specs[k]
                if 'gpu model' in specs:
                    partitions[pname]['gpu model'] = specs['gpu model']
            else:
                partitions[pname] = specs
    else:
        raise RuntimeError('sinfo output not in the expected format.'
			+' You will need to manually specify the number of threads and processes')

    return partitions

def suggest_partitions(fsp_file):
    """
    Suggest slurm partitions for solving fsp_file

    This function selects the slurm paritions with
    nodes with an optimal number of CPUs for the
    solve job. The solve job's memory requirements
    are estimated to determine the number of CPUs
    to use.

    Suggested partitions will all contain nodes with the same
    number of CPU
                    
    Parameters:
        fsp_file (sting): name of the fsp file, possibly
        including the path
                         
    Returns:
        string: comma separated list of slurm partitions.
    """

    req_mem = fdtd_memory_estimate(fsp_file)['memory']
    thread_mid = math.ceil(req_mem/100e6)
    thread_high = math.floor(req_mem/50e6)

    partitions = partition_info()

    potentials = list()
    for partition, info in partitions.items():

        if req_mem > info['memory']:
            continue
        if info.get('gpu') > 0:
            continue

        cpu = info['cpu']
        distance = abs(cpu-thread_mid)
        rank = 1 if cpu <= thread_high else 2

        potentials.append({'partition': partition,
                           'rank': rank,
                           'distance': distance,
                           'cpu': cpu})

    potentials.sort(key=lambda val: val['distance'])

    for r in [1, 2]:
        semifinalists = [x for x in potentials if x['rank'] == r]
        if len(semifinalists) > 0:
            lowest_distance = semifinalists[0]['distance']
            finalists = [x for x in semifinalists if x['distance'] == lowest_distance]
            finalists.sort(reverse=True, key=lambda val: val['cpu'])
            highest_cpu = finalists[0]['cpu']
            winners = [x['partition'] for x in finalists if x['cpu'] == highest_cpu]
            return ",".join(winners)

def slurm_license_exists(feature):
    """
    Check if the license feautre exists on the slurm server
    """

    args = ['scontrol', 'show', 'lic']
    p = subprocess.run(args, capture_output=True, encoding='utf-8', check=False)
    p.check_returncode()
    return f'LicenseName={feature}\n' in p.stdout

def get_hpc_count_license_estimation(cores, use_optislang = False, estimated_capacity = None):
    """
    Compute the number of HPC licenses needed for a job give the number of cores
    """
    capacity = 1 if estimated_capacity is None else estimated_capacity
    parallel = capacity * (cores - 4)
    if parallel < 0:
        parallel = 0

    optislang_lic_used = 1 if use_optislang else 0
    parametric = (capacity - optislang_lic_used) - 1
    if parametric < 0:
        parametric = 0

    # compute `HPC count` and `HPC Pack count`
    hpc_count = (8 * parametric) + parallel
    if not license_priority_hpc_pack:
        return hpc_count

    hpc_pack_count = 0
    if parallel > 0:
        if parallel > 8:
            hpc_pack_count = math.ceil(math.log(parallel / 2.0) / math.log(4.0))
        else:
            hpc_pack_count = 1

    return hpc_pack_count + parametric

def get_license_estimation(cores, is_license_standard=None, estimated_capacity=None):
    """
    Compute the number of licenses needed for a job given the number of cores 
    for enterprise or standard licenses. 

    : param cores: number of cores to use for the job
    : param is_license_standard: True if the license is standard, False if it is enterprise
    : param estimated_capacity: estimated concurrent job capacity. 
        This is only used for enterprise licenses
    : return: string with the estimated number of license features needed for the job
    """

    if is_license_standard is None:
        print('Warning: is_license_standard is not set. Opening FDTD to check')
        fdtd = lumapi.FDTD(serverArgs = {'hide':True, 'use-solve':True})
        is_license_standard = fdtd.islicensestandard() == 'true'
    if is_license_standard:
        license_units = math.ceil(cores/32)
        return f'lum_fdtd_solve:{license_units}'

    license_units = get_hpc_count_license_estimation(cores, estimated_capacity=estimated_capacity)
    hpc_type = 'anshpc_pack' if license_priority_hpc_pack else 'anshpc'
    return f'{hpc_type}:{license_units},lumerical_solve:1'

def sbatch(command,
           partition=None,
           nodes=1,
           ntasks_per_node=None,
           cpus_per_task=None,
           gpus_per_node=None,
           dependency=None,
           licenses=None,
           name=None,
           block=False):
    """
    Run a job using slurm's sbatch
    """

    #Remove licenses that Slurm does not know about
    licenses_filtered = []
    for license_item in licenses.split(','):
        feature, _ = license_item.split(':',2)
        if slurm_license_exists(feature):
            licenses_filtered.append(license_item)
        else:
            warnings.warn(f'Ignored license feature {feature} that Slurm does not recognize')

    args = [
            'sbatch',
            f'--nodes={nodes}'
           ]
    if partition:
        args.append(f'--partition={partition}')
    if ntasks_per_node:
        args.append(f'--ntasks-per-node={ntasks_per_node}')
    if cpus_per_task:
        args.append(f'--cpus-per-task={cpus_per_task}')
    if gpus_per_node:
        args.append(f'--gpus-per-node={gpus_per_node}')
    if dependency:
        args.append(f'--dependency={dependency}')
    if licenses_filtered:
        licenses = ','.join(licenses_filtered)
        args.append(f"--licenses={licenses}")
    if name:
        args.append(f'--job-name={name}')
    if block:
        args.append('--wait')

    script = '#!/bin/bash -x\n' + command
    try:
        p = subprocess.run(args, capture_output=True, input=script, encoding='utf-8', check=True)
        p.check_returncode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError('sbatch command failed with: ' + p.stderr) from e

    m = re.match('Submitted batch job ([0-9]+)\n', p.stdout)
    if not m:
        raise RuntimeError('sbatch returned unexpected output')

    return m[1]

def run_solve(fsp_file,
              partition='auto',
              nodes=1,
              processes_per_node='auto',
              threads_per_process='auto',
              gpus_per_node='auto',
              is_license_standard=None,
              shared_solve_license=None,
              block=False):
    """
    Run an FDTD solve job using slurm

    Parameters:
        fsp_file (string): name of fsp file, possibly including path
        partition (string, optional): name of slurm partition to run
            job on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    pinfo = []
    if partition == 'auto':
        partition = suggest_partitions(fsp_file)
        if not partition:
            raise RuntimeError('Could not automatically select a partition.'
                            +' You will need to manually specify the partition.')

    if gpus_per_node=='auto' or processes_per_node=='auto' or threads_per_process=='auto':
        first_partition = partition.split(",")[0]
        pinfo = partition_info()[first_partition]

    if gpus_per_node=='auto':
        gpus_per_node = pinfo['gpu'] if pinfo['gpu'] > 0 else None

    if gpus_per_node and gpus_per_node > 0:
        if nodes > 1:
            warnings.warn('GPU solver does not support more than 1 node. Setting nodes=1')
        nodes = 1

        if processes_per_node == 'auto':
            processes_per_node = 1
        if processes_per_node > 1:
            warnings.warn('GPU solver does not support more than 1 process per node. '
                        + 'Setting processes_per_node=1')
        processes_per_node = 1

        if threads_per_process == 'auto':
            threads_per_process = gpus_per_node * pinfo['cpu'] / pinfo['gpu']

    else:
        if processes_per_node == 'auto' and threads_per_process == 'auto':
            cpu = pinfo['cpu']
            if nodes == 1:
                processes_per_node = 1
                threads_per_process = cpu
            else:
                processes_per_node = cpu
                threads_per_process = 1
        elif processes_per_node == 'auto':
            cpu = pinfo['cpu']
            processes_per_node = round(cpu/threads_per_process)
        elif threads_per_process == 'auto':
            cpu = pinfo['cpu']
            threads_per_process = round(cpu/processes_per_node)

    args = [
            'FI_EFA_FORK_SAFE=1',
            mpirun,
            '--map-by socket',
            '--bind-to socket',
           ]
    if "intel" not in mpirun:
        args.append('--report-bindings')
    args.append(fdtd_engine)
    args.append(f'-t {threads_per_process}')
    if shared_solve_license is not None:
        args.append(f'-shared-solve-license {shared_solve_license}')
    if mpilib:
        args.insert(0, f"LD_LIBRARY_PATH={mpilib}")
    if gpus_per_node:
        args.append('-gpu')
    args.append(fsp_file)
    licenses = get_license_estimation(nodes*processes_per_node*threads_per_process,
                                      is_license_standard)

    jobid = sbatch(' '.join(args),
                   partition,
                   nodes,
                   processes_per_node,
                   threads_per_process,
                   gpus_per_node=gpus_per_node,
                   licenses=licenses,
                   name=f"S:{fsp_file}",
                   block=block)
    return jobid

def run_lum_script(script_file,
                   fsp_file=None,
                   partition=None,
                   threads='auto',
                   dependency=None,
                   job_name=None,
                   is_license_standard=None,
                   block=False):
    """
    Run a Lumerical script using slurm
    """

    if threads == 'auto':
        if partition:
            threads = partition_info()[partition]['cpu']
        else:
            threads = 1

    args = [
            fdtd_gui,
            f'-threads {threads}',
            f'-run {script_file}',
            '-use-solve',
            '-exit'
           ]

    if fsp_file:
        args.append(fsp_file)

    licenses = get_license_estimation(threads, is_license_standard)
    if not job_name:
        job_name = fsp_file

    jobid = sbatch(' '.join(args),
                   partition,
                   cpus_per_task=threads,
                   dependency=dependency,
                   licenses=licenses,
                   name=job_name,
                   block=block)
    return jobid

def run_py_script(py_file,
                  data_file=None,
                  partition=None,
                  threads='auto',
                  dependency=None,
                  job_name=None,
                  args=None,
                  is_license_standard=None,
                  block=False):
    """
    Run a Python script using slurm
    """
    if args is None:
        args = []

    if threads == 'auto':
        if partition:
            threads = partition_info()[partition]['cpu']
        else:
            threads = 1

    cl = [
            f'PYTHONPATH={pythonpath}',
            python,
            py_file,
            str(threads)
           ]

    if data_file:
        cl.append(data_file)
    for arg in args:
        cl.append(str(arg))

    licenses = get_license_estimation(threads, is_license_standard)
    if not job_name:
        job_name = data_file

    return sbatch(' '.join(cl),
                  partition,
                  cpus_per_task=threads,
                  dependency=dependency,
                  licenses=licenses,
                  name=job_name,
                  block=block)

def run_py_code(py_code,
                fsp_file=None,
                partition=None,
                threads='auto',
                dependency=None,
                job_name=None,
                is_license_standard=None,
                block=False):
    """
    Run a Python code block using slurm
    """

    if threads == 'auto':
        if partition:
            threads = partition_info()[partition]['cpu']
        else:
            threads = 1

    py_code = py_code.replace('\n', '\\n')
    args = [
            f'PYTHONPATH={pythonpath}',
            python,
            '-c',
            f"$'{py_code}'",
            str(threads)
           ]

    if fsp_file:
        args.append(fsp_file)

    licenses = get_license_estimation(threads, is_license_standard)
    if not job_name:
        job_name = fsp_file

    return sbatch(' '.join(args),
                  partition,
                  cpus_per_task=threads,
                  dependency=dependency,
                  licenses=licenses,
                  name=job_name,
                  block=block)

def run_script(script_file=None,
               script_code=None,
               fsp_file=None,
               partition=None,
               threads='auto',
               dependency=None,
               job_name=None,
               is_license_standard=None,
               block=False):
    """
    Run a script job using slurm

    A script job can be Lumerical script or Python. For Python, code can
    be passed as a string or a file. For Lumerical script, only files
    are supported.

    Parameters:
        script_file (string, optional): Filename of script possibly
            including path. If not specified then it is assumed you
            will supply the script_code parameter. Filename should 
            end with lsf or py extension
        script_code (string, optional): string containing Python code.
            If not specified it is assumes you will supply the
            script_file parameter
        fsp_file (string): name of fsp file, possibly including path.
            The fsp file will be passed as the second command line
            argument for Python. The fsp file will be loaded before
            the lsf script is run
        partition (string, optional): name of slurm partition to run
            job on. Multiple partitions allowed as comma separated list.
            Default value of None will result in slurms's default 
            partition being selected
        threads (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        dependency (string, optional): ID or IDs of other slurm job(s) that
            must complete before this job will run. If none supplied then
            there is no dependency
        job_name (string, optional): Name to use for slurm job. This will be
            displayed in squeue output
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default value of false will queue job and return
                         
    Returns:
        string: slurm job ID for the script job
    """
    if script_file:
        if script_file.endswith('.py'):
            return run_py_script(script_file, fsp_file, partition, threads,
                                 dependency, job_name, is_license_standard, block)
        else:
            return run_lum_script(script_file, fsp_file, partition, threads,
                                  dependency, job_name, is_license_standard, block)
    else:
        return run_py_code(script_code, fsp_file, partition, threads,
                           dependency, job_name, is_license_standard, block)

def run_solve_and_script(fsp_file,
                         script_file=None,
                         script_code=None,
                         solve_partition='auto',
                         solve_nodes=1,
                         solve_processes_per_node='auto',
                         solve_threads_per_process='auto',
                         solve_gpus_per_node=None,
                         script_partition=None,
                         script_threads='auto',
                         is_license_standard=None,
                         shared_solve_license=None,
                         block=False):
    """
    Run a solve job, then a script job to process 
    the solve output
    """

    solveid = run_solve(fsp_file,
                        solve_partition,
                        solve_nodes,
                        solve_processes_per_node,
                        solve_threads_per_process,
                        solve_gpus_per_node,
                        is_license_standard,
                        shared_solve_license)

    return run_script(script_file,
                      script_code,
                      fsp_file,
                      script_partition,
                      script_threads,
                      "afterok:"+solveid,
                      f"P:{fsp_file}",
                      is_license_standard,
                      block)

def run_batch(fsp_file_pattern,
              postprocess_script=None,
              postprocess_code=None,
              collect_script=None,
              collect_code=None,
              solve_partition='auto',
              solve_nodes=1,
              solve_processes_per_node='auto',
              solve_threads_per_process='auto',
              solve_gpus_per_node=None,
              script_partition=None,
              script_threads='auto',
              job_name=None,
              is_license_standard=None,
              shared_solve_license=None,
              block=True):
    """
    Run a set of FDTD solve jobs with optional post-processing scripts

    Post-processing scripts or code are run after every solve job. The
    fsp file is supplied as a command line argument (Python) or loaded
    before script runs (lsf).

    The optional collect script can be run when all solve and
    post-processing script jobs have been completed.

    Solve jobs and script jobs can run on different slurm partitions.

    For solve jobs you can configure distributed solves with nodes>1.
    You can easily set the number of processes per node and number of
    threads per process. If no values provided the function will choose
    good default values.

    Parameters:
        fsp_file_pattern (string): pattern for fsp file name,
            possibly including path. Pattern follows glob syntax
        postprocess_script (string, optional): Filename of post-process
            script possibly including path. Post-process script is run
            after every solve job. Filename should end with lsf or py
            extension.
        postprocess_code (string, optional): string containing Python code
            for post-process script.
        collect_script (string, optional): Filename of collect
            script possibly including path. Collect script is run
            after every solve job and post-process script job has completed.
            Filename should end with lsf or py extension.
        collect_code (string, optional): string containing Python code for
            collect script.
        solve_partition (string, optional): name of slurm partition to run
            solve job on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        solve_nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        solve_processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        solve_threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        script_partition (string, optional): name of slurm partition to run
            script jobs on. Multiple partitions allowed as comma separated list.
            Default value of None will result in default slurm partition being selected
        script_threads (int, optional): number of threads to use for each script process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        job_name (string, optional): Name to use for slurm job. This will be
            displayed in squeue output
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    fsp_files = glob.glob(fsp_file_pattern)

    jobids = list()
    for fsp_file in fsp_files:
        jobid = run_solve_and_script(fsp_file,
                                     postprocess_script,
                                     postprocess_code,
                                     solve_partition,
                                     solve_nodes,
                                     solve_processes_per_node,
                                     solve_threads_per_process,
                                     solve_gpus_per_node,
                                     script_partition,
                                     script_threads,
                                     is_license_standard,
                                     shared_solve_license
                                     )
        jobids.append(jobid)

    dependency = 'afterok:' + ':'.join(jobids)
    if not job_name:
        job_name = fsp_file_pattern

    if collect_script:
        return run_script(script_file=collect_script,
                          partition=script_partition,
                          threads=script_threads,
                          dependency=dependency,
                          job_name=f"C:{job_name}",
                          is_license_standard=is_license_standard,
                          block=block)
    elif collect_code:
        return run_script(script_code=collect_code,
                          partition=script_partition,
                          threads=script_threads,
                          dependency=dependency,
                          job_name=f"C:{job_name}",
                          is_license_standard=is_license_standard,
                          block=block)
    else:
        return jobids

def compute_results(fsp_file, sweep_name, threads):
    """
    Helper function to compute all the results needed
    for a sweep.

    Used as post-processing script for sweeps
    """

    print(fsp_file)
    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {
                           'hide':True,
                           'use-solve':True,
                           'threads': f'{threads}'})

    print(sweep_name)
    props = fdtd.getsweep(sweep_name).split('\n')

    for prop in props:
        val = fdtd.getsweep(sweep_name, prop)
        if isinstance(val,dict) and 'result' in val.keys():
            result_path = val['result'].split("::")
            result_name = result_path.pop()
            result_object = "::".join(result_path)

            print(f'{result_object}, {result_name}')
            fdtd.getresult(result_object, result_name)

    fdtd.save()

def load_sweep(fsp_file, sweep_name, threads):
    """
    Helper function to run the loadsweep command.

    Used as the collection script when running sweeps
    """

    print(f"loadsweep: {fsp_file}|{sweep_name}")
    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {
                           'hide':True,
                           'use-solve':True,
                           'threads': f'{threads}'})

    top_sweep = sweep_name.split("::")[0]
    fdtd.loadsweep(top_sweep)
    fdtd.save()

def run_sweep(fsp_file,
              sweep_name,
              solve_partition='auto',
              solve_nodes=1,
              solve_processes_per_node='auto',
              solve_threads_per_process='auto',
              solve_gpus_per_node=None,
              script_partition=None,
              script_threads='auto',
              block=True):
    """
    Run an FDTD sweep or nested sweeps as independent jobs for solves
    and result computation

    Each solve job for a parameter value is run as a slurm job. Solve
    jobs can be distributed or single node.
    
    Results are computed as a post-processing step for each solve job
    in their own slurm job. This greatly accelerates heavy post-
    processing like far field projections
    
    Sweep results are collected by a final slurm job that runs when all
    the solve and post-processing jobs have finished.

    Solve jobs and script jobs can run on different slurm partitions.

    For solve jobs you can configure distributed solves with nodes>1.
    You can easily set the number of processes per node and number of
    threads per process. If no values provided the function will choose
    good default values.

    Parameters:
        fsp_file (string): fsp file name, possibly including path.
        sweep_name (string): name of the sweep in fsp file that you
            will run (solve). For nested sweeps include the full
            hierarchy with :: separators. Eg outersweep::innersweep
        solve_partition (string, optional): name of slurm partition to run
            solve jobs on. Multiple partitions allowed as comma separated list.
            Default value of 'auto' will result in partition being selected
            automatically. See suggest_partition()
        solve_nodes (int, optional): number of nodes for distributed solve.
            Default value of 1 runs solve on a single node
        solve_processes_per_node (int, optional): number of processes to run on each node.
            Default value 'auto' will automatically determine the 
            number of processes based on available CPU and number of 
            threads_per_process (if specified)
        solve_threads_per_process (int, optional): number of threads to use for each process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU and the number of
            processes_per_node (if specified)
        script_partition (string, optional): name of slurm partition to run
            script jobs on (result computations and data collection).
            Multiple partitions allowed as comma separated list.
            Default value of None will result in default slurm partition being selected
        script_threads (int, optional): number of threads to use for each script process.
            Default value of 'auto' will automatically determine the 
            number of threads based on available CPU
        block (boolean, optional): Wait until solve job completes before returning
            from this function. Default is true. Value of false will queue a job
                         
    Returns:
        string: slurm job ID for the solve job
    """

    fdtd = lumapi.FDTD(filename = fsp_file,
                       serverArgs = {'hide':True,
                                     'keepCADOpened':True})

    top_sweep = sweep_name.split("::")[0]
    fdtd.savesweep(top_sweep)
    fsp_file_pattern = f'{os.path.splitext(fsp_file)[0]}_{top_sweep}/*.fsp'

    shared_solve_license = None
    is_license_standard = fdtd.islicensestandard() == 'true'
    if not is_license_standard:
        shared_solve_license = fdtd.gethpcsharingcontext()
    print(f'License standard is set to: {is_license_standard}')

    postprocess_code = '\n'.join([
            'import lumslurm',
            'import sys',
            f'lumslurm.compute_results(sys.argv[2], "{sweep_name}", int(sys.argv[1]))'
            ])

    collect_code = '\n'.join([
            'import lumslurm',
            'import sys',
            f'lumslurm.load_sweep("{fsp_file}", "{sweep_name}", int(sys.argv[1]))'
            ])

    result = run_batch(fsp_file_pattern,
                     postprocess_code=postprocess_code,
                     collect_code=collect_code,
                     solve_partition=solve_partition,
                     solve_nodes=solve_nodes,
                     solve_processes_per_node=solve_processes_per_node,
                     solve_threads_per_process=solve_threads_per_process,
                     solve_gpus_per_node=solve_gpus_per_node,
                     script_partition=script_partition,
                     script_threads=script_threads,
                     job_name=fsp_file,
                     is_license_standard=is_license_standard,
                     shared_solve_license=shared_solve_license,
                     block=block)
    fdtd.close()
    return result

def launch_notebook_server(partition=None, threads='auto'):
    """
    Launch a Jupyter notebook server on slurm
    """

    if threads == 'auto':
        if partition:
            threads = partition_info()[partition]['cpu']
        else:
            threads = 1

    d = random.randint(10, 99)

    script = f"""vncserver :{d} -geometry 1920x1080
    DISPLAY=:{d} jupyter notebook --ip=0.0.0.0
    vncserver -kill :{d}"""

    job = sbatch(script,
                 partition,
                 cpus_per_task=threads,
                 name='notebook',
                 block=False
                )

    print('Waiting for server to start...')
    filename = f'slurm-{job}.out'
    while not os.path.exists(filename):
        time.sleep(1)

    print('Waiting for notebook to start...')
    e = re.compile(r'\s+http://([\w-]+):8888/\?token\=(\w+)')
    while True:
        with open(filename, encoding='utf-8') as f:
            contents = f.read()
            m = e.search(contents)
            if m:
                host = m[1]
                token = m[2]
                break
        time.sleep(1)

    ip = socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP)[0][4][0]
    url = f'http://{ip}:8888/?token={token}'
    print('Server running at:')
    print(url)
