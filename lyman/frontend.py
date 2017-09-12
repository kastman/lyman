"""Forward facing lyman tools with information about ecosystem."""
import os
import re
import sys
import imp
import os.path as op

import numpy as np
import pandas as pd


def gather_project_info():
    """Import project information based on environment settings."""
    lyman_dir = os.environ["LYMAN_DIR"]
    proj_file = op.join(lyman_dir, "project.py")
    try:
        project = sys.modules["project"]
    except KeyError:
        project = imp.load_source("project", proj_file)

    project_dict = dict()
    for dir in ["data", "analysis", "working", "crash"]:
        path = op.abspath(op.join(lyman_dir, getattr(project, dir + "_dir")))
        project_dict[dir + "_dir"] = path
    project_dict["default_exp"] = project.default_exp
    project_dict["rm_working_dir"] = project.rm_working_dir

    if hasattr(project, "ants_normalization"):
        use_ants = project.ants_normalization
        project_dict["normalization"] = "ants" if use_ants else "fsl"
    else:
        project_dict["normalization"] = "fsl"

    return project_dict


def gather_experiment_info(exp_name=None, altmodel=None, args=None):
    """Import an experiment module and add some formatted information."""
    lyman_dir = os.environ["LYMAN_DIR"]

    # Allow easy use of default experiment
    if exp_name is None:
        project = gather_project_info()
        exp_name = project["default_exp"]

    # Import the base experiment
    try:
        exp = sys.modules[exp_name]
    except KeyError:
        exp_file = op.join(lyman_dir, exp_name + ".py")
        exp = imp.load_source(exp_name, exp_file)

    exp_dict = default_experiment_parameters()

    def keep(k):
        return not re.match("__.*__", k)

    exp_dict.update({k: v for k, v in exp.__dict__.items() if keep(k)})

    # Possibly import the alternate model details
    if altmodel is not None:
        try:
            alt = sys.modules[altmodel]
        except KeyError:
            alt_file = op.join(lyman_dir, "%s-%s.py" % (exp_name, altmodel))
            alt = imp.load_source(altmodel, alt_file)

        alt_dict = {k: v for k, v in alt.__dict__.items() if keep(k)}

        # Update the base information with the altmodel info
        exp_dict.update(alt_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = "" if exp.__doc__ is None else exp.__doc__
    if altmodel is not None:
        exp_dict["comments"] += "\n"
        exp_dict["comments"] += "" if alt.__doc__ is None else alt.__doc__

    # Check if it looks like this is a partial FOV acquisition
    exp_dict["partial_brain"] = bool(exp_dict.get("whole_brain_template"))

    # Temporal resolution. Mandatory.
    exp_dict["TR"] = float(exp_dict["TR"])

    # Set up the default contrasts
    if ("contrasts_for_conditions" in exp_dict.keys() and
            exp_dict["contrasts_for_conditions"] and
            exp_dict["condition_names"] is not None):
        cs = [(name, [name], [1]) for name in exp_dict["condition_names"]]
        exp_dict["contrasts"] = cs + exp_dict["contrasts"]

    # Build contrasts list if neccesary
    exp_dict["contrast_names"] = [c[0] for c in exp_dict["contrasts"]]

    # Add command line arguments for reproducibility
    if args is not None:
        exp_dict["command_line"] = vars(args)

    return exp_dict


def default_experiment_parameters():
    """Return default values for experiments."""
    exp = dict(

        source_template="",
        whole_brain_template="",
        fieldmap_template="",
        n_runs=0,

        TR=2,
        frames_to_toss=0,
        fieldmap_pe=("y", "y-"),
        temporal_interp=False,
        interleaved=True,
        coreg_init="fsl",
        slice_order="up",
        intensity_threshold=4.5,
        motion_threshold=1,
        spike_threshold=None,
        wm_components=6,
        smooth_fwhm=6,
        hpf_cutoff=128,

        design_name=None,
        condition_names=None,
        regressor_file=None,
        regressor_names=None,
        confound_sources=["motion"],
        remove_artifacts=True,
        hrf_model="GammaDifferenceHRF",
        temporal_deriv=False,
        confound_pca=False,
        hrf_params={},
        contrasts=[],
        memory_request=5,

        flame_mode="flame1",
        cluster_zthresh=2.3,
        grf_pthresh=0.05,
        peak_distance=30,
        surf_name="inflated",
        surf_smooth=5,
        sampling_units="frac",
        sampling_method="average",
        sampling_range=(0, 1, .1),
        surf_corr_sign="pos",

        )

    return exp


def determine_subjects(subject_arg=None):
    """Intelligently find a list of subjects in a variety of ways."""
    if subject_arg is None:
        subject_file = op.join(os.environ["LYMAN_DIR"], "subjects.txt")
        subjects = np.loadtxt(subject_file, str).tolist()
    elif op.isfile(subject_arg[0]):
        subjects = np.loadtxt(subject_arg[0], str).tolist()
    else:
        try:
            subject_file = op.join(os.environ["LYMAN_DIR"],
                                   subject_arg[0] + ".txt")
            subjects = np.loadtxt(subject_file, str).tolist()
        except IOError:
            subjects = subject_arg
    return subjects


def determine_covars(covars_file=None):
    if not covars_file:
        covars_file = op.join(os.environ['LYMAN_DIR'], 'group_covariates.txt')
    return pd.read_table(covars_file, index_col='subject')


def check_missing_regressors(regressor_names, all_columns):
    missing_regressors = []
    for name in regressor_names:
        if name not in all_columns:
            missing_regressors.append(name)
    if len(missing_regressors):
        raise StandardError("Regressor(s) %s not found in group covariates "
                            "file. Please check the name." %
                            " ".join(missing_regressors))


def build_regressor(all_covars,
                    regressor_names,
                    subject_list,
                    mean_center='adjust'):
    regressors = dict()
    for s in subject_list:
        if s not in all_covars.index:
            raise ValueError('Subject %s not in group_covariates list' % s)

    for name, vals in all_covars.loc[subject_list, regressor_names].iteritems(
    ):
        values = vals.values.astype(float)  # Cast to float for division.
        # import pdb; pdb.set_trace()
        if not np.allclose(values.mean(), 0):
            msg = 'Covariate %s was not mean-centered - ' % name
            if mean_center == 'adjust':
                values -= values.mean()
                msg += 'mean-centering for you'
            elif mean_center == 'warn':
                msg += '- FSL does not fix this for you, probably invalid!'
            elif mean_center == 'raise':
                raise StandardError(msg)
        regressors[name] = values.tolist()
    return regressors


def group_models(group_contrasts,
                 all_covars,
                 subject_list,
                 add_group_mean=True):
    '''Create a dictionary of group models to run.

    Top level dictionary key is model name,values are 'design' and 'contrasts'
    * Design is a dictionary, where keys are column names and values are X data
      for the design matrix
    * Contrasts is a list of standard 3-part lyman (4-part nipype) contrast
      tuples
    '''
    models = {}
    if add_group_mean:  # 1 Sample T Special Case
        models['group_mean'] = dict()
        regressors = dict(group_mean=[1] * len(subject_list))
        contrasts = [["group_mean", "T", ["group_mean"], [1]]]
        groups = [1] * len(subject_list)
        models['group_mean']['design'] = regressors
        models['group_mean']['contrasts'] = contrasts
        models['group_mean']['groups'] = groups
    # import pdb; pdb.set_trace()
    for con_i, con in enumerate(group_contrasts):
        if len(con) < 3:
            raise StandardError(
                'Bad group contrast (%d) should have names and weights: %s' %
                (con_i, con))
        con_name, regressor_names, con_weights = con[0], con[1], con[2]
        if len(con) >= 4:
            group_col = con[3]  # Column name for group variance
            con = con[:3]
        else:
            group_col = None
        models[con_name] = dict()
        check_missing_regressors(regressor_names, all_covars.columns)
        regressors = build_regressor(all_covars,
                                     regressor_names,
                                     subject_list,
                                     mean_center='warn')
        models[con_name]['design'] = regressors
        full_con = list(con)  # [c.insert('T', 1) for c in con]
        full_con.insert(1, 'T')
        full_con = tuple(full_con)
        models[con_name]['contrasts'] = [full_con]

        if group_col:
            groups = all_covars.loc[subject_list, group_col].astype(
                int).tolist()
        else:
            groups = [1] * len(subject_list)
        models[con_name]['groups'] = groups

    return models


def lookup_model(group_regression_models, group_regression_name):
    model = group_regression_models[group_regression_name]
    return model['design'], model['contrasts'], model['groups']


def determine_engine(args):
    """Read command line args and return Workflow.run() args."""
    plugin_dict = dict(linear="Linear",
                       multiproc="MultiProc",
                       ipython="IPython",
                       torque="PBS",
                       sge="SGE",
                       slurm="SLURM")

    plugin = plugin_dict[args.plugin]

    plugin_args = dict()
    qsub_args, sbatch_args = "", ""

    if plugin == "MultiProc":
        plugin_args['n_procs'] = args.nprocs
    elif plugin in ["SGE", "PBS"]:
        qsub_args += "-V -e /dev/null -o /dev/null "
        if args.queue is not None:
            qsub_args += "-q %s " % args.queue
        plugin_args["qsub_args"] = qsub_args
    elif plugin in ["SLURM"]:
        sbatch_args += '-N 1 '  # Run all cores on a single node.
        if args.queue is not None:
            sbatch_args += "-p %s" % args.queue
        plugin_args["sbatch_args"] = sbatch_args

    return plugin, plugin_args


def run_workflow(wf, name=None, args=None):
    """Run a workflow, if we asked to do so on the command line."""
    plugin, plugin_args = determine_engine(args)
    if (name is None or name in args.workflows) and not args.dontrun:
        wf.run(plugin, plugin_args)
