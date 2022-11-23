#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
import main_cl
import utils



## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_fc': False, 'generative': True}
    # Define input options
    parser = options.define_args(filename="compare_all.py",
                                 description="Compare class-incremental learning methods on specified benchmark.")
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_options_for_comparison(parser, **kwargs)
    # Parse chosen options
    parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    parser.add_argument('--test-again', action='store_true', help='even if already run, test each method again')
    parser.add_argument('--no-bir', action='store_true', help="don't include brain-inspired replay")
    # -slda options
    parser.add_argument('--covariance', type=str, choices=["identity", "fixed", "streaming"], default="streaming",
                      help="what covariance matrix to use?")
    parser.add_argument('--pure-streaming', action='store_true')
    args = parser.parse_args()
    # Process (i.e., set defaults for unselected options) and check chosen options
    options.set_defaults(args, set_hyper_params=False, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/accTI-{}.txt'.format(args.r_dir, param_stamp)) and \
            os.path.isfile('{}/accCI-{}.txt'.format(args.r_dir, param_stamp)):
        if utils.checkattr(args, 'test_again'):
            print("\n ...testing: {} ...".format(param_stamp))
            args.train = True if utils.checkattr(args, 'slda') else False
            main_cl.run(args)
        else:
            print(" already run: {}".format(param_stamp))
    else:
        print("\n ...running: {} ...".format(param_stamp))
        args.train = True
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/accTI-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave_ti = float(file.readline())
    file.close()
    fileName = '{}/accCI-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave_ci = float(file.readline())
    file.close()
    # -return it
    return (ave_ti, ave_ci)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Set arguments that will vary between methods
    args.slda = False


    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ## Standard classifier, Class-IL training
    STANDARD = {}
    STANDARD = collect_all(STANDARD, seed_list, args, name="None")

    ## Standard classifier, JOINT training
    tasks_temp = args.tasks
    args.tasks = 1
    iters_temp = args.iters
    args.iters = tasks_temp*iters_temp
    JOINT = {}
    JOINT = collect_all(JOINT, seed_list, args, name="Joint")
    args.tasks = tasks_temp
    args.iters = iters_temp

    ## SI
    args.si = True
    omega_max_temp = args.omega_max
    args.omega_max = None
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False
    args.omega_max = omega_max_temp

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.ewc = False

    ## The 'labels trick'
    neg_samples_temp = args.neg_samples
    args.neg_samples = 'current'
    LABELS = {}
    LABELS = collect_all(LABELS, seed_list, args, name="The 'labels trick'")
    args.neg_samples = neg_samples_temp

    ## CWR
    args.cwr = True
    args.freeze_convE = True
    args.freeze_fcE = True
    args.freeze_after_first = True
    CWR = {}
    CWR = collect_all(CWR, seed_list, args, name="CWR")

    ## CWR+
    args.cwr_plus = True
    CWRP = {}
    CWRP = collect_all(CWRP, seed_list, args, name="CWR+")

    ## AR1
    args.freeze_after_first = False
    args.freeze_fcE = False
    args.freeze_convE = False
    args.si = True
    args.si_c = args.ar1_c
    args.reg_only_hidden = True
    AR1 = {}
    AR1 = collect_all(AR1, seed_list, args, name="AR1")
    args.cwr = False
    args.cwr_plus = False
    args.si = False
    args.reg_only_hidden = False

    ## SLDA
    args.slda = True
    fc_layers_temp = args.fc_lay
    args.fc_lay = 1
    if args.experiment=="CIFAR10":
        depth_temp = args.depth
        args.depth = 0
        gp_temp = args.gp
        args.gp = False
    SLDA = {}
    SLDA = collect_all(SLDA, seed_list, args, name="SLDA")
    args.fc_lay = fc_layers_temp
    args.slda = False
    if args.experiment=="CIFAR10":
        args.depth = depth_temp
        args.gp = gp_temp

    ## Generative replay
    args.replay = "generative"
    hidden_temp = args.hidden
    args.hidden = False
    GR = {}
    GR = collect_all(GR, seed_list, args, name="Generative Replay")
    args.replay = "none"
    args.hidden = hidden_temp

    ## Brain-inspired replay
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        args.replay = "generative"
        args.feedback = True
        args.dg_gates = True
        args.distill = True
        args.prior = "GMM"
        args.per_class = True
        BIR = {}
        BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay")

    # BI-R + SI
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        args.si = True
        args.si_c = args.bir_c
        args.omega_max = None
        args.dg_prop = args.si_dg_prop
        BIRSI = {}
        BIRSI = collect_all(BIRSI, seed_list, args, name="BI-R + SI")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    acc_CI = {}

    ## For each seed, create list with average metrics
    for seed in seed_list:
        i = 1
        acc_CI[seed] = [STANDARD[seed][i], JOINT[seed][i],
                        EWC[seed][i], SI[seed][i],
                        LABELS[seed][i], CWR[seed][i], CWRP[seed][i], AR1[seed][i],
                        SLDA[seed][i], GR[seed][i]]
        if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
            acc_CI[seed].append(BIR[seed][i])
            acc_CI[seed].append(BIRSI[seed][i])

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- PRINT TO SCREEN -----#
    #---------------------------#

    # select names / ids
    names = ["None      --   'lower bound'", "Joint     --   'upper bound'",
             "EWC", "SI",
             "The 'labels trick'", "CWR", "CWR+", "AR1",
             "SLDA", "Generative Replay"]
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        names.append("Brain-Inspired Replay")
        ids.append(10)
        names.append("BI-R + SI")
        ids.append(11)


    # EVALUATION ACCORDING TO CLASS-INCREMENTAL SCENARIO
    # -calculate averages and SEMs
    means = [np.mean([acc_CI[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([acc_CI[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]

    # -print results to screen
    classes = 100 if args.experiment=="CIFAR100" else 10
    print("\n\n"+"#"*78+"\n Split {}  --  CLASS-INCREMENTAL EVALUATION (i.e., {}-way classification):\n".format(
        args.experiment, classes
    )+"-"*78)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:46s} {:5.2f}  (+/- {:5.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:46s} {:5.2f}".format(name, 100*means[i]))
        if i==3 or i==8 or i==7 or i==1:
            print("-"*78)
    print("#"*78)
