#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
import main_cl
import utils



## Parameter-values to compare
lambda_list = [0.1, 1., 10., 100., 1000., 10000., 100000., 1000000., 10000000.]
c_list = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000.]
omega_max_list = [0.0001, 0.001, 0.01, 0.1, 1.]
dg_prop_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dg_prop_si_list = [0., 0.2, 0.4, 0.6, 0.8]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True}
    # Define input options
    parser = options.define_args(filename="compare_hyperParams.py",
                                 description='Hyper-param gridsearch for SI, EWC, AR1 and BI-R.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_options_for_param_search(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    parser.add_argument('--test-again', action='store_true', help='even if already run, test each method again')
    parser.add_argument('--no-bir', action='store_true', help="don't run gridsearch for brain-inspired replay")
    args = parser.parse_args()
    options.set_defaults(args, set_hyper_params=False, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/precCI-{}.txt'.format(args.r_dir, param_stamp)):
        if utils.checkattr(args, 'test_again'):
            print("\n ...testing: {} ...".format(param_stamp))
            args.train = False
            main_cl.run(args)
        else:
            print(" already run: {}".format(param_stamp))
    else:
        print("\n ...running: {} ...".format(param_stamp))
        args.train = True
        main_cl.run(args)
    # -get average precision
    fileName = '{}/precCI-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()
    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)


    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    ## Baseline (Class-IL training)
    NONE = get_result(args)

    ## SI
    args.si = True
    SI = {}
    for si_c in c_list:
        args.si_c = si_c
        SI[si_c] = get_result(args)
    args.si = False

    ## EWC
    args.ewc = True
    EWC = {}
    for ewc_lambda in lambda_list:
        args.ewc_lambda = ewc_lambda
        EWC[ewc_lambda] = get_result(args)
    args.ewc = False

    ## CWR
    args.cwr = True
    args.freeze_fcE = True
    args.freeze_convE = True
    args.freeze_after_first = True
    CWR = get_result(args)

    ## CWR+
    args.cwr_plus = True
    CWRP = get_result(args)

    ## AR1 without SI (=CWR+, but no freezing)
    args.freeze_fcE = False
    args.freeze_convE = False
    args.freeze_after_first = False
    AR1_NOSI = get_result(args)

    ## AR1
    args.si = True
    args.reg_only_hidden = True
    AR1 = {}
    for omega_max in omega_max_list:
        AR1[omega_max] = {}
        args.omega_max = omega_max
        for si_c in c_list:
            args.si_c = si_c
            AR1[omega_max][si_c] = get_result(args)
    args.si = False
    args.reg_only_hidden = False
    args.cwr = False
    args.cwr_plus = False

    ## BI-R
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        args.replay = "generative"
        args.feedback = True
        args.dg_gates = True
        args.distill = True
        args.prior = "GMM"
        args.per_class = True
        BIR = {}
        for dg_prop in dg_prop_list:
            args.dg_prop = dg_prop
            BIR[dg_prop] = get_result(args)

    # BI-R + SI
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        args.si = True
        args.omega_max = None
        BIRSI = {}
        for dg_prop in dg_prop_si_list:
            args.dg_prop = dg_prop
            BIRSI[dg_prop] = {}
            for si_c in c_list:
                args.si_c = si_c
                BIRSI[dg_prop][si_c] = get_result(args)


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------#
    #----- COLLECT DATA AND PRINT TO SCREEN -----#
    #--------------------------------------------#

    ext_c_list = [0] + c_list
    ext_lambda_list = [0] + lambda_list
    print("\n")


    ###---SI---###
    # -collect data
    ave_prec_si = [NONE] + [SI[c] for c in c_list]
    # -print on screen
    print("\n\nSYNAPTIC INTELLIGENCE (SI)")
    print(" param list (si_c): {}".format(ext_c_list))
    print("  {}".format(ave_prec_si))
    print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_prec_si)], np.max(ave_prec_si)))


    ###---EWC---###
    # -collect data
    ave_prec_ewc = [NONE] + [EWC[ewc_lambda] for ewc_lambda in lambda_list]
    # -print on screen
    print("\n\nELASTIC WEIGHT CONSOLIDATION (EWC)")
    print(" param list (ewc_lambda): {}".format(ext_lambda_list))
    print("  {}".format(ave_prec_ewc))
    print("---> ewc_lambda = {}     --    {}".format(ext_lambda_list[np.argmax(ave_prec_ewc)], np.max(ave_prec_ewc)))


    ###---CWR---###
    print("\n\nCWR")
    print("--->  {}".format(CWR))


    ###---CWR+---###
    print("\n\nCWR+")
    print("--->  {}".format(CWRP))


    ###---AR1---###
    # -collect data
    ave_prec_per_omega = []
    for omega_max in omega_max_list:
        ave_prec_temp = [AR1_NOSI] + [AR1[omega_max][si_c] for si_c in c_list]
        ave_prec_per_omega.append(ave_prec_temp)
    # -print on screen
    if len(omega_max_list) > 0:
        print("\n\nAR1")
        print(" param-list (si_c): {}".format(ext_c_list))
        curr_max = 0
        for omega_max in omega_max_list:
            ave_prec_temp = [AR1_NOSI] + [AR1[omega_max][si_c] for si_c in c_list]
            print("  (omega_max={}):   {}".format(omega_max, ave_prec_temp))
            if np.max(ave_prec_temp) > curr_max:
                omega_max_max = omega_max
                c_max = ext_c_list[np.argmax(ave_prec_temp)]
                curr_max = np.max(ave_prec_temp)
        print("--->  omega_max = {}  -  si_c = {}     --    {}".format(omega_max_max, c_max, curr_max))


    ###---BI-R---###
    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        # -collect data
        ave_prec_bir = [BIR[dg_prop] for dg_prop in dg_prop_list]
        # -print on screen
        print("\n\nBRAIN-INSPIRED REPLAY (BI-R)")
        print(" param list (dg_prop): {}".format(dg_prop_list))
        print("  {}".format(ave_prec_bir))
        print("---> dg_prop = {}     --    {}".format(dg_prop_list[np.argmax(ave_prec_bir)], np.max(ave_prec_bir)))

    if args.experiment in ("CIFAR100", "CORe50") and not utils.checkattr(args, 'no_bir'):
        print("\n\nBI-R + SI")
        print(" param-list (si_c): {}".format(ext_c_list))
        curr_max = 0
        for dg_prop in dg_prop_si_list:
            ave_prec_temp = [BIR[dg_prop]] + [BIRSI[dg_prop][si_c] for si_c in c_list]
            print("  (dg_prop={}):   {}".format(dg_prop, ave_prec_temp))
            if np.max(ave_prec_temp) > curr_max:
                dg_prop_max = dg_prop
                c_max = ext_c_list[np.argmax(ave_prec_temp)]
                curr_max = np.max(ave_prec_temp)
        print("--->  dg_prop = {}  -  si_c = {}     --    {}".format(dg_prop_max, c_max, curr_max))

