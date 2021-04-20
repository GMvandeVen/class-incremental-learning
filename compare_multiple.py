#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options_gen_classifier as options
import main_generative



## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'generative': True, 'multiple': True}
    # Define input options
    parser = options.define_args(filename="compare_multiple", description='Train & test generative classifier.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args, gen_classifier=True)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/prec-{}--evalN{}-S{}.txt'.format(args.r_dir, param_stamp, args.eval_n, args.eval_s)):
        print(" already run: {}".format(param_stamp))
    elif os.path.isfile("{}/gC-{}".format(args.m_dir, param_stamp)):
        print("\n ...testing: {} ...".format(param_stamp))
        args.train = False
        args.from_replay = False
        main_generative.run(args, verbose=True)
    else:
        print("\n ...running: {} ...".format(param_stamp))
        args.train = True
        args.from_replay = False
        main_generative.run(args, verbose=True)
    # -get average precisions
    fileName = '{}/prec-{}--evalN{}-S{}.txt'.format(args.r_dir, param_stamp, args.eval_n, args.eval_s)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    file.close()
    # -return it
    return ave


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


    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ## Generative classifier
    GC = {}
    GC = collect_all(GC, seed_list, args, name="Generative Classifier")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    prec = {}
    prec_replay = {}

    ## For each seed, create list with average metrics
    for seed in seed_list:
        prec[seed] = [GC[seed]]



    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- PRINT TO SCREEN -----#
    #---------------------------#

    # select names / ids
    names = ["Generative Classifier"]
    ids = [0]

    # EVALUATION OF GENERATIVE CLASSIFIER
    # -calculate averages and SEMs
    means = [np.mean([prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]

    # -print results to screen
    classes = 10 if args.experiment in ("MNIST", "CORe50-category") else 100
    print("\n\n"+"#"*78+"\n  {}  --  GENERATIVE CLASSIFIER:.\n".format(args.experiment)+"-"*78)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:46s} {:5.2f}  (+/- {:5.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:46s} {:5.2f}".format(name, 100*means[i]))
        if i==5 or i==10 or i==9 or i==3:
            print("-"*78)
    print("#"*78)
