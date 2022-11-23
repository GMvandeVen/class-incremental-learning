import argparse
from utils import checkattr

##-------------------------------------------------------------------------------------------------------------------##

# Where to store the data / results / models / plots
store = "./store"

##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, single_task=False, only_fc=True, **kwargs):
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    if single_task and (not only_fc):
        parser.add_argument('--convE-stag', type=str, metavar='STAG', default='none',help="tag for saving convE-layers")
        parser.add_argument('--fcE-stag', type=str, metavar='STAG', default='none', help="tag for saving fcE-layers")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    if not single_task:
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--data-dir', type=str, default='{}/datasets'.format(store), dest='d_dir',
                        help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='{}/models'.format(store), dest='m_dir',
                        help="default: %(default)s")
    if not single_task:
        parser.add_argument('--plot-dir', type=str, default='{}/plots'.format(store), dest='p_dir',
                            help="default: %(default)s")
        parser.add_argument('--results-dir', type=str, default='{}/results'.format(store), dest='r_dir',
                            help="default: %(default)s")
    return parser


def add_eval_options(parser, single_task=False, generative=False, **kwargs):
    # evaluation parameters
    eval = parser.add_argument_group('Evaluation Parameters')
    if (not single_task):
        eval.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
        eval.add_argument('--pdf', action='store_true', help="generate pdf with plots for individual experiment(s)")
    eval.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    if not single_task:
        eval.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
    eval.add_argument('--loss-log', type=int, default=500, metavar="N", help="# iters after which to plot loss")
    if generative:
        eval.add_argument('--sample-log', type=int, metavar="N", help="# iters after which to plot samples")
        eval.add_argument('--sample-n', type=int, default=64, help="# images to show")
        eval.add_argument('--no-samples', action='store_true', help="don't plot generated images")
    eval.add_argument('--acc-log', type=int, default=None if single_task else 500, metavar="N",
                      help="# iters after which to plot accuracy")
    eval.add_argument('--acc-n', type=int, default=1024, help="# samples for evaluating accuracy (visdom-plots)")
    return parser


def add_task_options(parser, only_fc=False, single_task=False, **kwargs):
    # benchmark parameters
    task_params = parser.add_argument_group('Benchmark Parameters')
    task_choices = ['MNIST'] if only_fc else ['CIFAR100', 'CIFAR10', 'MNIST', 'CORe50']
    task_default = 'MNIST'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=task_choices)
    task_params.add_argument('--offline', action='store_true', help="always train on all data so far")
    if not single_task:
        task_params.add_argument('--tasks', type=int, default=None, help='number of tasks')
    if not single_task:
        task_params.add_argument('--iters', type=int, help="# of iterations to optimize main model")
        task_params.add_argument('--single-epochs', action='store_true',
                                  help='single pass over data (replaces "--iters")')
    else:
        iter_epochs = task_params.add_mutually_exclusive_group(required=False)
        iter_epochs.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='# of epochs (default: %(default)d)')
        iter_epochs.add_argument('--iters', type=int, metavar='N', help='# of iterations (replaces "--epochs")')
    task_params.add_argument('--batch', type=int, default=256 if single_task else None, help="batch-size")
    if not only_fc:
        task_params.add_argument('--pre-convE', action='store_true', help="use pretrained convE-layers")
        task_params.add_argument('--convE-ltag', type=str, metavar='LTAG', default='s100N',
                                  help="tag for loading convE-layers")
    if not only_fc:
        task_params.add_argument('--augment', action='store_true',
                                 help="augment training data (random crop & horizontal flip)")
        task_params.add_argument('--no-norm', action='store_false', dest='normalize',
                                 help="don't normalize images (only for CIFAR)")
    return parser


def add_model_options(parser, only_fc=False, single_task=False, generative=False, **kwargs):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    if not only_fc:
        # -conv-layers
        model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
        model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
        model.add_argument('--depth', type=int, default=5 if single_task else None,
                           help="# of convolutional layers (0 = only fc-layers)")
        model.add_argument('--reducing-layers', type=int, dest='rl',help="# of layers with stride (=image-size halved)")
        model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
        model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
        model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
        model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, default=2000 if single_task else None, metavar="N",
                       help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    if generative:
        model.add_argument('--z-dim', type=int, default=100, help='size of latent representation (def=100)')
        model.add_argument('--prior', type=str, default="standard", choices=["standard", "vampprior", "GMM"])
        model.add_argument('--n-modes', type=int, default=1, help="how many modes for prior? (def=1)")
        if not only_fc:
            model.add_argument('--deconv-type', type=str, default="standard", choices=["standard", "resNet"])
    return parser


def add_slda_options(parser, **kwargs):
    # parameters specific for Streaming LDA
    slda = parser.add_argument_group('SLDA Parameters')
    slda.add_argument('--slda', action='store_true', help="use SLDA")
    slda.add_argument('--covariance', type=str, choices=["identity", "fixed", "streaming", "pure_streaming"],
                      default="streaming", help="what covariance matrix to use?")
    return parser


def add_train_options(parser, only_fc=False, single_task=False, generative=False, **kwargs):
    # training settings / initialization
    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--neg-samples', type=str, default='all-so-far',
                              choices=["all-so-far", "all", "current", "single-from-batch"],
                              help="how to select negative samples?")
    #--> The above command controls which output units will be set to "active" (the active classes can also
    #    be thought of as 'negative samples', see Li et al., 2020, https://arxiv.org/abs/2011.12216):
    #    - "all-so-far":        the output units of all classes seen so far are set to active
    #    - "all":               always the output units of all classes are set to active
    #    - "current":           only output units of the classes in the current 'task' (or 'episode') are set to active
    #    - "single-from-batch": only a single other output unit (randomly selected from current batch) is set to active
    train_params.add_argument('--lr', type=float, default=0.0001 if single_task else None, help="learning rate")
    if not single_task:
        train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')
    train_params.add_argument('--init-weight', type=str, default='standard', choices=['standard', 'xavier'])
    train_params.add_argument('--init-bias', type=str, default='standard', choices=['standard', 'constant'])
    if not single_task:
        train_params.add_argument('--reinit', action='store_true', help='reinitialize networks before each new task')
    if not only_fc:
        train_params.add_argument('--freeze-convE', action='store_true', help="freeze convE-layers")
    train_params.add_argument('--freeze-fcE', action='store_true', help='freeze fcE-layers')
    if not single_task:
        train_params.add_argument('--freeze-after-first', action='store_true',
                                  help='freeze specified layers only after 1st epoch')
    if generative:
        train_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    train_params.add_argument('--hidden', action='store_true', help='conv layers are fixed feature extractor')
    train_params.add_argument('--train-on-first', action='store_true', help="train feature extractor on first task")
    return parser


def add_replay_options(parser, **kwargs):
    replay = parser.add_argument_group('Replay Parameters')
    replay_choices = ['generative', 'none', 'current']
    replay.add_argument('--replay', type=str, default='none', choices=replay_choices)
    replay.add_argument('--distill', action='store_true', help="use distillation for replay")
    replay.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    # -options specific for 'brain-inspired replay'
    replay.add_argument('--brain-inspired', action='store_true', help="select defaults for brain-inspired replay")
    replay.add_argument('--feedback', action="store_true", help="equip main model with feedback connections")
    replay.add_argument('--pred-weight', type=float, default=1., dest='pl', help="(FB) weight of prediction loss (def=1)")
    replay.add_argument('--classify', type=str, default="beforeZ", choices=['beforeZ', 'fromZ'])
    replay.add_argument('--per-class', action='store_true', help="if selected, each class has own modes")
    replay.add_argument('--dg-gates', action='store_true', help="use class-specific gates in decoder")
    replay.add_argument('--dg-prop', type=float, help="decoder-gates: masking-prop")
    return parser


def add_regularization_options(parser,  **kwargs):
    cl = parser.add_argument_group('Options relating to EWC / SI')
    cl.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
    cl.add_argument('--lambda', type=float, dest="ewc_lambda",help="--> EWC: regularisation strength")
    cl.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
    cl.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
    cl.add_argument('--fisher-n', type=int, default=1000, help="--> EWC: sample size estimating Fisher Information")
    cl.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
    cl.add_argument('--c', type=float, dest="si_c", help="-->  SI: regularisation strength")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    cl.add_argument('--omega-max', type=float, help="-->  SI: max penalty for any parameter")
    cl.add_argument('--reg-only-hidden', action='store_true', help="use EWC and/or SI only on hidden layers")
    return  parser

def add_bias_correcting_options(parser, **kwargs):
    bc = parser.add_argument_group('Options for bias-correcting')
    bc.add_argument('--cwr', action='store_true', help="use 'CWR' (Lomonaco and Maltoni, 2017)")
    bc.add_argument('--cwr-plus', action='store_true', help="use 'CWR+' (Maltoni and Lomonaco, 2019)")
    return parser


##-------------------------------------------------------------------------------------------------------------------##



def add_options_for_comparison(parser,  **kwargs):
    cl = parser.add_argument_group('Parameters for SI / CWR / AR1')
    cl.add_argument('--fisher-n', type=int, default=1000, help="--> EWC: sample size estimating Fisher Information")
    cl.add_argument('--lambda', type=float, dest="ewc_lambda",help="--> EWC: regularisation strength")
    cl.add_argument('--c', type=float, dest="si_c", help="-->  SI: regularisation strength")
    cl.add_argument('--ar1-c', type=float, dest="ar1_c", help="-->  AR1: regularisation strength for SI")
    cl.add_argument('--bir-c', type=float, dest="bir_c", help="-->  BI-R + SI: regularisation strength for SI")
    cl.add_argument('--si-dg-prop', type=float, dest="si_dg_prop", help="-->  BI-R + SI: gating prop")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    cl.add_argument('--omega-max', type=float, help="-->  SI: max penalty for any parameter")
    return  parser

def add_options_for_param_search(parser,  **kwargs):
    cl = parser.add_argument_group('Parameters for SI / AR1 / Replay')
    cl.add_argument('--fisher-n', type=int, default=1000, help="--> EWC: sample size estimating Fisher Information")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    cl.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    cl.add_argument('--pred-weight', type=float, default=1., dest='pl', help="(FB) weight of prediction loss (def=1)")
    cl.add_argument('--classify', type=str, default="beforeZ", choices=['beforeZ', 'fromZ'])
    return  parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args, set_hyper_params=True, single_task=False, no_boundaries=False, **kwargs):
    # -if 'brain-inspired' is selected, select corresponding defaults
    if checkattr(args, 'brain_inspired'):
        if hasattr(args, "replay") and not args.replay=="generative":
            raise Warning("To run with brain-inspired replay, select both '--brain-inspired' and '--replay=generative'")
        args.feedback = True     #--> replay-through-feedback
        args.prior = 'GMM'       #--> conditional replay
        args.per_class = True    #--> conditional replay
        args.dg_gates = True     #--> gating based on internal context (has hyper-param 'dg_prop')
        args.hidden = True       #--> internal replay
        args.pre_convE = True    #--> internal replay
        args.distill = True      #--> distillation
    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.augment = args.augment if args.experiment in ('CIFAR10', 'CIFAR100') else False
    if hasattr(args, "depth"):
        args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
    if hasattr(args, 'recon_loss'):
        args.recon_loss = ('BCE' if args.experiment=="MNIST" else 'MSE') if args.recon_loss is None else args.recon_loss
    if not single_task:
        args.tasks= (
            5 if args.experiment in ('MNIST', 'CIFAR10', 'CORe50') else 10
        ) if args.tasks is None else args.tasks
        if hasattr(args, 'iters'):
            args.iters = 2000 if args.iters is None else args.iters
        args.lr = (0.0001 if args.experiment=='CIFAR100' else 0.001) if args.lr is None else args.lr
        args.batch = (128 if args.experiment=='MNIST' else 256) if args.batch is None else args.batch
        args.fc_units = (
            400 if args.experiment=='MNIST' else (2000 if args.experiment=='CIFAR100' else 1000)
        ) if args.fc_units is None else args.fc_units
    # -set hyper-parameter values (typically found by grid-search) based on chosen experiment
    if set_hyper_params and (not single_task) and (not no_boundaries):
        if args.experiment=='MNIST':
            args.si_c = 0.1 if args.si_c is None else args.si_c
            args.ewc_lambda = 100000. if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
        elif args.experiment=='CIFAR100':
            args.si_c = 1. if args.si_c is None else args.si_c
            args.ewc_lambda = 1. if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1 if args.gamma is None else args.gamma
    # -for other unselected options, set default values (not specific to chosen experiment)
    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim
    if hasattr(args, "rl"):
        args.rl = args.depth-1 if args.rl is None else args.rl
    # -if [log_per_task], reset all logs
    if checkattr(args, 'log_per_task'):
        args.acc_log = args.iters
        args.loss_log = args.iters
    return args


def check_for_errors(args, **kwargs):
    if checkattr(args, "normalize") and hasattr(args, "recon_los") and args.recon_loss=="BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with normalized images")