import argparse
from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, **kwargs):
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='./store/models', dest='m_dir', help="default: %(default)s")
    parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
    parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir', help="default: %(default)s")
    return parser


def add_eval_options(parser, **kwargs):
    # evaluation parameters
    eval = parser.add_argument_group('Evaluation Parameters')
    eval.add_argument('--eval-n', type=int, default=0, help="number of test samples per class (0=all)")
    eval.add_argument('--eval-s', type=int, default=10, help="number of importance samples")
    eval.add_argument('--no-normal-eval', action='store_true', help="don't evaluate gen model with importance sampling")
    eval.add_argument('--from-replay', action='store_true', help="train classifier with replay from gen models")
    eval.add_argument('--replay-iters', type=int, default=2000, help="# of iters for training with replay")
    eval.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
    eval.add_argument('--loss-log', type=int, default=100, metavar="N", help="# iters after which to plot loss")
    eval.add_argument('--sample-log', type=int, metavar="N", help="# iters after which to plot samples")
    eval.add_argument('--sample-n', type=int, default=64, help="# images to show")
    eval.add_argument('--no-samples', action='store_true', help="don't plot generated images")
    eval.add_argument('--eval-tag', type=str, metavar="ETAG", default="e20N", help="tag for evaluation model")
    return parser


def add_task_options(parser, **kwargs):
    # benchmark parameters
    task_params = parser.add_argument_group('Benchmark Parameters')
    task_choices = ['CIFAR10', 'CIFAR100', 'CORe50', 'MNIST']
    task_params.add_argument('--experiment', type=str, default='MNIST', choices=task_choices)
    task_params.add_argument('--iters', type=int, help="# of iterations to optimize main model")
    task_params.add_argument('--single-epochs', action='store_true', help='single pass over data(replaces "--iters")')
    task_params.add_argument('--batch', type=int, default=None, help="batch-size")
    task_params.add_argument('--pre-convE', action='store_true', help="use pretrained convE-layers")
    task_params.add_argument('--convE-ltag', type=str, metavar='LTAG', default='s100N',
                              help="tag for loading convE-layers")
    task_params.add_argument('--augment', action='store_true',
                             help="augment training data (random crop & horizontal flip)")
    task_params.add_argument('--no-norm', action='store_false', dest='normalize',
                             help="don't normalize images (only for CIFAR)")
    return parser


def add_model_options(parser, **kwargs):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    model.add_argument('--hidden', action='store_true', help="learn generative classifier on latent features")
    # -conv-layers
    model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
    model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
    model.add_argument('--depth', type=int, help="# of convolutional layers (0 = only fc-layers)")
    model.add_argument('--reducing-layers', type=int, dest='rl',help="# of layers with stride (=image-size halved)")
    model.add_argument('--channels', type=int, default=16, help="# of channels 1st conv-layer (doubled every 'rl')")
    model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
    model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
    model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    model.add_argument('--z-dim', type=int, default=100, help='size of latent representation (def=100)')
    model.add_argument('--deconv-type', type=str, default="standard", choices=["standard", "resNet"])
    model.add_argument('--no-bn-dec', action='store_true', help="don't use batchnorm in decoder")
    model.add_argument('--prior', type=str, default="standard", choices=["standard", "vampprior", "GMM"])
    model.add_argument('--n-modes', type=int, default=1, help="how many modes for prior? (def=1)")
    return parser


def add_train_options(parser, **kwargs):
    # training hyperparameters / initialization
    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
    train_params.add_argument('--init-weight', type=str, default='standard', choices=['standard', 'xavier'])
    train_params.add_argument('--init-bias', type=str, default='standard', choices=['standard', 'constant'])
    train_params.add_argument('--freeze-convE', action='store_true', help="freeze parameters of convE-layers")
    # NOTE: when using hidden, the conv-layers are automatically frozen
    train_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    return parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args, **kwargs):

    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
    if hasattr(args, "recon_loss"):
        args.recon_loss = ("BCE" if args.experiment=="MNIST" else "MSE") if args.recon_loss is None else args.recon_loss
    args.batch = (128 if args.experiment in ('MNIST', 'CORe50') else 256) if args.batch is None else args.batch
    args.fc_units = (400 if args.experiment in ('MNIST', 'CORe50') else 2000) if args.fc_units is None else args.fc_units
    args.iters = (200 if args.experiment in ('MNIST', 'CORe50') else 500) if args.iters is None else args.iters
    # -for other unselected options, set default values (not specific to chosen scenario / experiment)
    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim
    if hasattr(args, "rl"):
        args.rl = args.depth-1 if args.rl is None else args.rl
    return args


def check_for_errors(args, **kwargs):
    if checkattr(args, "normalize") and hasattr(args, "recon_los") and args.recon_loss=="BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with normalized images")
    if checkattr(args, "hidden") and hasattr(args, "recon_los") and args.recon_loss=="BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with option '--hidden'")