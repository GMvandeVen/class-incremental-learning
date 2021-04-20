#!/usr/bin/env python3
import numpy as np
import os
import copy
import torch
from torch import optim

# -custom-written libraries
import options_gen_classifier as options
import utils
import define_models as define
from data.load import get_experiment
from eval import evaluate
from eval import callbacks as cb
from train import train_gen_classifiers, train_from_gen
from param_stamp import get_param_stamp_gen_classifier
from data.manipulate import FeatureDataset
import visual.plt as my_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Define input options
    parser = options.define_args(filename="main_generative", description='Train & test generative classifier.')
    parser = options.add_general_options(parser)
    parser = options.add_eval_options(parser)
    parser = options.add_task_options(parser)
    parser = options.add_model_options(parser)
    parser = options.add_train_options(parser)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args)
    options.check_for_errors(args)
    return args


## Function for running one continual learning experiment
def run(args, verbose=False):

    # Create results- and plotting-directory, if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it and exit
    if args.get_stamp:
        from param_stamp import get_param_stamp_from_args
        print(get_param_stamp_from_args(args=args, gen_classifier=True))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config = get_experiment(
        name=args.experiment, data_dir=args.d_dir,
        normalize=True if utils.checkattr(args, "normalize") else False,
        augment=True if utils.checkattr(args, "augment") else False,
        verbose=verbose, exception=True if args.seed<10 else False,
        per_class=True,
    )


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------#
    #----- FEATURE EXTRACTOR -----#
    #-----------------------------#

    # Define the feature extractor
    depth = args.depth if hasattr(args, 'depth') else 0
    if utils.checkattr(args, 'hidden'):
        if verbose and utils.checkattr(args, "pre_convE"):
            print("\nDefining the feature extractor...")
        # - define the feature extractor
        feature_extractor = define.define_feature_extractor(args=args, config=config, device=device)
        feature_extractor_name = feature_extractor.name if depth>0 else None
        # - initialize (pre-trained) parameters
        feature_extractor = define.init_params(feature_extractor, args)
        # - freeze the parameters & set model to eval()-mode
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        # - reset size and # of channels to reflect the extracted features rather than the original images
        config = config.copy()  # -> make a copy to avoid overwriting info in the original config-file
        config['size'] = feature_extractor.conv_out_size
        config['channels'] = feature_extractor.conv_out_channels
        depth = 0
    else:
        feature_extractor = feature_extractor_name = None

    # Print characteristics of feature extractor on the screen
    if verbose and feature_extractor is not None:
        utils.print_model_info(feature_extractor, title="FEATURE EXTRACTOR")

    # Convert original data to features (so this doesn't need to be done at run-time)
    # (Note: augmentation can not be used with this!)
    if (feature_extractor is not None) and args.depth>0:
        if verbose:
            print("\nPutting the data through the feature extractor...")
        new_train_datasets = []
        new_test_datasets = []
        for class_id in range(config['classes']):
            # -training data
            if args.train:
                loader = utils.get_data_loader(train_datasets[class_id], batch_size=args.batch, drop_last=False, cuda=cuda)
                all_of_this_class = torch.empty((len(loader.dataset), config['channels'], config['size'], config['size']))
                #--> pre-allocate a large tensor, which will be filled slice-by-slice
                count = 0
                for x, _ in loader:
                    x = feature_extractor(x.to(device)).cpu()
                    all_of_this_class[count:(count+x.shape[0])] = x
                    count += x.shape[0]
                new_train_datasets.append(FeatureDataset(all_of_this_class, class_id))
            # -testing data
            loader = utils.get_data_loader(test_datasets[class_id], batch_size=args.batch, drop_last=False, cuda=cuda)
            all_of_this_class = torch.empty((len(loader.dataset), config['channels'], config['size'], config['size']))
            count = 0
            for x, _ in loader:
                x = feature_extractor(x.to(device)).cpu()
                all_of_this_class[count:(count+x.shape[0])] = x
                count += x.shape[0]
            new_test_datasets.append(FeatureDataset(all_of_this_class, class_id))
        # Reset datasets, and set feature extractor to None (as no longer needed now!)
        train_datasets = new_train_datasets
        test_datasets = new_test_datasets
        feature_extractor = None


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- MAIN MODEL -----#
    #----------------------#

    # Define main model
    if verbose:
        print("\nDefining the model...")
    model = define.define_gen_classifer(args=args, config=config, device=device, depth=depth)

    # Separately initialize and set optimizer for each VAE
    for class_id in range(config['classes']):
        current_model = getattr(model, 'vae{}'.format(class_id))
        # - initialize (pre-trained) parameters
        current_model = define.init_params(current_model, args)
        # - freeze weights of conv-layers?
        if utils.checkattr(args, "freeze_convE"):
            for param in current_model.convE.parameters():
                param.requires_grad = False
            current_model.convE.frozen = True  # --> needed to ensure batchnorm-layers also do not change
        # - define optimizer (only optimize parameters that "requires_grad")
        current_model.optim_list = [
            {'params': filter(lambda p: p.requires_grad, current_model.parameters()), 'lr': args.lr},
        ]
        current_model.optimizer = optim.Adam(current_model.optim_list, betas=(0.9, 0.999))


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp_gen_classifier(args, model.get_name(), feature_extractor_name=feature_extractor_name,
                                                 verbose=verbose)

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model.vae0, title="MAIN MODEL (x{})".format(config['classes']))

    # Prepare for plotting in visdom
    visdom = None
    if args.visdom:
        env_name = args.experiment
        graph_name = "gen_classifier"
        visdom = {'env': env_name, 'graph': graph_name}


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Determine after how many iterations to plot samples from the model (default=after each class)
    sample_log = args.sample_log if (args.sample_log is not None) else args.iters

    # Callbacks for reporting on and visualizing loss
    loss_cbs = [
        cb._gen_classifier_loss_cb(log=args.loss_log, classes=config['classes'], visdom=visdom)
    ]

    # Callbacks for plotting generated samples
    no_samples = (utils.checkattr(args, "no_samples") or feature_extractor is not None)
    sample_cbs = [
        cb._sample_cb(log=sample_log, visdom=visdom, config=config, sample_size=args.sample_n)
    ] if not no_samples else [None]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        train_gen_classifiers(model, train_datasets, iters=args.iters, epochs=1 if args.single_epochs else None,
                              batch_size=args.batch, feature_extractor=feature_extractor,
                              loss_cbs=loss_cbs, sample_cbs=sample_cbs)
        # Save trained model(s), if requested
        if args.save:
            save_name = "gC-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)

    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of previously trained model...")
        load_name = "gC-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- EVALUATION of CLASSIFIER-----#
    #-----------------------------------#

    if not utils.checkattr(args, 'no_normal_eval'):
        if verbose:
            print("\n\nEVALUATION RESULTS:")

        # Evaluate precision of final model on full test-set
        if verbose:
            print("\n Precision on test-set:")
        precs = []
        for i in range(config['classes']):
            prec = evaluate.validate(model, test_datasets[i], verbose=False, allowed_classes=None, S=args.eval_s,
                                     feature_extractor=feature_extractor,
                                     test_size=None if args.eval_n==0 else args.eval_n)
            if verbose:
                print(" - Class {}: {:.4f}".format(i + 1, prec))
            precs.append(prec)
        average_precs = sum(precs)/config['classes']
        if verbose:
            print('=> Average precision over all {} classes: {:.4f}\n'.format(config['classes'], average_precs))
        # -write out to text file
        output_file = open("{}/prec-{}--evalN{}-S{}.txt".format(args.r_dir, param_stamp, args.eval_n, args.eval_s), 'w')
        output_file.write('{}\n'.format(average_precs))
        output_file.close()


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------------------------#
    #----- USE GENERATIVE MODELS TO TRAIN DISCRIMINATIVE CLASSIFIER -----#
    #--------------------------------------------------------------------#

    if utils.checkattr(args, 'from_replay'):
        # Specify discriminative model
        args_copy = copy.deepcopy(args)  # -> make a copy to avoid overwriting original args-object
        if verbose and utils.checkattr(args, "pre_convE") and depth>0:
            print("\nDefining the discriminative model...")
        if args.experiment=="MNIST":
            args_copy.fc_units = args_copy.h_dim = 400
            args_copy.fc_lay = 3
        elif args.experiment=="CIFAR10":
            args_copy.fc_lay = 1
            depth = 5
            args_copy.rl = 3
            args_copy.conv_type = "resNet"
            args_copy.gp = True
            args_copy.channels = 20
        elif args.experiment=="CIFAR100":
            args_copy.fc_units = args_copy.h_dim = 2000
            args_copy.fc_lay = 3
        elif args.experiment=="CORe50":
            args_copy.fc_units = args_copy.h_dim = 1024
            args_copy.fc_lay = 2
        cnn = define.define_classifier(args=args_copy, config=config, device=device, depth=depth)
        cnn = define.init_params(cnn, args_copy)
        optim_list = [{'params': filter(lambda p: p.requires_grad, cnn.parameters()), 'lr': args.lr}]
        cnn.optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        # -print info of discriminative classifier to be trained on generated samples
        if verbose:
            utils.print_model_info(cnn, title="DISCRIMINATIVE CLASSIFIER")

        # Train the discriminative model on generated samples
        iters = args.replay_iters
        loss_cbs = [cb._loss_cb(log=args.loss_log, visdom=None)]
        train_from_gen(model=cnn, gen_model=model, iters=iters, batch_size=args.batch, loss_cbs=loss_cbs)

        # Evaluate discriminative model
        if verbose:
            print("\n Precision on test-set (discriminative model trained on generated samples, iters={}):".format(iters))
        precs = []
        for i in range(config['classes']):
            prec = evaluate.validate(cnn, test_datasets[i], verbose=False, feature_extractor=feature_extractor,
                                     allowed_classes=None, test_size=None)
            if verbose:
                print(" - Class {}: {:.4f}".format(i + 1, prec))
            precs.append(prec)
        average_precs = sum(precs)/config['classes']
        if verbose:
            print('=> Average precision over all {} classes: {:.4f}\n'.format(config['classes'], average_precs))
        # -write out to text file
        output_file = open("{}/precReplay-{}--i{}.txt".format(args.r_dir, param_stamp, iters), 'w')
        output_file.write('{}\n'.format(average_precs))
        output_file.close()


    #-------------------------------------------------------------------------------------------------#

    #-------------------------------------#
    #----- PLOT SAMPLES of GENERATOR -----#
    #-------------------------------------#

    if args.experiment in ("CIFAR10", "MNIST"):
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = my_plt.open_pdf(plot_name)
        # -plot samples
        for class_id in range(config["classes"]):
            evaluate.show_samples(model, config, pdf=pp, visdom=None, size=100,
                                  title="Generated samples (class_id={})".format(class_id), class_id=class_id)
        # -close pdf
        pp.close()
        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))


    #-------------------------------------------------------------------------------------------------#



if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)