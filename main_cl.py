#!/usr/bin/env python3
import numpy as np
import os
import torch
from torch import optim
from torch.utils.data import TensorDataset

# -custom-written libraries
import options
import utils
import train
import define_models as define
from data.load import get_experiment
from eval import evaluate
from eval import callbacks as cb
from param_stamp import get_param_stamp
from models.cl.continual_learner import ContinualLearner
from visual import plt as visual_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_fc': False, 'generative': True}
    # Define input options
    parser = options.define_args(filename="main_cl", description='...')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_slda_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_regularization_options(parser, **kwargs)
    parser = options.add_bias_correcting_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


## Function for running one continual learning experiment
def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if utils.checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it and exit
    if args.get_stamp:
        from param_stamp import get_param_stamp_from_args
        print(get_param_stamp_from_args(args=args))
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
        name=args.experiment, tasks=args.tasks, data_dir=args.d_dir,
        normalize=True if utils.checkattr(args, "normalize") else False,
        augment=True if utils.checkattr(args, "augment") else False,
        verbose=verbose, exception=True if args.seed<10 else False,
        per_class=False,
    )
    classes_per_task = config['classes_per_task']


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------#
    #----- FEATURE EXTRACTOR -----#
    #-----------------------------#

    # Define the feature extractor
    depth = args.depth if hasattr(args, 'depth') else 0
    if (utils.checkattr(args, 'hidden') or utils.checkattr(args, 'slda')):
        if verbose:
            print("\nDefining the feature extractor...")
        feature_extractor = define.define_feature_extractor(args=args, config=config, device=device)
        feature_extractor_name = feature_extractor.name if depth>0 else None
        # - initialize (pre-trained) parameters
        feature_extractor = define.init_params(feature_extractor, args)
        # - if requested, train feature extractor on first task
        if utils.checkattr(args, 'train_on_first'):
            pass
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
        for task_id in range(args.tasks):
            # -training data
            loader = utils.get_data_loader(train_datasets[task_id], batch_size=args.batch, drop_last=False, cuda=cuda)
            # -pre-allocate tensors, which will be filled slice-by-slice
            all_features = torch.empty((len(loader.dataset), config['channels'], config['size'], config['size']))
            all_labels = torch.empty((len(loader.dataset)), dtype=torch.long)
            count = 0
            for x, y in loader:
                x = feature_extractor(x.to(device)).cpu()
                all_features[count:(count+x.shape[0])] = x
                all_labels[count:(count+x.shape[0])] = y
                count += x.shape[0]
            new_train_datasets.append(TensorDataset(all_features, all_labels))
            # -testing data
            loader = utils.get_data_loader(test_datasets[task_id], batch_size=args.batch, drop_last=False, cuda=cuda)
            # -pre-allocate tensors, which will be filled slice-by-slice
            all_features = torch.empty((len(loader.dataset), config['channels'], config['size'], config['size']))
            all_labels = torch.empty((len(loader.dataset)), dtype=torch.long)
            count = 0
            for x, y in loader:
                x = feature_extractor(x.to(device)).cpu()
                all_features[count:(count+x.shape[0])] = x
                all_labels[count:(count+x.shape[0])] = y
                count += x.shape[0]
            new_test_datasets.append(TensorDataset(all_features, all_labels))
        # Reset datasets, and set feature extractor to None (as no longer needed now!)
        train_datasets = new_train_datasets
        test_datasets = new_test_datasets


    #-------------------------------------------------------------------------------------------------#

    #-----------------------#
    #----- EXPERT GATE -----#
    #-----------------------#

    # Define the expert gate
    if utils.checkattr(args, 'expert_gate'):
        if verbose:
            print("\nDefining the expert gate...")
        #expert_gate = ...


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- CLASSIFIER -----#
    #----------------------#

    # Define the classifier
    if verbose:
        print("\nDefining the classifier...")
    if utils.checkattr(args, 'slda'):
        model = define.define_slda(args=args, num_features=feature_extractor.conv_out_units, classes=config['classes'],
                                   device=device)
    elif utils.checkattr(args, 'feedback'):
        model = define.define_vae_classifier(args=args, config=config, device=device, depth=depth)
    else:
        model = define.define_classifier(args=args, config=config, device=device, depth=depth)

    # Initialize / use pre-trained / freeze model-parameters
    if not utils.checkattr(args, 'slda'):
        # - initialize (pre-trained) parameters
        model = define.init_params(model, args)
        # - freeze weights of conv-layers?
        if utils.checkattr(args, "freeze_convE") and not utils.checkattr(args, "freeze_after_first"):
            for param in model.convE.parameters():
                param.requires_grad = False
            model.convE.frozen = True
        # - freeze weights of fcE-layers?
        if utils.checkattr(args, "freeze_fcE") and not utils.checkattr(args, "freeze_after_first"):
            for param in model.fcE.parameters():
                param.requires_grad = False
            model.fcE.frozen = True

    # Define optimizer (only optimize parameters that "requires_grad")
    if not utils.checkattr(args, 'slda'):
        model.optim_type = args.optimizer
        model.optim_list = [
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
        ]
        if model.optim_type in ("adam", "adam_reset"):
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        elif model.optim_type == "sgd":
            model.optimizer = optim.SGD(model.optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))


    # -------------------------------------------------------------------------------------------------#

    # -------------------------------#
    # ----- CL-STRATEGY: REPLAY -----#
    # -------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay') and not args.replay=="none":
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    if (hasattr(args, 'replay') and args.replay=="generative") and not utils.checkattr(args, 'feedback'):
        # Specify architecture
        generator = define.define_autoencoder(args, config, device, depth=depth)

        # Initialize parameters
        generator = define.init_params(generator, args)

        # Set optimizer(s)
        generator.optim_type = args.optimizer
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr},
        ]
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(model.optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))
    else:
        generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------------------------#
    #----- CL-STRATEGY: REGULARIZATION -----#
    #---------------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'ewc'):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.online = utils.checkattr(args, 'online')
        if model.online:
            model.gamma = args.gamma
        if utils.checkattr(args, 'reg_only_hidden'):
            model.param_list = [model.convE.named_parameters, model.fcE.named_parameters]

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon
        model.omega_max = args.omega_max if hasattr(args, 'omega_max') else None
        # model.param_iterator = model.named_parameters()
        if utils.checkattr(args, 'reg_only_hidden'):
            model.param_list = [model.convE.named_parameters, model.fcE.named_parameters]


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------------#
    #----- CL-STRATEGY: BIAS-CORRECTION -----#
    #----------------------------------------#

    # Copy-Weight and Reinit (CWR)
    if isinstance(model, ContinualLearner) and (utils.checkattr(args, 'cwr') or utils.checkattr(args, 'cwr_plus')):
        model.cwr = True
        model.cwr_plus = utils.checkattr(args, 'cwr_plus')


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(args, model.name, replay_model_name=None if generator is None else generator.name,
                                  feature_extractor_name=feature_extractor_name, verbose=verbose)

    # Print some model-characteristics on the screen
    if verbose:
        # -classifier
        utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if utils.checkattr(args, 'pdf') or utils.checkattr(args, 'metrics'):
        # -define [metrics_dict] to keep track of performance during training for storing & for later plotting in pdf
        metrics_dict = evaluate.initiate_metrics_dict(n_tasks=args.tasks)
        # -evaluate randomly initiated model on all tasks & store accuracies in [metrics_dict] (for calculating metrics)
        metrics_dict = evaluate.intial_accuracy(model, test_datasets, metrics_dict, classes_per_task=classes_per_task,
                                                test_size=None)
    else:
        metrics_dict = None

    # Prepare for plotting in visdom
    visdom = None
    if utils.checkattr(args, 'visdom'):
        env_name = "{exp}{tasks}".format(exp=args.experiment, tasks=args.tasks)
        graph_name = "name"
        visdom = {'env': env_name, 'graph': graph_name}


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # NOTE: if --single-epochs is selected, the iters here are not correct anymore!!

    # Callbacks for reporting on and visualizing loss
    loss_cbs = [
        cb._loss_cb(log=args.loss_log if utils.checkattr(args, 'visdom') else None, visdom=visdom, model=model,
                    iters_per_task=args.iters, tasks=args.tasks)
    ]
    gen_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log if utils.checkattr(args, 'visdom') else None, visdom=visdom, model=generator,
                        iters_per_task=args.iters, tasks=args.tasks)
    ] if generator is not None else [None]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [acc_log]
    eval_cbs = [
        cb._eval_cb(log=args.acc_log, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.acc_n, classes_per_task=classes_per_task)
    ] if utils.checkattr(args, 'visdom') else [None]

    # Callbacks for plotting generated samples
    sample_log = args.sample_log if (hasattr(args, 'sample_log') and args.sample_log is not None) else args.iters
    no_samples = (utils.checkattr(args, "no_samples") or utils.checkattr(args, 'hidden')) \
                 or (not utils.checkattr(args, 'visdom'))
    sample_cbs = [
        cb._sample_cb(log=sample_log, visdom=visdom, config=config, sample_size=args.sample_n)
    ] if (generator is not None) and not no_samples else [None]

    # Callbacks for calculating statists required for metrics
    # -pdf / reporting: summary plots (i.e, only after each task)
    metric_cbs = [
        cb._metric_cb(test_datasets=test_datasets, classes_per_task=classes_per_task, metrics_dict=metrics_dict,
                      iters_per_task=args.iters)
    ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        if utils.checkattr(args, 'slda'):
            train.train_slda(model, train_datasets, batch_size=args.batch, metric_cbs=metric_cbs)
        else:
            train.train_cl(
                model, train_datasets, classes_per_task=classes_per_task, iters=args.iters,
                epochs=1 if args.single_epochs else None, args=args,
                batch_size=args.batch, eval_cbs=eval_cbs, loss_cbs=loss_cbs, reinit=utils.checkattr(args, 'reinit'),
                only_last=utils.checkattr(args, 'only_last'), metric_cbs=metric_cbs,
                offline=utils.checkattr(args, 'offline'),
                replay_mode=args.replay if hasattr(args, 'replay') else "none",
                generator=generator, gen_loss_cbs=gen_loss_cbs, sample_cbs=sample_cbs,
            )
        # Save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of previously trained model...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose, strict=False)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------------------#
    #----- EVALUATE CLASSIFICATION PERFORMANCE -----#
    #-----------------------------------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate accuracy of final model on full test-set
    ##--> Task-IL
    accs_ti = []
    for i in range(args.tasks):
        acc = evaluate.validate(model, test_datasets[i], verbose=False, test_size=None,
                                 allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))), S=10)
        accs_ti.append(acc)
    average_accs_ti = sum(accs_ti) / args.tasks
    ##--> Class-IL
    if verbose:
        print("\n Accuracy of final model on test-set:")
    accs_ci = []
    for i in range(args.tasks):
        acc = evaluate.validate(model, test_datasets[i], verbose=False, allowed_classes=None, S=10, test_size=None)
        if verbose:
            print(" - For classes from task {}: {:.4f}".format(i + 1, acc))
        accs_ci.append(acc)
    average_accs_ci = sum(accs_ci)/args.tasks
    if verbose:
        print('=> Average accuracy over all {} classes: {:.4f}\n'.format(args.tasks*classes_per_task, average_accs_ci))

    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average accuracy on full test set
    output_file = open("{}/accTI-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_accs_ti))
    output_file.close()
    output_file = open("{}/accCI-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_accs_ci))
    output_file.close()
    # -metrics-dict
    if utils.checkattr(args, 'metrics') and args.train:
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(metrics_dict, file_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if utils.checkattr(args, 'pdf'):
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = visual_plt.open_pdf(plot_name)

        # -show metrics reflecting progression during training
        if args.train and (not utils.checkattr(args, 'only_last')):
            # -create list to store all figures to be plotted.
            figure_list = []
            # -generate all figures (and store them in [figure_list])
            plot_list = []
            # -Task-IL
            key = "acc per task (only classes in task)"
            for i in range(args.tasks):
                plot_list.append(metrics_dict[key]["task {}".format(i + 1)])
            figure = visual_plt.plot_lines(
                plot_list, x_axes=metrics_dict["x_task"],
                line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
            )
            figure_list.append(figure)
            figure = visual_plt.plot_lines(
                [metrics_dict["average"]], x_axes=metrics_dict["x_task"],
                line_names=['average all tasks so far']
            )
            figure_list.append(figure)
            # -Class-IL
            key = "acc per task (all classes up to trained task)"
            for i in range(args.tasks):
                plot_list.append(metrics_dict[key]["task {}".format(i + 1)])
            figure = visual_plt.plot_lines(
                plot_list, x_axes=metrics_dict["x_task"],
                line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
            )
            figure_list.append(figure)
            figure = visual_plt.plot_lines(
                [metrics_dict["average"]], x_axes=metrics_dict["x_task"],
                line_names=['average all tasks so far']
            )
            figure_list.append(figure)
            # -add figures to pdf
            for figure in figure_list:
                pp.savefig(figure)

        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)