from data.load import get_experiment
from utils import checkattr


def get_param_stamp_from_args(args, gen_classifier=False):
    '''To get param-stamp a bit quicker.'''
    import define_models as define

    # -get configurations of experiment
    config = get_experiment(
        name=args.experiment, tasks=args.tasks if hasattr(args, 'tasks') else 1, data_dir=args.d_dir, only_config=True,
        normalize=args.normalize if hasattr(args, "normalize") else False, verbose=False,
    )

    # -get feature extractor
    feature_extractor_name = None
    depth = args.depth if hasattr(args, 'depth') else 0
    if (checkattr(args, 'hidden') or checkattr(args, 'slda')):
        feature_extractor = define.define_feature_extractor(args=args, config=config, device='cpu')
        feature_extractor_name = feature_extractor.name if depth > 0 else None
        config = config.copy()  # -> make a copy to avoid overwriting info in the original config-file
        config['size'] = feature_extractor.conv_out_size
        config['channels'] = feature_extractor.conv_out_channels
        depth = 0
    # -get classifier architecture
    if gen_classifier:
        model = define.define_gen_classifer(args=args, config=config, device='cpu', depth=depth)
    elif checkattr(args, 'slda'):
        model = define.define_slda(args=args, num_features=feature_extractor.conv_out_units, classes=config['classes'],
                                   device='cpu')
    elif checkattr(args, 'feedback'):
        model = define.define_vae_classifier(args=args, config=config, device='cpu', depth=depth)
    else:
        model = define.define_classifier(args=args, config=config, device='cpu', depth=depth)
    # -get generator architecture
    if (hasattr(args, 'replay') and args.replay=="generative") and (not checkattr(args, 'feedback')):
        generator = define.define_autoencoder(args, config, device='cpu', depth=depth)
        replay_model_name = generator.name
    else:
        replay_model_name = None

    # -extract and return param-stamp
    param_stamp = get_param_stamp_gen_classifier(
        args, model.name, feature_extractor_name=feature_extractor_name, verbose=False
    ) if gen_classifier else get_param_stamp(
        args, model.name, replay_model_name=replay_model_name, feature_extractor_name=feature_extractor_name,
        verbose=False
    )
    return param_stamp



def get_param_stamp_gen_classifier(args, model_name, feature_extractor_name=None, verbose=True):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    task_stamp = "{exp}{norm}{aug}".format(
        exp=args.experiment, norm="-N" if hasattr(args, 'normalize') and args.normalize else "",
        aug="+" if hasattr(args, "augment") and args.augment else "",
    )
    if verbose:
        print(" --> task:           "+task_stamp)

    # -for model
    model_stamp = model_name if feature_extractor_name is None else "{}--{}".format(feature_extractor_name, model_name)
    if verbose:
        print(" --> model:          "+model_stamp)

    # -for training
    pre_conv = ""
    if checkattr(args, "pre_convE") and (args.depth>0 or feature_extractor_name is not None):
        ltag = "" if not hasattr(args, "convE_ltag") or args.convE_ltag=="none" else "-{}".format(args.convE_ltag)
        pre_conv = "-pCvE{}".format(ltag)
    freeze_conv = "-fCvE" if (
        checkattr(args, "freeze_convE") and (args.depth>0 or feature_extractor_name is not None)
    ) else ""
    train_stamp = "{i_e}{num}-lr{lr}-b{bsz}{pretr}{freeze}{recon}".format(
        i_e="e" if (args.iters is None) or checkattr(args, 'single_epochs') else "i",
        num=1 if checkattr(args, 'single_epochs') else (args.epochs if (args.iters is None) else args.iters),
        lr=args.lr, bsz=args.batch, pretr=pre_conv, freeze=freeze_conv,
        recon="-{}".format(args.recon_loss) if hasattr(args, 'recon_loss') else "",
    )
    if verbose:
        print(" --> training-params: " + train_stamp)

    # --> combine
    param_stamp = "{}--{}--{}{}".format(
        task_stamp, model_stamp, train_stamp, "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp



def get_param_stamp(args, model_name, replay_model_name=None, feature_extractor_name=None, verbose=True):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for tasks settings
    multi_n_stamp = "-{n}{off}".format(
        n=args.tasks, off="-OFF" if checkattr(args, 'offline') else ""
    ) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{norm}{aug}{multi_n}".format(
        exp=args.experiment, norm="-N" if hasattr(args, 'normalize') and args.normalize else "",
        aug="+" if hasattr(args, "augment") and args.augment else "", multi_n=multi_n_stamp,
    )
    if verbose:
        print(" --> tasks:         "+task_stamp)

    # -for model
    model_stamp = model_name if feature_extractor_name is None else "H{}{}--{}".format(
        feature_extractor_name, "-tr1" if checkattr(args, 'train_on_first') else "", model_name
    )
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for (pre-)training / freezing of feature extractor
    if checkattr(args, "pre_convE") and hasattr(args, 'depth') and args.depth>0:
        ltag = "" if (not hasattr(args, "convE_ltag")) or args.convE_ltag=="none" else "-{}".format(args.convE_ltag)
        pre = "pCvE{}".format(ltag)
    else:
        pre = "pNo"
    freeze_conv = (checkattr(args, "freeze_convE") and hasattr(args, 'depth') and args.depth>0)
    freeze = "-f{}{}".format("All" if checkattr(args, 'freeze_fcE') else "CvE",
                             "-af1" if checkattr(args, 'freeze_after_first') else "") if (
            (freeze_conv and (feature_extractor_name is None)) or checkattr(args, "freeze_fcE")
    ) else ""
    pretrain_stamp = "{pre}{freeze}".format(pre=pre, freeze=freeze)
    if verbose:
        print(" --> pretraining:   " + pretrain_stamp)

    # -for training parameters
    needed = (checkattr(args, 'train_on_first') and feature_extractor_name is not None) or (not checkattr(args, 'slda'))
    if needed:
        epochs = "{i_e}{num}".format(
            i_e="e" if (args.iters is None) or checkattr(args, 'single_epochs') else "i",
            num=1 if checkattr(args, 'single_epochs') else (args.epochs if (args.iters is None) else args.iters),
        )
        hyper_stamp = "{epochs}-{optim}-lr{lr}-b{bsz}{reinit}".format(
            epochs=epochs, optim=args.optimizer, lr=args.lr, bsz=args.batch,
            reinit="-R" if checkattr(args, 'reinit') else ""
        )
        if verbose:
            print(" --> train-params:  " + hyper_stamp)
    train_stamp = "--{}".format(hyper_stamp) if needed else ""

    # -for negative samples (i.e., which classes to train on?)
    neg_sample_stamp = "--{}".format(args.neg_samples) if not checkattr(args, 'slda') else ""
    if verbose and not checkattr(args, 'slda'):
        print(" --> neg sampling:  " + args.neg_samples)

    # -for EWC / SI
    if (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0):
        ewc_stamp = "EWC{l}-{fi}{o}{only_hid}".format(
            l=args.ewc_lambda, fi="{}".format("N" if args.fisher_n is None else args.fisher_n),
            o="-O{}".format(args.gamma) if checkattr(args, 'online') else "",
            only_hid="-oh" if checkattr(args, 'reg_only_hidden') else ""
        ) if (checkattr(args, 'ewc') and args.ewc_lambda>0) else ""
        si_stamp = "SI{c}-{eps}{max}{only_hid}".format(
            c=args.si_c, eps=args.epsilon, #--> below line is artefact of earlier mistake
            max="-m{}".format(args.omega_max) if hasattr(args, 'omega_max') and (args.omega_max is not None) else "-mNone",
            only_hid="-oh" if checkattr(args, 'reg_only_hidden') else ""
        ) if (checkattr(args,'si') and args.si_c>0) else ""
        both = "--" if (checkattr(args,'ewc') and args.ewc_lambda>0) and (checkattr(args,'si') and args.si_c>0) else ""
        if verbose and checkattr(args, 'ewc') and args.ewc_lambda>0:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and checkattr(args, 'si') and args.si_c>0:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
            (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0)
    ) else ""

    # -for bias-correcting
    if checkattr(args, 'cwr') or checkattr(args, 'cwr_plus'):
        bias_stamp = "--cwr{}".format("+" if checkattr(args, "cwr_plus") else "")
    else:
        bias_stamp = ""

    # -for replay
    if hasattr(args, 'replay') and not args.replay=="none":
        replay_stamp = "{rep}{distil}{model}".format(
            rep="gen" if args.replay=="generative" else args.replay,
            distil="-Di{}".format(args.temp) if args.distill else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
        )
        if verbose:
            print(" --> replay:   " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if (hasattr(args, 'replay') and not args.replay=="none") else ""

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}".format(
        task_stamp, model_stamp, pretrain_stamp, train_stamp, neg_sample_stamp, replay_stamp, ewc_stamp, bias_stamp,
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp
