import utils
from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining auto-encoder model
def define_vae_classifier(args, config, device, depth=0):
    # -import required model
    from models.vae_with_classifier import AutoEncoder
    # -create model
    if depth > 0:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=True if args.conv_bn == "yes" else False, conv_nl=args.conv_nl,
            global_pooling=checkattr(args, 'gp'),
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "n_modes") else 1, z_dim=args.z_dim,
            per_class=args.per_class if hasattr(args, "prior") else False,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment == "MNIST" else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
            dg_gates=utils.checkattr(args, 'dg_gates'), device=device,
            dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
            # -classifier
            classifier=True, classify_opt=args.classify if hasattr(args, "classify") else "beforeZ", lamda_pl=1.
        ).to(device)
    else:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "n_modes") else 1, z_dim=args.z_dim,
            per_class=args.per_class if hasattr(args, "prior") else False,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment == "MNIST" else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
            dg_gates=utils.checkattr(args, 'dg_gates'), device=device,
            dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
            # -classifier
            classifier=True, classify_opt=args.classify if hasattr(args, "classify") else "beforeZ", lamda_pl=1.,
        ).to(device)
    # -return model
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining auto-encoder model
def define_autoencoder(args, config, device, depth=0):
    # -import required model
    from models.vae import AutoEncoder
    # -create model
    if depth > 0:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            # -conv-layers
            conv_type=args.conv_type, depth=depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=True if args.conv_bn=="yes" else False, conv_nl=args.conv_nl,
            global_pooling=False, no_fnl=True if args.conv_type=="standard" else False,
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "n_modes") else 1, z_dim=args.z_dim,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment=="MNIST" else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
        ).to(device)
    else:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "n_modes") else 1, z_dim=args.z_dim,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment=="MNIST" else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining feature extractor model
def define_feature_extractor(args, config, device):
    # -import required model
    from models.feature_extractor import FeatureExtractor
    # -create model
    model = FeatureExtractor(
        image_size=config['size'], image_channels=config['channels'],
        # -conv-layers
        conv_type=args.conv_type, depth=args.depth, start_channels=args.channels, reducing_layers=args.rl,
        num_blocks=args.n_blocks, conv_bn=True if args.conv_bn=="yes" else False, conv_nl=args.conv_nl,
        global_pooling=checkattr(args, 'gp'),
    ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining SLDA model
def define_slda(args, num_features, classes, device='cpu'):
    from models.slda import StreamingLDA
    # -create model
    classifier = StreamingLDA(
        num_features=num_features, classes=classes,
        # -slda parameters
        epsilon=1e-4, device=device, covariance=args.covariance if hasattr(args, 'covariance') else "identity",
    ).to(device)
    return classifier

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining classifier model
def define_classifier(args, config, device, no_fnl_fc=False, depth=0):
    # -import required model
    from models.classifier import Classifier
    # -create model
    if depth > 0:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=True if args.conv_bn=="yes" else False, conv_nl=args.conv_nl,
            global_pooling=checkattr(args, 'gp'), no_fnl=True if args.conv_type=="standard" else False,
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim, no_fnl_fc=no_fnl_fc,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -training related parameters
            neg_samples=args.neg_samples if hasattr(args, "neg_samples") else "all",
            classes_per_task=config['classes_per_task'] if hasattr(args, "tasks") else None
        ).to(device)
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim, no_fnl_fc=no_fnl_fc,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -training related parameters
            neg_samples=args.neg_samples if hasattr(args, "neg_samples") else "all",
            classes_per_task=config['classes_per_task'] if hasattr(args, "tasks") else None
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining auto-encoder model
def define_gen_classifer(args, config, device, convE=None, depth=0):
    # -import required model
    from models.gen_classsifier import GenClassifier
    # -create model
    if depth > 0:
        model = GenClassifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=depth,
            start_channels=args.channels, reducing_layers=args.rl, conv_bn=(args.conv_bn=="yes"), conv_nl=args.conv_nl,
            num_blocks=args.n_blocks, convE=convE, global_pooling=checkattr(args, 'gp'),
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=(args.fc_bn=="yes"), fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior, n_modes=args.n_modes, z_dim=args.z_dim,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment=='MNIST' else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
        ).to(device)
    else:
        model = GenClassifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=(args.fc_bn=="yes"), fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior, n_modes=args.n_modes, z_dim=args.z_dim,
            # -decoder
            recon_loss=args.recon_loss, network_output="sigmoid" if args.experiment=='MNIST' else "none",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args):
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)
    # - use pre-trained weights in conv-layers?
    if utils.checkattr(args, "pre_convE") and hasattr(model, 'depth') and model.depth>0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag=="none"
        ) else "{}-{}".format(model.convE.name, args.convE_ltag)
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name)
    return model

##-------------------------------------------------------------------------------------------------------------------##