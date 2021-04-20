import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers,DeconvLayers
from models.fc.nets import MLP
from models.fc.layers import fc_layer,fc_layer_split
from utils import get_data_loader



class AutoEncoder(nn.Module):
    """Class for variational auto-encoder (VAE) models."""

    def __init__(self, image_size, image_channels,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, convE=None, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=False,
                 fc_gated=False,
                 # -prior
                 z_dim=20, prior="standard", n_modes=1,
                 # -decoder
                 recon_loss='BCE', network_output="sigmoid", deconv_type="standard"):

        # Set configurations for setting up the model
        super().__init__()
        self.label = "VAE"
        self.image_size = image_size
        self.image_channels = image_channels
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth if convE is None else convE.depth
        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss # options: BCE|MSE
        self.network_output = network_output

        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # Prior-related parameters (for "vamp-prior" / "GMM")
        self.prior = prior
        self.n_modes = n_modes
        # -vampprior-specific (note that these are about initializing the vamp-prior's pseudo-inputs):
        self.prior_mean = 0.25   # <-- data-specific!! TO BE CHANGED
        self.prior_sd = 0.05     # <-- data-specific!! TO BE CHANGED

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")


        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = ConvLayers(conv_type=conv_type, block_type="basic", num_blocks=num_blocks,
                                image_channels=image_channels, depth=self.depth, start_channels=start_channels,
                                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
                                output="none" if no_fnl else "normal", global_pooling=global_pooling,
                                gated=conv_gated) if (convE is None) else convE
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers<2:
            self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
        elif fc_layers==2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
        real_h_dim = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                       excit_buffer=excit_buffer, gated=fc_gated)
        # to z
        self.toZ = fc_layer_split(real_h_dim, z_dim, nl_mean='none', nl_logvar='none')#, drop=fc_drop)

        ##>----Decoder (= p[x|z])----<##
        out_nl = True if fc_layers > 1 else (True if (self.depth > 0 and not no_fnl) else False)
        real_h_dim_down = h_dim if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none")
        fc_layer_sizes_down = self.fc_layer_sizes
        fc_layer_sizes_down[0] = self.convE.out_units(image_size, ignore_gp=True)
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        self.fcD = MLP(
            size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
            gated=fc_gated, output=self.network_output if self.depth==0 else 'normal',
        )
        # to image-shape
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth>0 else image_channels)
        # through deconv-layers
        self.convD = DeconvLayers(
            image_channels=image_channels, final_channels=start_channels, depth=self.depth,
            reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated,
            output=self.network_output, deconv_type=deconv_type,
        )

        ##>----Prior----<##
        # -if using the vamp-prior, add pseudo-inputs
        if self.prior=="vampprior":
            # -create
            self.add_pseudoinputs()
            # -initialize
            self.initialize_pseudoinputs(prior_mean=self.prior_mean, prior_sd=self.prior_sd)
        # -if using the GMM-prior, add its parameters
        if self.prior=="GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False



    ##------ PRIOR --------##

    def add_pseudoinputs(self):
        '''Create pseudo-inputs for the vamp-prior.'''
        n_inputs = self.image_channels * self.image_size**2
        shape = [self.n_modes, self.image_channels, self.image_size, self.image_size]
        # define nn-object with learnable parameters, that transforms "idle-inputs" to the learnable pseudo-inputs
        self.make_pseudoinputs = nn.Sequential(
            nn.Linear(self.n_modes, n_inputs, bias=False),
            nn.Hardtanh(min_val=0.0, max_val=1.0) if self.network_output=="sigmoid" else modules.Identity(),
            modules.Shape(shape=shape)
        )
        # create "idle"-input
        self.idle_input = torch.eye(self.n_modes, self.n_modes)

    def initialize_pseudoinputs(self, prior_mean=0.2, prior_sd=0.05):
        '''Initialize the learnable parameters of the pseudo-inputs for the vamp-prior.'''
        self.make_pseudoinputs[0].weight.data.normal_(prior_mean, prior_sd)



    ##------ NAMES --------##

    def get_name(self):
        convE_label = "{}_".format(self.convE.name) if self.depth>0 else ""
        fcE_label = "{}_".format(self.fcE.name) if self.fc_layers>1 else "{}{}_".format("h" if self.depth>0 else "i",
                                                                                        self.conv_out_units)
        z_label = "z{}{}".format(self.z_dim, "" if self.prior=="standard" else "-{}{}".format(self.prior, self.n_modes))
        return "{}={}{}{}".format(self.label, convE_label, fcE_label, z_label)

    @property
    def name(self):
        return self.get_name()



    ##------ UTILITIES --------##

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda



    ##------ LAYERS --------##

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        '''Return list with shape of all hidden layers.'''
        # create list with hidden convolutional layers
        layer_list = self.convE.layer_info(image_size=self.image_size)
        # add output of final convolutional layer (if there was at least one conv-layer and there's fc-layers after)
        if (self.fc_layers>0 and self.depth>0):
            layer_list.append([self.conv_out_channels, self.conv_out_size, self.conv_out_size])
        # add layers of the MLP
        if self.fc_layers>1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list



    ##------ FORWARD FUNCTIONS --------##

    def encode(self, x):
        '''Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE].'''
        # Forward-pass through conv-layers
        image_features = self.flatten(self.convE(x))
        # Forward-pass through fc-layers
        hE = self.fcE(image_features)
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE

    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()#.requires_grad_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        '''Decode latent variable activations.

        INPUT:  - [z]            <2D-tensor>; latent variables to be decoded

        OUTPUT: - [image_recon]  <4D-tensor>'''

        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, reparameterize=True, **kwargs):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction'''

        mu, logvar, hE = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        x_recon = self.decode(z)
        return (x_recon, mu, logvar, z)

    def feature_extractor(self, images):
        '''Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images.'''
        return self.fcE(self.flatten(self.convE(images)))



    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, sample_mode=None, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device.'''

        # set model to eval()-mode
        self.eval()

        # sample for each sample the prior-mode to be used
        if self.prior in ["vampprior", "GMM"] and sample_mode is None:
            sampled_modes = np.random.randint(0, self.n_modes, size)

        # sample z
        if self.prior in ["vampprior", "GMM"]:
            if self.prior=="vampprior":
                with torch.no_grad():
                    # -get pseudo-inputs
                    X = self.make_pseudoinputs(self.idle_input.to(self._device()))
                    #- pass pseudo-inputs through ("variational") encoder
                    prior_means, prior_logvars, _ = self.encode(X)
            else:
                prior_means = self.z_class_means
                prior_logvars = self.z_class_logvars
            # -for each sample to be generated, select the previously sampled mode
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim).to(self._device())

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor
        return X



    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, average=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''

        batch_size = x.size(0)
        if self.recon_loss=="MSE":
            # reconL = F.mse_loss(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            # reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss=="BCE":
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                            reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError("Wrong choice for type of reconstruction-loss!")
        # --> if [average]=True, reconstruction loss is averaged over all pixels/elements (otherwise it is summed)
        #       (averaging over all elements in the batch will be done later)
        return reconL


    def calculate_log_p_z(self, z):
        '''Calculate log-likelihood of sampled [z] under the prior distirbution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            log_p_z = lf.log_Normal_standard(z, average=False, dim=1)   # [batch_size]
        elif self.prior in ("vampprior", "GMM"):
            # Get [means] and [logvars] of all (possible) modes
            allowed_modes = list(range(self.n_modes))
            # -calculate/retireve the means and logvars for all modes
            if self.prior == "vampprior":
                X = self.make_pseudoinputs(self.idle_input.to(self._device()))  # get pseudo-inputs
                prior_means, prior_logvars, _ = self.encode(X[allowed_modes])  # pass them through encoder
            else:
                prior_means = self.z_class_means[allowed_modes, :]
                prior_logvars = self.z_class_logvars[allowed_modes, :]
            # -rearrange / select for each batch prior-modes to be used
            z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
            means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
            logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

            # Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            n_modes = len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            # --> for each element in batch, calculate log-likelihood for all modes: [batch_size] x [n_modes]
            a_max, _ = torch.max(a, dim=1)  # [batch_size]
            # --> for each element in batch, take highest log-likelihood over all modes
            #     this is calculated and used to avoid underflow in the below computation
            a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
            a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))  # -> sum over modes: [batch_size]
            log_p_z = a_logsum + a_max  # [batch_size]

        return log_p_z


    def calculate_variat_loss(self, z, mu, logvar):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            # --> calculate analytically
            # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elif self.prior in ("vampprior", "GMM"):
            # --> calculate "empirically"

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on prior)
            log_p_z = self.calculate_log_p_z(z) # [batch_size]

            ## Calculate "log_q_z" (entropy of "reparameterized" [z] given [x])
            log_q_z = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
            # ----->  log_q_z: [batch_size]

            ## Combine
            variatL = -(log_p_z - log_q_z)

        return variatL


    def loss_function(self, x, x_recon, mu, z, logvar, batch_weights=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"'''

        ###-----Reconstruction loss-----###
        batch_size = x.size(0)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True,
                                           x_recon=x_recon.view(batch_size, -1)) # -> average over pixels
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)       # -> average over batch

        ###-----Variational loss-----###
        variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar)
        variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)  # -> average over batch
        variatL /= (self.image_channels * self.image_size ** 2)               # -> divide by # of input-pixels

        # Return a tuple of the calculated losses
        return reconL, variatL



    ##------ EVALUATION FUNCTIONS --------##

    def calculate_recon_error(self, dataset, batch_size=128, max_batches=None, average=False, feature_extractor=None):
        '''Calculate reconstruction error of the model for each datapoint in [dataset].

        [average]     <bool>, if True, reconstruction-error is averaged over all pixels/units; otherwise it is summed'''

        # Create data-loader
        data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=self._is_on_cuda())

        # Break loop if max number of batches has been reached
        for index, (x, _) in enumerate(data_loader):
            if max_batches is not None and index >= max_batches:
                break

            # Move [x] to correct device
            x = x.to(self._device())

            # Preprocess, if required
            if feature_extractor is not None:
                with torch.no_grad():
                    x = feature_extractor(x)

            # Run forward pass of model to get [z_mean]
            with torch.no_grad():
                z_mean, _, _ = self.encode(x)

            # Run backward pass of model to reconstruct input
            with torch.no_grad():
                x_recon = self.decode(z_mean)

            # Calculate reconstruction error
            recon_error = self.calculate_recon_loss(x.view(x.size(0), -1), x_recon.view(x.size(0), -1), average=average)

            # Concatanate the calculated reconstruction errors for all evaluated samples
            all_res = torch.cat([all_res, recon_error]) if index > 0 else recon_error

        # Convert to <np-array> (with one entry for each evaluated sample in [dataset]) and return
        return all_res.cpu().numpy()


    def estimate_loglikelihood_single(self, x, S=5000, batch_size=128):
        '''Estimate average marginal log-likelihood for [x] using [S] importance samples.'''

        # Move [x]  to correct device
        x = x.to(self._device())

        # Run forward pass of model to get [z_mu] and [z_logvar]
        with torch.no_grad():
            z_mu, z_logvar, _ = self.encode(x)

        # Importance samples will be calcualted in batches, get number of required batches
        repeats = int(np.ceil(S / batch_size))

        # For each importance sample, calculate log_likelihood
        for rep in range(repeats):
            batch_size_current = (S % batch_size) if rep==(repeats-1) else batch_size

            # Reparameterize (i.e., sample z_s)
            z = self.reparameterize(z_mu.expand(batch_size_current, -1), z_logvar.expand(batch_size_current, -1))

            # Calculate log_p_z
            with torch.no_grad():
                log_p_z = self.calculate_log_p_z(z)

            # Calculate log_q_z_x
            log_q_z_x = lf.log_Normal_diag(z, mean=z_mu, log_var=z_logvar, average=False, dim=1)

            # Calcuate log_p_x_z
            # -reconstruct input
            with torch.no_grad():
                x_recon = self.decode(z)
            # -calculate p_x_z (under Gaussian observation model with unit variance)
            log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)

            # Calculate log-likelihood for each importance sample
            log_likelihoods = log_p_x_z + log_p_z - log_q_z_x

            # Concatanate the log-likelihoods of all importance samples
            all_lls = torch.cat([all_lls, log_likelihoods]) if rep > 0 else log_likelihoods

        # Calculate average log-likelihood over all importance samples for this test sample
        #  (for this, convert log-likelihoods back to likelihoods before summing them!)
        log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)

        return log_likelihood


    def estimate_loglikelihood(self, dataset, S=5000, batch_size=128, max_n=None, feature_extractor=None):
        '''Estimate average marginal log-likelihood for x|y of the model on [dataset] using [S] importance samples.'''

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        # List to store estimated log-likelihood for each datapoint
        ll_per_datapoint = []

        # Break loop if max number of samples has been reached
        for index, (x, _) in enumerate(data_loader):
            if max_n is not None and index >= max_n:
                break

            # Preprocess, if required
            if feature_extractor is not None:
                x = x.to(self._device())
                with torch.no_grad():
                    x = feature_extractor(x)

            # Estimate log-likelihood for the input-output pair (x,y)
            log_likelihood = self.estimate_loglikelihood_single(x, S=S, batch_size=batch_size)

            # Add it to list
            ll_per_datapoint.append(log_likelihood.cpu().numpy())

        return ll_per_datapoint



    ##------ TRAINING FUNCTIONS --------##

    def train_a_batch(self, x, x_=None, rnt=0.5, **kwargs):
        '''Train model for one batch ([x]).

        [x]                 <tensor> batch of inputs
        [x_]                None or <tensor> batch of replayed inputs
        [rnt]               <number> in [0,1], relative importance of new task
        '''

        # Set model to training-mode
        self.train()
        # -however, if some layers are frozen, they shoud be set to eval() to prevent batch-norm layers from changing
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()

        # Reset optimizer
        self.optimizer.zero_grad()


        ##-- CURRENT DATA --##

        # Run the model
        recon_batch, mu, logvar, z = self(x, full=True, reparameterize=True)

        # Calculate all losses
        reconL, variatL = self.loss_function(x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

        # Weigh losses
        loss_cur = reconL + variatL


        ##-- REPLAYED DATA --##
        if x_ is not None:
            # Run the model
            recon_batch, mu, logvar, z = self(x_, full=True, reparameterize=True)

            # Calculate all losses
            reconL_r, variatL_r = self.loss_function(x=x_, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

            # Weigh losses
            loss_replay = reconL_r + variatL_r


        # Calculate total loss
        loss = loss_cur if x_ is None else rnt*loss_cur + (1-rnt)*loss_replay

        # Backpropagate gradients
        loss.backward()

        # Take optimization-step
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss.item(),
            'recon': reconL.item() if x is not None else 0,
            'variat': variatL.item() if x is not None else 0,
        }
