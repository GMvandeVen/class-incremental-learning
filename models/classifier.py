import numpy as np
import random
import torch
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.cl.continual_learner import ContinualLearner


class Classifier(ContinualLearner):
    '''Model for encoding (i.e., feature extraction) and classifying images, enriched as "ContinualLearner"--object.'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=False, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False, no_fnl_fc=False,
                 # -training parameters (on which classes to train)
                 neg_samples="all-so-far", classes_per_task=None):

        # Model configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # How to select the negative samples?
        self.neg_samples = neg_samples
        self.classes_per_task = classes_per_task

        # Optimizer (needs to be set before training starts))
        self.optim_type = None
        self.optimizer = None
        self.optim_list = []

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
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
        self.units_before_classifier = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        #--> fully connected layers
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated,
                       output="none" if no_fnl_fc else "normal")
        #--> classifier
        self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none', drop=fc_drop)

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.fc_layer_sizes[0], "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def forward(self, x, y=None):
        hidden_rep = self.convE(x)
        final_features = self.fcE(self.flatten(hidden_rep))
        return self.classifier(final_features)

    def feature_extractor(self, images):
        return self.fcE(self.flatten(self.convE(images)))

    def classify(self, x):
        '''For input [x] (image or extracted "intermediate" image features), return all predicted "scores"/"logits".'''
        image_features = self.flatten(self.convE(x))
        hE = self.fcE(image_features)
        return self.classifier(hE)


    def train_a_batch(self, x, y, x_=None, y_=None, scores_=None, rnt=0.5, classes_so_far=None, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 <tensor> batch of corresponding labels
        [x_]                None or <tensor> batch of replayed inputs
        [y_]                None or <tensor> batch of corresponding "replayed" labels
        [scores_]           None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]               <number> in [0,1], relative importance of new task
        [classes_so_far]    <list> with all classes seen so far'''

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

        # Run model
        y_hat = self(x)

        # Remove predictions for classes not to be trained on
        if self.neg_samples=="single-from-batch":
            # -only use a single "negative sample" sampled from current batch
            batch_size = x.shape[0]
            # -for each entry in the batch, select another label from `y_list_to_sample_from` as the negative sample
            y_list = list(y.cpu().numpy())
            y_list_to_sample_from = y_list
            ys_to_test = torch.LongTensor(batch_size, 2).to(self._device())
            for i in range(len(y_list)):
                if len(np.unique(y_list_to_sample_from))==1:
                    # the below would break down if all entries in the batch have the same label!!!
                    raise RuntimeError('All samples in this batch have the same label!!')
                else:
                    while True:
                        neg_sample = random.choice(y_list_to_sample_from)
                        if not neg_sample == y_list[i]:
                            break
                ys_to_test[i] = torch.tensor([y_list[i], neg_sample]).to(self._device())
            y_hat = y_hat.gather(1, ys_to_test)
            y = torch.zeros_like(y)      #--> because the first sample is always the correct one now
        else:
            if self.neg_samples=="current":
                # -train on all classes in current task
                class_entries = classes_so_far[-self.classes_per_task:]
                y = y - class_entries[0]  #--> to make [y] consistent with order of [y_hat] below
            elif self.neg_samples=="all-so-far":
                # -train on all classes so far
                class_entries = classes_so_far
            elif self.neg_samples=="all":
                # -train on all classes (also those not yet seen)
                class_entries = None
            y_hat = y_hat[:, class_entries] if class_entries is not None else y_hat

        # Calculate loss
        pred_loss = F.cross_entropy(input=y_hat, target=y, reduction='mean')
        loss_cur = pred_loss

        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)


        ##-- REPLAYED DATA --##
        # Note: replay can currently only be used with neg_samples=="all" or neg_samples="all-so-far"
        if x_ is not None:

            # Run model
            y_hat = self(x_)

            # Remove predictions for classes not to be trained on
            if self.neg_samples=="all-so-far":
                class_entries = classes_so_far
            elif self.neg_samples=="all":
                class_entries = None
            y_hat = y_hat[:, class_entries] if class_entries is not None else y_hat

            # Calculate loss
            if self.replay_targets == "hard":
                loss_replay = F.cross_entropy(y_hat, y_, reduction='mean')
            elif self.replay_targets == "soft":
                # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                n_classes_to_consider = y_hat.size(1)    # --> zeros will be added to [scores] to make it this size!
                loss_replay = lf.loss_fn_kd(
                    scores=y_hat[:, :n_classes_to_consider], target_scores=scores_, T=self.KD_temp,
                )  # --> summing over classes & averaging over batch within this function


        # Calculate total loss
        loss = loss_cur if x_ is None else rnt*loss_cur + (1-rnt)*loss_replay


        ##-- EWC/SI SPECIFIC LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss += self.ewc_lambda * ewc_loss


        # Backpropagate errors
        loss.backward()

        # Take optimization-step
        self.optimizer.step()


        # Return the dictionary with training-loss split in categories
        return {
            'loss_total': loss.item(),
            'pred': pred_loss.item(),
            'ewc': ewc_loss.item(),
            'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }

