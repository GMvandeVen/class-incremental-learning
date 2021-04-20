import tqdm
import copy
import torch
from torch import optim
from torch.utils.data import ConcatDataset
import utils
from models.cl.continual_learner import ContinualLearner


def train(model, train_loader, iters, loss_cbs=list(), eval_cbs=list(), save_every=None, m_dir="./store/models"):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].

    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''

    device = model._device()

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict, epoch=epoch)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration, epoch=epoch)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break

            # Save checkpoint?
            if (save_every is not None) and (iteration % save_every) == 0:
                utils.save_checkpoint(model, model_dir=m_dir)



def train_cl(model, train_datasets, classes_per_task=None, iters=2000, epochs=None, batch_size=32, offline=False,
             loss_cbs=list(), eval_cbs=list(), reinit=False, args=None, only_last=False, metric_cbs=list(),
             replay_mode="none", rnt=None, generator=None, gen_loss_cbs=list(), sample_cbs=list()):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [classes_per_task]  <int>, # classes per task
    [iters]             <int>, # optimization-steps (=batches) per task
    [only_last]         <bool>, only train on final task / episode
    [replay_mode]       <str>, choice from "generative", "current", "offline" and "none"
    [rnt]               <float>, indicating relative importance of new task (if None, relative to # old tasks)
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''

    # Use cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Initiate indicators for replay (no replay for 1st task)
    Generative = Current = False

    # Register starting param-values (needed for SI).
    if isinstance(model, ContinualLearner) and model.si_c>0:
        for gen_params in model.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

    # Set parameters of output-layer to 0 and store "cw"-version (needed for CWR and Weights Replay).
    if isinstance(model, ContinualLearner) and model.cwr:
        for n, p in model.classifier.named_parameters():
            # -if requested, set to zero
            if model.cwr:
                p.data.zero_()
            # -initialize the "stored version" of the classifier weights
            n = n.replace('.', '__')
            model.register_buffer('{}_stored_version'.format(n), p.detach().clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):

        # If 'offline' training, merge training data of all tasks so far
        if offline:
            train_dataset = ConcatDataset(train_datasets[:task])

        # If [epochs] is provided, adjust number of iterations
        if epochs is not None:
            data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader)*epochs

        # Initialize # iters left on data-loader(s)
        iters_left = 1

        # Prepare <dicts> to store running importance estimates and parameter-values before update
        if isinstance(model, ContinualLearner) and model.si_c>0:
            W = {}
            p_old = {}
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        W[n] = p.data.clone().zero_()
                        p_old[n] = p.data.clone()

        # Find [classes_so_far] (=list of all classes in tasks seen so far)
        classes_so_far = list(range(classes_per_task*task))

        # Reinitialize the model's parameters (if requested)
        if reinit:
            from define_models import init_params
            init_params(model, args)
            if generator is not None:
                init_params(generator, args)

        # If using CWR, reinitialize weights of output layer (or set them to 0, if using CWR+)
        if isinstance(model, ContinualLearner) and model.cwr:
            if model.cwr_plus:
                for n, p in model.classifier.named_parameters():
                    p.data.zero_()
            else:
                from define_models import init_params
                init_params(model.classifier, args)

        # Reset state of optimizer(s) for every task (if requested)
        if hasattr(model, "optim_type") and model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
            if generator is not None:
                generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        iters_to_use = iters
        # -if only the final task should be trained on:
        if only_last and not task==len(train_datasets):
            iters_to_use = 0
        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader and, if needed, create new one
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                         drop_last=True if epochs is None else False))
                iters_left = len(data_loader)

            # Collect data from current task
            x, y = next(data_loader)                                    #--> sample training data of current task
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
            #y = y.expand(1) if len(y.size())==1 else y                 #--> hack for if batch-size is 1

            # If requested, collect data to be replayed
            if not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay
            else:
                # -collect/generate the inputs
                if Current:
                    x_ = x  #--> use current task inputs
                else:
                    # -generate inputs representative of previous tasks
                    allowed_classes = list(range(classes_per_task * (task-1)))
                    x_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes, only_x=True)
                # -produce the targets
                # Get target scores & possibly labels (i.e., [scores_] / [y_]) -- use previous model, with no_grad()
                with torch.no_grad():
                    all_scores_ = previous_model.classify(x_)
                scores_ = all_scores_[:, :(classes_per_task*(task-1))]  #-> 0s for current task added in [loss_fn_kd]
                # -also get the 'hard target'
                _, y_ = torch.max(scores_, dim=1)
                # -only keep predicted y_/scores_ if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets=="hard") else None
                scores_ = scores_ if (model.replay_targets=="soft") else None

            # ---> Train MAIN MODEL
            loss_dict = model.train_a_batch(x, y=y, x_=x_, y_=y_, scores_=scores_, classes_so_far=classes_so_far,
                                            task=task, rnt=(1. if task==1 else 1./task) if rnt is None else rnt)

            # Update running parameter importance estimates in W
            if isinstance(model, ContinualLearner) and model.si_c>0:
                for gen_params in model.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, task=task)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index, task=task)

            # ---> Train GENERATOR
            if generator is not None:
                loss_dict = generator.train_a_batch(x, x_=x_, rnt=(1. if task==1 else 1./task) if rnt is None else rnt)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index)

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()


        ##----------> UPON FINISHING EACH TASK...

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and model.ewc_lambda>0:
            # -select allowed classes (which of the below two to use?)
            #allowed_classes = list(range(classes_per_task*(task-1), classes_per_task*task))
            allowed_classes = list(range(classes_per_task*task))
            # -estimate FI-matrix
            model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and model.si_c>0:
            model.update_omega(W, model.epsilon)

        # CWR: normalize output-layer weights and consolidate them
        if isinstance(model, ContinualLearner) and model.cwr:
            for n, p in model.classifier.named_parameters():
                # -get previous "stored version"
                n = n.replace('.', '__')
                p_stored = getattr(model, "{}_stored_version".format(n)).clone()
                # -get weights of classes in current epoch
                current_classes = list(range(classes_per_task*(task-1), classes_per_task*task))
                p_stored[current_classes] = p.detach().clone()[current_classes]
                # -if requested, standardize weights of classes in current epoch
                if model.cwr_plus:
                    classes_to_standardize = (list(range(model.classes)) if args.neg_samples=="all" else (
                        current_classes if args.neg_samples=="current" else list(range(classes_per_task * task))
                    ))
                    weights_mean = p_stored[classes_to_standardize].mean()
                    p_stored[current_classes] -= weights_mean
                # -consolidate (standardized) weights of classes in current epoch
                model.register_buffer('{}_stored_version'.format(n), p_stored)
                # -set weights of the model itself for testing
                p.data = p_stored.clone()

        # REPLAY: update source for replay
        if replay_mode in ("generative", "current"):
            previous_model = copy.deepcopy(model).eval()
            if replay_mode == "generative":
                Generative = True
                previous_generator = previous_model if (generator is None) else copy.deepcopy(generator).eval()
            elif replay_mode == 'current':
                Current = True


        # Freeze weights of hiden layers after first task?
        if utils.checkattr(args, 'freeze_after_first') and task==1:
            if utils.checkattr(args, "freeze_convE"):
                for param in model.convE.parameters():
                    param.requires_grad = False
                model.convE.frozen = True
            if utils.checkattr(args, "freeze_fcE"):
                for param in model.fcE.parameters():
                    param.requires_grad = False
                model.fcE.frozen = True

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)



def train_slda(model, train_datasets, batch_size=32, metric_cbs=list()):
    '''Train SLDA model on sequential data from [train_datasets].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''

    # Use cuda?
    device = model.device
    cuda = model.cuda

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):

        # Define data-loader (for SLDA there is no point in seeing same data twice, so perform single run over data)
        train_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
        iters = len(train_loader)

        if task==1 and (not model.covariance_type=="pure_streaming"):
            # If first task, do the "base fitting"
            # -initialize arrays for base init data because it must be provided all at once to SLDA
            base_init_data = torch.empty((len(train_dataset), model.num_features))
            base_init_labels = torch.empty(len(train_dataset)).long()
            # -put features into array since base init needs all features at once
            start = 0
            for batch_x, batch_y in train_loader:
                batch_size = batch_x.shape[0]
                end = start + batch_size
                base_init_data[start:end] = batch_x.view(batch_size, -1).to(device)
                base_init_labels[start:end] = batch_y.squeeze()
                start = end
            # -fit base initialization stage
            model.fit_slda_base(base_init_data, base_init_labels)
        else:
            # If not first task (or if we're doing "pure" streaming), do "streaming updates" to the SLDA parameters
            for batch_x, batch_y in train_loader:
                # -fit SLDA one example at a time
                batch_size = batch_x.shape[0]
                batch_x = batch_x.view(batch_size, -1)
                for x, y in zip(batch_x, batch_y):
                    model.fit_slda(x, y.view(1, ))

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)



def train_gen_classifiers(model, train_datasets, iters=2000, epochs=None, batch_size=32,
                          feature_extractor=None, loss_cbs=list(), sample_cbs=list()):

    # Use cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Loop over all tasks.
    for class_id, train_dataset in enumerate(train_datasets):

        # Initialize # iters left on data-loader(s)
        iters_left = 1

        if epochs is not None:
            data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader)*epochs

        # Define a tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                         drop_last=True if epochs is None else False))
                iters_left = len(data_loader)

            # Collect data
            x, y = next(data_loader)                                    #--> sample training data of current task
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
            #y = y.expand(1) if len(y.size())==1 else y                 #--> hack for if batch-size is 1

            # Preprocess, if required
            if feature_extractor is not None:
                with torch.no_grad():
                    x = feature_extractor(x)

            # Select model to be trained
            model_to_be_trained = getattr(model, "vae{}".format(class_id))

            # Train the VAE model of this class with this batch
            loss_dict = model_to_be_trained.train_a_batch(x)

            # Fire callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, class_id=class_id)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(model_to_be_trained, batch_index, class_id=class_id)

        # Close progres-bar(s)
        progress.close()



def train_from_gen(model, gen_model, iters, batch_size, loss_cbs=list(), eval_cbs=list()):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data sampled from [gen_model].

    [model]             model to optimize
    [gen_model]         model from which to sample training data
    [iters]             <int> number of iterations (i.e., batches) to train for
    [batch_size]        <int> number of samples per batch
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model'''

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    # Loop over all iterations
    for iteration in range(iters):

            iteration += 1

            # Sample training-data from [gen_model]
            data, y = gen_model.sample(size=batch_size, only_x=False)

            # Train the model on the generated batch
            loss_dict = model.train_a_batch(data, y=y)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break
