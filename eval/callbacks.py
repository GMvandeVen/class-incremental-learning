import numpy as np
import visual.visdom
from . import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _eval_cb(log, test_datasets, visdom=None, iters_per_task=None, test_size=None, classes_per_task=None):
    '''Initiates function for evaluating performance of classifier (in terms of precision).

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task'''

    def eval_cb(classifier, batch, task=None, **kwargs):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if (task is None or task==1) else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.precision(classifier, test_datasets, task, iteration,
                               classes_per_task=classes_per_task, test_size=test_size, visdom=visdom)

    ## Return the callback-function (except if neither visdom or [precision_dict] is selected!)
    return eval_cb if (visdom is not None) else None


def _sample_cb(log, config, visdom=None, sample_size=64):
    '''Initiates function for evaluating samples of generative model.'''

    def sample_cb(generator, iter, class_id=None, **kwargs):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        if iter % log == 0:

            # Set model to evaluation-mode
            generator.eval()

            # Generate samples from the model
            sample = generator.sample(sample_size)
            # -correctly arrange pixel-values and move to cpu (if needed)
            image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
            # -denormalize images if needed
            if config['normalize']:
                image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

            # Plot generated images in [pdf] and/or [visdom]
            # -number of rows
            nrow = int(np.ceil(np.sqrt(sample_size)))
            visual.visdom.visualize_images(
                tensor=image_tensor, env=visdom["env"], nrow=nrow,
                title='Samples{} ({})'.format(" VAE-{}".format(class_id) if class_id is not None else "",
                                              visdom["graph"]),
            )

    # Return the callback-function (except if visdom is not selected!)
    return sample_cb if (visdom is not None) else None


##------------------------------------------------------------------------------------------------------------------##

################################################
## Callback-functions for calculating metrics ##
################################################

def _metric_cb(test_datasets, metrics_dict=None, iters_per_task=None, test_size=None, classes_per_task=None):
    '''Initiates function for calculating statistics required for calculating metrics.

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task'''

    def metric_cb(classifier, batch, task=1):
        '''Callback-function, to calculate statistics for metrics.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        evaluate.metric_statistics(classifier, test_datasets, task, iteration,
                                   classes_per_task=classes_per_task, metrics_dict=metrics_dict, test_size=test_size)

    ## Return the callback-function (except if no [metrics_dict] is selected!)
    return metric_cb if (metrics_dict is not None) else None



##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def _loss_cb(log=100, visdom=None, model=None, tasks=None, iters_per_task=None, epochs=None, progress_bar=True,
             virtual_epochs=False, name="CLASSIFIER"):
    '''Initiates function for keeping track of, and reporting on, the progress of training of classifier.'''

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            if virtual_epochs:
                task_stm = "" if (tasks is None) else " Virt Epoch: {}/{} |".format(task, tasks)
            else:
                task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else " Epoch: {}/{} |".format(epoch, epochs)
            bar.set_description(
                ' <{model}> |{t_stm}{e_stm} training loss: {loss:.3} |{prec}'.format(
                    model=name, t_stm=task_stm, e_stm=epoch_stm, loss=loss_dict['loss_total'],
                    prec=" training precision: {:.3} |".format(loss_dict['precision']) if name=="CLASSIFIER" else ""
                )
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # log the loss of the solver (to visdom)
        if (visdom is not None) and (iteration % log == 0):
            if name=="CLASSIFIER":
                if tasks is None or tasks==1:
                    # -overview of loss -- single task
                    plot_data = [loss_dict['pred']]
                    names = ['prediction']
                else:
                    # -overview of losses -- multiple tasks
                    plot_data = [loss_dict['pred']]
                    names = ['pred']
                    if model.ewc_lambda>0:
                        plot_data += [model.ewc_lambda * loss_dict['ewc']]
                        names += ['EWC (lambda={})'.format(model.ewc_lambda)]
                    if model.si_c>0:
                        plot_data += [model.si_c * loss_dict['si_loss']]
                        names += ['SI (c={})'.format(model.si_c)]
            else:
                # -overview of losses
                plot_data = list()
                names = list()
                plot_data += [loss_dict['recon']]
                names += ['Recon loss']
                plot_data += [loss_dict['variat']]
                names += ['Variat loss']
            # -log to visdom
            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="{}: loss ({})".format(name, visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb



def _VAE_loss_cb(log=100, visdom=None, model=None, tasks=None, iters_per_task=None, epochs=None, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else " Epoch: {}/{} |".format(epoch, epochs)
            bar.set_description(
                ' <GENERATOR>  |{t_stm}{e_stm} training loss: {loss:.3} |'
                    .format(t_stm=task_stm, e_stm=epoch_stm, loss=loss_dict['loss_total'])
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # plot training loss every [log]
        if (visdom is not None) and (iteration % log == 0):
            # -overview of losses
            plot_data = list()
            names = list()
            plot_data += [loss_dict['recon']]
            names += ['Recon']
            plot_data += [loss_dict['variat']]
            names += ['Variat']
            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="VAE: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb



def _gen_classifier_loss_cb(log, classes, visdom=None, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, class_id=0):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        ##--------------------------------PROGRESS BAR---------------------------------##
        if progress_bar and bar is not None:
            class_stm = "" if (classes is None) else " Class: {}/{} |".format(class_id+1, classes)
            bar.set_description(
                ' <GENERATOR>  |{c_stm} training loss: {loss:.3} |'
                    .format(c_stm=class_stm, loss=loss_dict['loss_total'])
            )
            bar.update(1)
        ##-----------------------------------------------------------------------------##

        # plot training loss every [log]
        if (visdom is not None) and (iter % log == 0):
            # -overview of losses
            plot_data = list()
            names = list()
            plot_data += [loss_dict['recon']]
            names += ['Recon loss']
            plot_data += [loss_dict['variat']]
            names += ['Variat loss']

            visual.visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iter,
                title="VAE-{}: loss ({})".format(class_id, visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb