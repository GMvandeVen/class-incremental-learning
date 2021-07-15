import numpy as np
import torch
import visual.visdom
import visual.plt
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def validate(model, dataset, feature_extractor=None, batch_size=128, test_size=1024, verbose=True,
             allowed_classes=None, S=50):
    '''Evaluate the accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model.device if hasattr(model, 'device') else model._device()
    cuda = model.cuda if hasattr(model, 'cuda') else model._is_on_cuda()

    # If not provided, set [allowed_classes]
    if model.label=="GenClassifier" and allowed_classes is None:
        allowed_classes = list(range(model.classes))

    # Set model to eval()-mode
    model.eval()

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, 1 if model.label=="GenClassifier" else batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for x, y in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -pre-process
        if feature_extractor is not None:
            with torch.no_grad():
                x = feature_extractor(x.to(device))
        # -evaluate model (if requested, only on [allowed_classes])
        if model.label=="GenClassifier":
            predicted = model.classify(x, S=S, batch_size=batch_size, allowed_classes=allowed_classes)
        else:
            y = y-allowed_classes[0] if (allowed_classes is not None) else y
            with torch.no_grad():
                scores = model.classify(x.to(device))
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores.cpu(), 1)
        # -update statistics
        if model.label=="GenClassifier":
            if predicted==y.numpy():
                total_correct += 1
            total_tested += 1
        else:
            total_correct += (predicted == y).sum().item()
            total_tested += len(x)
    accuracy = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> Averge accuracy: {:.4f}'.format(accuracy))
    return accuracy


def test_accuracy(model, datasets, current_task, iteration, classes_per_task=None,
                  test_size=None, visdom=None, verbose=False, summary_graph=True):
    '''Evaluate accuracy of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [classes_per_task]  <int> number of active classes er task
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_tasks = len(datasets)

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    accs_taskIL = []
    accs_classIL = []
    for i in range(n_tasks):
        if (current_task is None) or (i+1 <= current_task):
            # -evaluate according to Task-IL scenario
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
            accs_taskIL.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                         allowed_classes=allowed_classes))
            # -evaluate according to Class-IL scenario
            allowed_classes = None if current_task is None else list(range(classes_per_task * current_task))
            accs_classIL.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                          allowed_classes=allowed_classes))
        else:
            accs_taskIL.append(0)
            accs_classIL.append(0)
    if current_task is None:
        average_accs_taskIL = sum([accs_taskIL[task_id] for task_id in range(n_tasks)]) / n_tasks
        average_accs_classIL = sum([accs_classIL[task_id] for task_id in range(n_tasks)]) / n_tasks
    else:
        average_accs_taskIL = sum([accs_taskIL[task_id] for task_id in range(current_task)]) / current_task
        average_accs_classIL = sum([accs_classIL[task_id] for task_id in range(current_task)]) / current_task

    # Print results on screen
    if verbose:
        print(' => ave accuracy (Task-IL):  {:.3f}'.format(average_accs_taskIL))
        print(' => ave accuracy (Class-IL): {:.3f}'.format(average_accs_classIL))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(
            accs_taskIL, names=names, title="acc Task-IL ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test accuracy"
        )
        if n_tasks>1 and summary_graph:
            visual.visdom.visualize_scalars(
                [average_accs_taskIL], names=["ave"], title="ave acc Task-IL ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test accuracy"
            )
        visual.visdom.visualize_scalars(
            accs_classIL, names=names, title="acc Class-IL ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test accuracy"
        )
        if n_tasks>1 and summary_graph:
            visual.visdom.visualize_scalars(
                [average_accs_classIL], names=["ave"], title="ave acc Class-IL ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test accuracy"
            )



####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####

def initiate_metrics_dict(n_tasks):
    '''Initiate <dict> with all measures to keep track of.'''
    metrics_dict = {}
    metrics_dict["average_TaskIL"] = []  # ave acc over tasks so far: Task-IL -> only classes in task
    metrics_dict["average_ClassIL"] = [] # ave acc over tasks so far: Class-IL-> all classes so far (up to trained task)
    metrics_dict["x_iteration"] = []     # total number of iterations so far
    metrics_dict["x_task"] = []          # number of tasks so far (indicating the task on which training just finished)
    # Accuracy matrices
    metrics_dict["acc per task (only classes in task)"] = {}
    metrics_dict["acc per task (all classes up to trained task)"] = {}
    metrics_dict["acc per task (all classes up to evaluated task)"] = {}
    metrics_dict["acc per task (all classes)"] = {}
    for i in range(n_tasks):
        metrics_dict["acc per task (only classes in task)"]["task {}".format(i+1)] = []
        metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(i + 1)] = []
        metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(i + 1)] = []
        metrics_dict["acc per task (all classes)"]["task {}".format(i + 1)] = []
    return metrics_dict


def intial_accuracy(model, datasets, metrics_dict, classes_per_task=None, test_size=None, verbose=False):
    '''Evaluate accuracy of a classifier (=[model]) on all tasks using [datasets] before any learning.'''

    n_tasks = len(datasets)

    accs_all_classes = []
    accs_only_classes_in_task = []
    accs_all_classes_upto_task = []

    for i in range(n_tasks):
        # -all classes
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None)
        accs_all_classes.append(accuracy)
        # -only classes in task
        allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes)
        accs_only_classes_in_task.append(accuracy)
        # -classes up to evaluated task
        allowed_classes = list(range(classes_per_task * (i + 1)))
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes)
        accs_all_classes_upto_task.append(accuracy)

    metrics_dict["initial acc per task (all classes)"] = accs_all_classes
    metrics_dict["initial acc per task (only classes in task)"] = accs_only_classes_in_task
    metrics_dict["initial acc per task (all classes up to evaluated task)"] = accs_all_classes_upto_task
    return metrics_dict


def metric_statistics(model, datasets, current_task, iteration, classes_per_task=None, metrics_dict=None,
                      test_size=None, verbose=False):
    '''Evaluate accuracy of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [metrics_dict]      None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task'''

    n_tasks = len(datasets)

    # Calculate accurcies per task in various ways
    accs_all_classes = []
    accs_all_classes_so_far = []
    accs_only_classes_in_task = []
    accs_all_classes_upto_task = []
    for i in range(n_tasks):
        # -all classes
        accuracy = validate(
            model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None,
        )
        accs_all_classes.append(accuracy)
        # -all classes up to trained task
        allowed_classes = list(range(classes_per_task * current_task))
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes)
        accs_all_classes_so_far.append(accuracy)
        # -all classes up to evaluated task
        allowed_classes = list(range(classes_per_task * (i+1)))
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes)
        accs_all_classes_upto_task.append(accuracy)
        # -only classes in that task
        allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
        accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes)
        accs_only_classes_in_task.append(accuracy)

    # Calcualte average accuracy over all tasks thus far
    average_accsTIL = sum([accs_only_classes_in_task[task_id] for task_id in range(current_task)]) / current_task
    average_accsCIL = sum([accs_all_classes_so_far[task_id] for task_id in range(current_task)]) / current_task

    # Append results to [metrics_dict]-dictionary
    for task_id in range(n_tasks):
        metrics_dict["acc per task (all classes)"]["task {}".format(task_id+1)].append(accs_all_classes[task_id])
        metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(task_id+1)].append(
            accs_all_classes_so_far[task_id]
        )
        metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(task_id+1)].append(
            accs_all_classes_upto_task[task_id]
        )
        metrics_dict["acc per task (only classes in task)"]["task {}".format(task_id+1)].append(
            accs_only_classes_in_task[task_id]
        )
    metrics_dict["average_TaskIL"].append(average_accsTIL)
    metrics_dict["average_ClassIL"].append(average_accsCIL)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen
    if verbose:
        print(' => ave accuracy (Task-IL):  {:.3f}'.format(average_accsTIL))
        print(' => ave accuracy (Class-IL): {:.3f}'.format(average_accsCIL))

    return metrics_dict



####--------------------------------------------------------------------------------------------------------------####

####----------------------------####
####----GENERATOR EVALUATION----####
####----------------------------####

def show_samples(model, config, pdf=None, visdom=None, size=32, sample_mode=None, title="Generated samples",
                 allowed_classes=None, allowed_domains=None, class_id=None):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    model.eval()

    # Generate samples from the model
    sample = model.sample(size, sample_mode=sample_mode, allowed_classes=allowed_classes,
                          allowed_domains=allowed_domains, only_x=True, class_id=class_id)
    # -correctly arrange pixel-values and move to cpu (if needed)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual.plt.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    if visdom is not None:
        mode = "" if sample_mode is None else "(mode = {})".format(sample_mode)
        visual.visdom.visualize_images(
            tensor=image_tensor, env=visdom["env"], nrow=nrow,
            title='Generated samples {} ({})'.format(mode, visdom["graph"]),
        )

