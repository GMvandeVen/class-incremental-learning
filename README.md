# Class-Incremental Learning with Generative Classifiers
A PyTorch implementation of the CVPRW-2021 paper "Class-Incremental Learning with Generative Classifiers"
(a preprint version is available [here](...)).
Besides an implementation of the VAE-based generative classifier explored in this paper, this repository also provides 
implementations of all class-incremental learning mehods to which the generative classifier is compared
(i.e., DGR, BI-R, EWC, SI, CWR, CWR+, AR1, the 'labels trick' & SLDA).

## Installation & requirements
The current version of the code has been tested with `Python 3.6.9` on a Linux operating system with the following versions of PyTorch and Torchvision:
* `pytorch 1.7.1`
* `torchvision 0.8.2`
 
Assuming  Python and pip are set up, the Python-packages used by this code can be installed using:
```bash
pip install -r requirements.txt
```

The code in this repository itself does not need to be installed, but a number of scripts might need to be made executable:
```bash
chmod +x main_*.py compare_*.py commands.sh preprocess_core50.py
```

#### Loading and pre-processing the CORe50 dataset
The MNIST, CIFAR-10 and CIFAR-100 datasets will be automatically downloaded when their benchmarks are run for the first time.
By default, they will be downloaded to the folder `./store/datasets`, but this can be changed with the option `--data-dir`.
To use CORe50, the following command should be run first in order to download and pre-process the CORe50 dataset:
```bash
./preprocess_core50.py
```

More information about the CORe50 dataset can be found here: <https://vlomonaco.github.io/core50/>.


## Running comparisons from the paper
The script `commands.sh` provides step-by-step instructions for re-running the experiments reported in the paper.

Although it is possible to run this script as is, it will take long and it might be sensible to use fewer random seeds
or to parallellize the experiments.


## Running custom experiments
With this code it is also possible to run custom class-incremental learning experiments.

#### Generative classifier
Individual experiments with the VAE-based generative classifier can be run with `main_generative.py`.
The main options for this script are:
- `--experiment`: which dataset? (`MNIST`|`CIFAR10`|`CIFAR100`|`CORe50`)
- `--iters`: how many iterations per class?
- `--batch`: what mini-batch size to use?

For information on further options: `./main_generative.py -h`

#### Other class-incremental learning methods
Using `main_cl.py`, it is possible to run custom individual experiments with other class-incremental learrning methods.
It is also possible to combine some of the methods together.
The main options for this script are:
- `--experiment`: which dataset? (`MNIST`|`CIFAR10`|`CIFAR100`|`CORe50`)
- `--tasks`: how many tasks?
- `--iters`: how many iterations per task?
- `--batch`: what mini-batch size to use?

To run specific methods, the following can be used:
- Synaptic intelligenc (SI): `./main_cl.py --si --c=0.1`
- Elastic weight consolidation (EWC): `./main_cl.py --ewc --lambda=5000`
- Deep Generative Replay (DGR): `./main_cl.py --replay=generative`
- Brain-Inspired Replay (BI-R): `./main_cl.py --replay=generative --brain-inspired`
- CopyWeights with Re-init (CWR): `./main_cl.py --cwr --freeze-after-first --freeze-fcE --freeze-convE`
- CWR+: `./main_cl.py --cwr-plus --freeze-after-first --freeze-fcE --freeze-convE`
- AR1: `./main_cl.py --cwr-plus --si --reg-only-hidden --c=0.1 --omega-max=0.1`
- The 'labels trick': `./main_cl.py --neg-samples=current`
- Streaming LDA: `./main_cl.py --slda`

For information on further options: `./main_cl.py -h`

Note that this repository only supports class-incremental learning methods that do not store data.
PyTorch-implementations for several methods relying on a memory buffer with stored data
(e.g., Experience Replay, iCaRL, A-GEM) can be found here: <https://github.com/GMvandeVen/continual-learning>.


## On-the-fly plots during training
It is possible to track progress during training with on-the-fly plots. This feature requires `visdom`.
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main_generative.py` or `./main_cl.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.


### Citation
Please consider citing the accompanying paper if you use this code in your research:
```
@article{vandeven2021class,
  title={Class-incremental learning with generative classifiers},
  author={van de Ven, Gido M and Li, Zhe and Tolias, Andreas S},
  journal={arXiv preprint arXiv:...},
  year={2021}
}
```

### Acknowledgments
- The script for loading the CORe50 dataset has been based on https://github.com/Continvvm/continuum
- The code for SLDA has been based on https://github.com/tyler-hayes/Deep_SLDA

The research project from which this code originated has been supported by the 
Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA) via contract number 
HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of 
Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. Disclaimer: views and conclusions 
contained herein are those of the authors and should not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of DARPA, IARPA, DoI/IBC, or the U.S. Government.