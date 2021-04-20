#!/usr/bin/env bash


# Command to pre-train convolutional layers on CIFAR-10 for the CIFAR-100 benchmark
# (no need to do this yourself, these pre-trained conv-layers are provided in the repo)
#./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=s100N

# Load and pre-process the CORe50 dataset
#./preprocess_core50.py

# Gridsearches for selecting hyperparameters for some of the methods compared with (EWC, SI, AR1 & BI-R; --> Table A.1)
./compare_hyperParams.py --experiment=MNIST --seed=1
./compare_hyperParams.py --experiment=CORe50 --seed=1 --single-epochs --batch=1 --fc-layers=2 --z-dim=200 --fc-units=1024 --lr=0.0001
./compare_hyperParams.py --experiment=CIFAR100 --pre-convE --hidden --seed=1 --iters=5000
./compare_hyperParams.py --experiment=CIFAR100 --pre-convE --seed=1 --no-bir --iters=5000
./compare_hyperParams.py --experiment=CIFAR10 --tasks=5 --seed=1 --conv-type=resNet --fc-layers=1 --iters=5000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001

# Run the generative classifier (--> last row of Table 2)
./compare_multiple.py --experiment=MNIST --iters=1000 --fc-units=85 --z-dim=5 --fc-layers=3 --eval-s=10000 --n-seeds=10 --seed=2
./compare_multiple.py --experiment=CIFAR10 --iters=2500 --depth=3 --reducing-layers=3 --channels=15 --z-dim=100 --fc-layers=1 --n-seeds=10-eval-s=10000 --seed=2
./compare_multiple.py --experiment=CIFAR100 --iters=500 --hidden --pre-convE --fc-layers=2 --fc-units=85 --z-dim=20 --seed=2 --n-seeds=10 --eval-s=10000
./compare_multiple.py --experiment=CORe50 --single-epochs --z-dim=110 --fc-layers=1 --batch=1 --lr=0.0001 --eval-s=10000 --n-seeds=10 --seed=2

# Run all methods compared against (--> rest of Table 2)
./compare_all.py --experiment=MNIST --n-seeds=10 --seed=11 --c=1000 --lambda=1000000. --omega-max=0.01 --ar1-c=10 --dg-prop=0.
./compare_all.py --experiment=CIFAR10 --tasks=5 --n-seeds=10 --seed=11 --c=1 --lambda=10 --omega-max=0.1 --ar1-c=100 --conv-type=resNet --fc-layers=1 --iters=5000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --deconv-type=resNet --z-dim=100
./compare_all.py --experiment=CIFAR100 --pre-convE --hidden --iters=5000 --n-seeds=10 --seed=11 --c=1. --lambda=100. --omega-max=0.01 --ar1-c=100 --dg-prop=0.6
./compare_all.py --experiment=CIFAR100 --pre-convE --iters=5000 --n-seeds=10 --seed=11 --c=1. --lambda=100. --omega-max=10. --ar1-c=100
./compare_all.py --experiment=CORe50 --n-seeds=10 --seed=11 --single-epochs --batch=1 --fc-layers=2 --z-dim=200 --fc-units=1024 --lr=0.0001 --c=10 --lambda=10 --omega-max=0.1 --ar1-c=1. --dg-prop=0. --bir-c=0.01 --si-dg-prop=0.6

# Visualization of samples from generative models underlying the generative classifier (--> Figure 2)
./main_generative.py --experiment=MNIST --iters=1000 --fc-units=85 --z-dim=5 --fc-layers=3 --seed=2 --test --no-normal-eval
./main_generative.py --experiment=CIFAR10 --iters=2500 --depth=3 --reducing-layers=3 --channels=15 --z-dim=100 --fc-layers=1 --seed=2 --test --no-normal-eval
# (NOTE: these commands assume the Table 2 comparison is already run. If not, the `--test` flag should be removed)

# Comparison of generative classifier and offline trained discriminative classifier (--> Table 3)
./compare_replay.py --experiment=MNIST --iters=1000 --fc-units=85 --z-dim=5 --fc-layers=3 --replay-iters=10000 --seed=2 --n-seeds=10
./compare_replay.py --experiment=CIFAR10 --iters=2500 --depth=3 --reducing-layers=3 --channels=15 --z-dim=100 --fc-layers=1 --replay-iters=25000 --seed=2 --n-seeds=10
./compare_replay.py --experiment=CIFAR100 --iters=500 --hidden --pre-convE --fc-layers=2 --fc-units=85 --z-dim=20 --replay-iters=50000 --seed=2 --n-seeds=10
./compare_replay.py --experiment=CORe50 --single-epochs --z-dim=110 --fc-layers=1 --batch=1 --lr=0.0001 --replay-iters=104903 --seed=2 --n-seeds=10

# Effect of reducing number of importance samples on performance of generative classifier (--> Table 4)
./compare_s.py --experiment=MNIST --iters=1000 --fc-units=85 --z-dim=5 --fc-layers=3 --n-seeds=10 --seed=2
./compare_s.py --experiment=CIFAR100 --iters=500 --hidden --pre-convE --fc-layers=2 --fc-units=85 --z-dim=20 --seed=2 --n-seeds=10
./compare_s.py --experiment=CORe50 --single-epochs --z-dim=110 --fc-layers=1 --batch=1 --lr=0.0001 --n-seeds=10 --seed=2
./compare_s.py --experiment=CIFAR10 --iters=2500 --depth=3 --reducing-layers=3 --channels=15 --z-dim=100 --fc-layers=1 --n-seeds=10 --seed=2

