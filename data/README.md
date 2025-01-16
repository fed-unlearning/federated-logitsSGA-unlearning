# Environment Setup
## 1. Python Environment
Make sure Python 3.8 or higher is installed on your system.
## 2. Dependencies
### PyTorch
This project uses PyTorch 1.12.0, which supports CUDA 11.6. You can install the correct version of PyTorch using the following command:
> pip install torch==1.12.0+cu116
### Other Dependencies
* torchvision==0.13.0
* numpy==1.23.5
# Data

MNIST & CIFAR-10 & SVHN datasets will be downloaded automatically by the torchvision package.

# Run the code
Use the following command to run the code. You can adjust the parameters to fit your needs:
> python fedsga-mnist-main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0  
> python fedsga-cifar-main.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0
> python fedsga-SVHN-main.py --dataset svhn --iid --num_channels 3 --model cnn --epochs 50 --gpu 0