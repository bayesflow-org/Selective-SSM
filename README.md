# Selective-SSM
Experimental exploration of applying Selective State Space Model (S-SSM) architectures within Amortized Bayesian Inference workflows

# Requirements:
- Linux / WSL
- Python 3.11+
- PyTorch 1.12+
- NVIDIA GPU
- CUDA 11.6+

# Installation
First create a new `conda` environment with at least Python 3.11 support  
```
conda create -n bf-ssm python=3.11
```

Install libraries (should use .yaml env for this)
```
conda install numpy pandas matplotlib seaborn ipykernel
```

The conda forge index is currently behind, so we'll have to use pip for the more prominent libraries
```
pip install torch
pip install keras
pip install triton
pip install mamba-ssm
```

Install development build of BayesFlow
```
pip install git+https://github.com/Chase-Grajeda/BayesFlow@ssm-wrapper
```

