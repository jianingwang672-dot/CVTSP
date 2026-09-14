# Environment used to create checkpoint-195

- Python: 3.10.19
- PyTorch: 2.5.1+cu124
- PyTorch CUDA build: 12.4
- cuDNN: 9.1.0
- NVIDIA driver on the source host: 570.195.03
- Gurobi / gurobipy: 12.0.3
- NumPy: 2.2.6
- pandas: 2.3.3
- CVXPY: 1.7.5
- cvxpylayers: 0.1.9
- diffcp: 1.1.9
- PyYAML: 6.0.3
- loguru: 0.7.3
- matplotlib: 3.10.8

The remote host does not need an identical NVIDIA driver, but its driver and
PyTorch build must be CUDA-compatible. A practical setup is:

```bash
conda create -n cvtsp python=3.10 -y
conda activate cvtsp
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Configure the remote machine's own Gurobi license before launching. Never copy
or commit the source host's license file.
