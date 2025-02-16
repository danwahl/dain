# dain

A minimalist Python CUDA library.

[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-blue)](https://github.com/danwahl/dain)
[![Documentation](https://img.shields.io/badge/View-Documentation-green)](https://danwahl.github.io/dain/)

## Features

- CUDA-accelerated neural network operations
- Clean Python interface with NumPy integration
- Core operations:
    - Matrix multiplication
    - Element-wise addition (with broadcasting)
    - ReLU activation
    - MSE loss
- Example implementations:
    - XOR neural network

## Requirements

- CUDA Toolkit
- Python 3.8+
- NumPy

## Installation

### From Source

1. Install system dependencies:
    ```bash
    apt install clang-format nvidia-cuda-toolkit
    ```

2. Install package with development dependencies:
    ```bash
    pip install ".[dev]"
    ```

3. Run tests:
    ```bash
    pytest tests
    ```

## Usage

Basic example using the library:

```python
import numpy as np
from dain import matmul, add, relu

# Initialize random matrices
a = np.random.randn(3, 2).astype(np.float32)
b = np.random.randn(2, 4).astype(np.float32)

# Perform operations
c = matmul(a, b)              # Matrix multiplication
d = relu(c)                   # ReLU activation
bias = np.zeros((1, 4))       # Broadcasted bias
output = add(d, bias)         # Add bias with broadcasting
```

### Examples

See `examples/xor.py` for a complete neural network implementation that learns the XOR function:

```bash
python examples/xor.py
```

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
