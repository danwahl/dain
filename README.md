# dain

A minimalist Python CUDA library.

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

### Neural Network Example

See `examples/xor.py` for a complete neural network implementation that learns the XOR function:

```bash
python examples/xor.py
```

## API Reference

### Core Operations

- `matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray`
  - Matrix multiplication
  - Shapes: (M,K) @ (K,N) -> (M,N)

- `add(a: np.ndarray, b: np.ndarray) -> np.ndarray`
  - Element-wise addition with broadcasting
  - Supports (M,N) + (1,N) broadcasting

- `relu(x: np.ndarray) -> np.ndarray`
  - ReLU activation function
  - Element-wise max(0,x)

- `mse(pred: np.ndarray, target: np.ndarray) -> float`
  - Mean squared error loss
  - Returns scalar loss value

### Gradient Operations

- `relu_grad(x: np.ndarray, grad_in: np.ndarray) -> np.ndarray`
  - ReLU gradient for backpropagation

- `mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray`
  - MSE loss gradient for backpropagation

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
