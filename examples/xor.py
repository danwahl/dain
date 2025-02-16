import numpy as np

from dain import add, matmul, mse, mse_grad, relu, relu_grad


def init_layer(in_dim: int, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize weights and biases for a layer."""
    weights = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2 / in_dim)
    biases = np.zeros((1, out_dim), dtype=np.float32)
    return weights, biases


class XORNetwork:
    def __init__(self, hidden_size: int):
        # Initialize weights and biases
        self.w1, self.b1 = init_layer(2, hidden_size)
        self.w2, self.b2 = init_layer(hidden_size, 1)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network."""
        # First layer
        hidden = matmul(x, self.w1)
        hidden = add(hidden, self.b1)
        hidden = relu(hidden)

        # Output layer
        output = matmul(hidden, self.w2)
        output = add(output, self.b2)

        return hidden, output

    def backward(
        self,
        x: np.ndarray,
        hidden: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        learning_rate: float = 0.1,
    ):
        """Backward pass to update weights."""
        # Compute gradients
        output_grad = mse_grad(output, target)

        # Gradient for output layer
        w2_grad = matmul(hidden.T, output_grad)
        b2_grad = output_grad.sum(axis=0, keepdims=True)
        hidden_grad = matmul(output_grad, self.w2.T)

        # Gradient for hidden layer
        hidden_grad = relu_grad(hidden, hidden_grad)
        w1_grad = matmul(x.T, hidden_grad)
        b1_grad = hidden_grad.sum(axis=0, keepdims=True)

        # Update weights and biases
        self.w1 -= learning_rate * w1_grad
        self.b1 -= learning_rate * b1_grad
        self.w2 -= learning_rate * w2_grad
        self.b2 -= learning_rate * b2_grad


def main():
    # Prepare XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Create and train network
    network = XORNetwork(hidden_size=8)

    # Training loop
    for epoch in range(100):
        # Forward pass
        hidden, output = network.forward(X)
        loss = mse(output, y)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            print("Predictions:", output.flatten())

        # Backward pass
        network.backward(X, hidden, output, y)

    # Final predictions
    _, output = network.forward(X)
    print("\nFinal predictions:")
    for inputs, pred in zip(X, output):
        print(f"{inputs} -> {pred[0]:.3f}")


if __name__ == "__main__":
    main()
