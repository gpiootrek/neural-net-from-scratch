import numpy as np
import matplotlib.pyplot as plt

from layer import Linear, Sigmoid, Tanh
from loss import MSE
from mlp import MLP
from optimizers import Adam
from model import Model

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

xor_network = MLP([
    Linear(n_inputs=2, n_outputs=4),
    Tanh(),
    Linear(n_inputs=4, n_outputs=1),
    Sigmoid()
])

xor_model = Model(xor_network)
xor_model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.1))

print("--- Starting training for XOR problem ---")
xor_model.fit(X_xor, y_xor, epochs=500, batch_size=4)

print("\n--- XOR Results ---")
predictions = xor_model.predict(X_xor)
for x, y, pred in zip(X_xor, y_xor, predictions):
    print(f"Input: {x}, Expected: {y[0]}, Predicted: {pred[0]:.4f} -> Outcome: {round(pred[0])}")
    
    
X_sin = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_sin = np.sin(X_sin)

sin_network = MLP([
    Linear(1, 16),
    Tanh(),
    Linear(16, 16),
    Tanh(),
    Linear(16, 1)
])

sin_model = Model(sin_network)
sin_model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))

print("\n--- Starting training for Sine problem ---")
sin_model.fit(X_sin, y_sin, epochs=1000, batch_size=32)

print("\n--- Generating plot ---")
predictions_sin = sin_model.predict(X_sin)

plt.figure(figsize=(10, 6))
plt.scatter(X_sin, y_sin, color='blue', label='True Data (Sine)', alpha=0.5)
plt.scatter(X_sin, predictions_sin, color='red', label='Network Predictions', s=10)
plt.title("Regression: Approximation of the Sine Function")
plt.legend()
plt.grid(True)
plt.show()