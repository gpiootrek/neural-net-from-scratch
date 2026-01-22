import numpy as np
import matplotlib.pyplot as plt

from layer import Linear, Sigmoid, Tanh
from loss import MSE
from mlp import MLP
from optimizers import Adam
from model import Model

# 1. Przygotowanie danych (XOR)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # Oczekiwane wyjścia (kształt (4, 1))

# 2. Definicja sieci
# Wejście (2) -> Ukryta (4 neurony, Tanh) -> Wyjście (1 neuron, Sigmoid)
xor_network = MLP([
    Linear(n_inputs=2, n_outputs=4),
    Tanh(),
    Linear(n_inputs=4, n_outputs=1),
    Sigmoid()
])

# 3. Konfiguracja modelu
xor_model = Model(xor_network)
# Używamy MSE i dość wysokiego learning rate, bo problem jest prosty, a Adam jest stabilny
xor_model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.1))

# 4. Trening
print("--- Rozpoczynam trening XOR ---")
xor_model.fit(X_xor, y_xor, epochs=500, batch_size=4)

# 5. Weryfikacja
print("\n--- Wyniki XOR ---")
predictions = xor_model.predict(X_xor)
for x, y, pred in zip(X_xor, y_xor, predictions):
    # Zaokrąglamy predykcję, aby zobaczyć decyzję (0 lub 1)
    print(f"Wejście: {x}, Oczekiwane: {y[0]}, Predykcja: {pred[0]:.4f} -> Decyzja: {round(pred[0])}")
    
    
# 1. Generowanie danych (Sinus)
# 100 punktów losowych z zakresu -PI do PI
X_sin = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_sin = np.sin(X_sin)

# 2. Definicja sieci
# Sieć głębsza: Wejście(1) -> 16 (Tanh) -> 16 (Tanh) -> Wyjście(1) (Linear)
# Ostatnia warstwa jest liniowa (bez aktywacji), bo chcemy przewidywać wartości z zakresu [-1, 1] a nie prawdopodobieństwa.
sin_network = MLP([
    Linear(1, 16),
    Tanh(),
    Linear(16, 16),
    Tanh(),
    Linear(16, 1)
])

# 3. Konfiguracja modelu
sin_model = Model(sin_network)
sin_model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))

# 4. Trening
print("\n--- Rozpoczynam trening Sinus ---")
# Więcej epok, bo problem jest trudniejszy niż XOR
sin_model.fit(X_sin, y_sin, epochs=1000, batch_size=32)

# 5. Wizualizacja
print("\n--- Generowanie wykresu ---")
predictions_sin = sin_model.predict(X_sin)

plt.figure(figsize=(10, 6))
plt.scatter(X_sin, y_sin, color='blue', label='Dane Prawdziwe (Sinus)', alpha=0.5)
plt.scatter(X_sin, predictions_sin, color='red', label='Predykcje Sieci', s=10)
plt.title("Regresja: Aproksymacja funkcji Sinus")
plt.legend()
plt.grid(True)
plt.show()