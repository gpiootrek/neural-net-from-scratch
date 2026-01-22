import numpy as np

class Model:
    def __init__(self, network):
        self.network = network
        self.loss = None
        self.optimizer = None

    def compile(self, loss, optimizer):
        """
        Konfiguruje model do treningu.
        """
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, x_train, y_train, epochs, batch_size):
        """
        Główna pętla treningowa implementująca Mini-Batch Gradient Descent.
        """
        num_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            # 1. Tasowanie danych (Shuffling) w każdej epoce
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            
            # 2. Iteracja po wsadach (Batches)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Wycięcie batcha
                batch_x = x_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # --- KROK A: Forward Pass ---
                y_pred = self.network.forward(batch_x)
                
                # --- KROK B: Obliczenie Kosztu ---
                loss_val = self.loss.forward(y_pred, batch_y)
                # Sumujemy stratę (ważoną wielkością batcha) do raportowania
                epoch_loss += loss_val * (end_idx - start_idx)

                # --- KROK C: Backward Pass ---
                # Najpierw inicjujemy proces backward w funkcji kosztu
                grad_loss = self.loss.backward()
                
                # Kluczowe: Zerowanie gradientów PRZED akumulacją nowych
                self.network.zero_grads()
                
                # Propagacja wsteczna przez sieć
                self.network.backward(grad_loss)

                # --- KROK D: Aktualizacja Wag ---
                self.optimizer.step(self.network.parameters(), self.network.grads())

            # Raportowanie średniego błędu w epoce
            avg_loss = epoch_loss / num_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    def predict(self, x):
        """
        Generuje predykcje dla nowych danych.
        """
        return self.network.forward(x)