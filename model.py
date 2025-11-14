from mlp import MLP

class Model():
    
    def __init__(self, mlp: MLP):
        self.mlp = mlp
        
    def compile(self, optimizer='adam', loss='rmse', metrics=None):
        pass
    
    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        pass
    
    def evaluate(self, x_test, y_test):
        pass