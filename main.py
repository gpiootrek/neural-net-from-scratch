from mlp import MLP

if __name__ == "__main__":
    # There will be logic for training/running neural net
    print("This is the main file.")
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])
    
    ITERATIONS = 20
    LEARNING_RATE = 0.05
    # BATCHES = ... // subset of inputs to feed forward the net (we do not want to use all every time as we may have millions of data inputs)

    # Gradient descent process
    for k in range(ITERATIONS):
        # Feed forward the net
        ypred = [n(x) for x in xs]
        loss = sum((yhat - y)**2 for y, yhat in zip(ys, ypred))

        # Reset gradients
        for p in n.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Update parameters based on grad and learning rate
        for p in n.parameters():
            p.data += -LEARNING_RATE * p.grad

        print(f'{k}. loss: {loss.data}')
    print(ypred)