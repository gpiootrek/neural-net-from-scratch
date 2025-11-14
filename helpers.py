
def softmax(x):
    """Compute the softmax of vector x."""
    import numpy as np
    e_x = np.exp(x - np.max(x))
    
    return e_x / e_x.sum(axis=0)