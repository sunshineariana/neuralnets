# Imports
import datetime
import os.path

import numpy as np
from src.test_utils import get_preprocessed_data, visualize_weights, visualize_loss

# Softmax function
def softmax(X):
    """Compute softmax values for each set of scores in X"""
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

# Loss and gradient computation
def softmax_loss_and_grad(W: np.array, X: np.array, y: np.array, reg: float) -> tuple:
    """
    TODO 2:
    Compute softmax classifier loss and gradient dL/dW
    Do not forget about regularization!
    :param W: classifier weights (D, C)
    :param X: input features (N, D)
    :param y: class labels (N, )
    :param reg: regularisation strength
    :return: loss, dW
    """
    loss = 0.0
    dL_dW = np.zeros_like(W)
    N = len(X)
    # *****START OF YOUR CODE*****
    # 1. Forward pass, compute loss as sum of data loss and regularization loss [sum(W ** 2)]
    z = X.dot(W)
    softmax_probs = softmax(z)
    loss = -np.log(softmax_probs[range(N), y]).mean()
    loss += np.sum(W * W)
    # 2. Backward pass, compute intermediate dL/dZ
    dL_dZ = softmax_probs.copy()
    dL_dZ[range(N), y] -= 1
    dL_dZ /= N
    # 3. Compute data gradient dL/dW
    dL_dW = X.T.dot(dL_dZ)
    # 4. Compute regularization gradient
    dL_dW += (2 * W)
    # 5. Return loss and sum of data + reg gradients

    # *****END OF YOUR CODE*****
    return loss, dL_dW

# Classifier class
class SoftmaxClassifier:

    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-3, 
          num_iters=1000, batch_size=64, verbose=True):
        """Train classifier using stochastic gradient descent"""
        
        # Initialize weights
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        self.W = 0.01 * np.random.randn(n_features, n_classes)
        
        # Run SGD
        loss_history = []
        for it in range(num_iters):
            
            # Sample batch
            batch_idx = np.random.choice(n_samples, batch_size)
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            
            # Compute loss and gradient
            loss, grad = softmax_loss_and_grad(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)
            
            # Update weights
            self.W -= learning_rate * grad
            
            if verbose and i % 100 == 0:
                print(f"iter {i} / {num_iters}: loss {loss:.4f}")
                
        return loss_history
    
    def evaluate(self, X, y):
        """Evaluate classifier accuracy on data"""
        y_pred = np.argmax(X @ self.W, axis=1)
        acc = np.mean(y_pred == y)
        return acc
        
# Training loop
def train_softmax():
    # Hyperparameters
    lr = 1e-3
    reg = 1e-4  
    iters = 10000
    batch_size = 64

    # Load data
    X_train, y_train, X_test, y_test = get_preprocessed_data()
    
    # Train model
    clf = SoftmaxClassifier()
    t0 = datetime.datetime.now()
    loss_history = clf.train(X_train, y_train, lr, reg, iters, batch_size)
    t1 = datetime.datetime.now()
    dt = t1 - t0

    # Print results
    print(f"""
    #Softmax Training Report
    Time elapsed: {dt.seconds} sec
    lr: {lr}
    reg: {reg}
    iters: {iters}
    batch size: {batch_size}
    
    Final loss: {loss_history[-1]}
    Train accuracy: {clf.evaluate(X_train, y_train)}
    Test accuracy: {clf.evaluate(X_test, y_test)}
    """)
    
    # Visualize
    out_dir = 'output/softmax'
    visualize_weights(clf, out_dir)
    visualize_loss(loss_history, out_dir)
    
if __name__ == '__main__':
    train_softmax()