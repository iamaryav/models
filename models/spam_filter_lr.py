# spam filter with logistic regression in numpy from scratch
# a toy example

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bce_loss(y_hat, y):
    # binary cross entropy loss
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7) # avoid log(0) -> -inf
    loss = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    loss = loss.mean()
    return loss

def forward(x, W, b):
    z = x @ W.T + b # (B, in_dim) @ (in_dim, out_dim) -> (B, out_dim)
    y_hat = sigmoid(z)
    return y_hat

def backward(x, y_hat, y):
    B = x.shape[0] # (B, in_dim)
    # loss -> sigmoid -> linear layer | bias

    # loss function 
    # dl / dy_hat
    dl_dy_hat = (y_hat - y) / y_hat * (1 - y_hat)

    # sigmoid functtion
    # dy_hat / dz
    dz = y_hat * (1 - y_hat) # (B, out_dim)

    # dl / dz = (dl_dy_hat * dy_hat_dz)
    # this we arrived by cancelling terms in BCE and sigmoid in chain rule
    # to see it do the diff by hand
    dl_dz = (y_hat - y) / B # (B, out_dim)

    # do manual backprop for the linear layer for better understanding
    # dl / dw = dl_dz * dz / dw
    dl_dw = dl_dz.T @ x # (out_dim, B) @ (B, in_dim) -> (out_dim, in_dim) # matches W
    
    # dl / db = dl_dz * dz/db
    dl_db = dl_dz.sum(axis=0) # (out_dim,)

    # gradient value for all the weights
    return dl_dw, dl_db

def gradient_check(x, w, y, b, eps=1e-5):
    y_hat = forward(x, w, b);
    dl_dw, dl_db = backward(x, y_hat, y)

    print(f"----- gradient check: W ------")
    print(f"w_shape: {w.shape}");
    for _ in range(3):
        # w_shape = (out_dim, in_dim) # (1, 3)
        i = np.random.randint(0, w.shape[0])
        j = np.random.randint(0, w.shape[1])

        # applying the central difference formula
        w_plus = w.copy(); w_plus[i, j] += eps
        w_minus = w.copy(); w_minus[i, j] -= eps

        loss_plus = bce_loss(forward(x, w_plus, b), y)
        loss_minus = bce_loss(forward(x, w_minus, b), y)
        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytical = dl_dw[i, j]

        rel_error = abs(numerical - analytical) / (abs(numerical) + abs(analytical) + 1e-12)
        print(f"W[{i},{j}] | numerical: {numerical:.8f} | analytical: {analytical:.8f} | rel_error: {rel_error:.2e}")

    print(f"----- gradient check: b ------")

if __name__ == "__main__":
    print("start...")
    # training data prep
    texts = [
    "free money now",
    "hey are you free tomorrow",
    "win free prize now",
    "let us meet tomorrow"
    ]
    labels = [1, 0, 1, 0]   # 1 = spam, 0 = not spam
    B, in_dim, out_dim = 4, 3, 1
    x = np.random.randn(B, in_dim) # (B, in_dim)
    y = np.random.randint(0, 2, (B, out_dim)).astype(float) # (B, out_dim)
    W = np.random.rand(out_dim, in_dim) * 0.01 # (out_dim, in_dim)
    b = np.zeros(out_dim)
    print(x)
    print(y)
    print(b)

    # forward pass
    y_hat = forward(x, W, b)
    loss = bce_loss(y_hat, y)
    print(f"y_hat shape: {y_hat.shape}")
    print(y_hat)
    print(f"loss: {loss}")

    # backward pass
    dl_dw, dl_db = backward(x, y_hat, y)
    print(f"dl_dw shape: {dl_dw.shape}")
    print(dl_dw)
    print(f"dl_db shape: {dl_db.shape}")
    print(dl_db)

    # checking gradient manually
    gradient_check(x, W, y, b)
