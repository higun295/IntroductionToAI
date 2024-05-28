import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

n_samples = 5000
minibatch_size = 50

n_feature = 2
n_class = 2
X, Y = make_moons(n_samples=5000, random_state=42, noise=0.1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=100, edgecolor="k", linewidths=2)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75, random_state=42)


def make_network(n_hidden=100):
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )
    return model


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def forward(x, model):
    h = x @ model['W1']
    h[h < 0] = 0
    prob = softmax(h @ model['W2'])
    return h, prob


def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0
    dW1 = xs.T @ dh
    return dict(W1=dW1, W2=dW2)


n_iteration = int(len(X_train) / minibatch_size)


def get_minibatch_grad(model, X_train, Y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, Y_train):
        h, y_pred = forward(x, model)

        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1

        err = y_true - y_pred
        xs.append(x)
        hs.append(h)
        errs.append(err)

    return backward(model, np.array(xs), np.array(hs), np.array(errs))


def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y


def get_minibatch(X, Y, minibatch_size):
    minibatches = []
    X, Y = shuffle(X, Y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        Y_mini = Y[i:i + minibatch_size]
        minibatches.append((X_mini, Y_mini))

    return minibatches


def GradientDescent(model, X_train, Y_train, minibatch_size, eta=1e-4):
    minibatches = get_minibatch(X_train, Y_train, minibatch_size)
    for idx in range(0, n_iteration):
        X_mini, Y_mini = minibatches[idx]
        grad = get_minibatch_grad(model, X_mini, Y_mini)

        for layer in grad:
            model[layer] += eta * grad[layer]
    return model


n_experiment = 100
learning_rate = 1e-4
accs = np.zeros(n_experiment)

for k in range(n_experiment):
    model = make_network()
    model = GradientDescent(model, X_train, Y_train, minibatch_size, learning_rate)
    Y_pred = np.zeros_like(Y_test)

    for i, x in enumerate(X_test):
        _, prob = forward(x, model)
        y = np.argmax(prob)
        Y_pred [i] = y

    accs[k] = (Y_pred == Y_test).sum() / Y_test.size

print('정확도 평균 : {}, 표준편차 : {}'.format(accs.mean(), accs.std()))