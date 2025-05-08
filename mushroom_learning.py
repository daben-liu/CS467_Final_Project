import pandas as pd
from sklearn.model_selection import train_test_split


import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.preprocessing import LabelEncoder

OPTS = None

NEGATIVE_CLASS = 'p'
POSITIVE_CLASS = 'e'

def predict(w, X):
    temp = X.dot(w)
    y_pred = np.sign(temp)
    y_pred[y_pred ==0] = 1
    return y_pred

def train(X_train, y_train, lr=1e-1, num_iters=5000, l2_reg=0.0):
    N, D = X_train.shape
    w = np.zeros(D)
    for i in range(num_iters):
        pred = sigmoid(y_train * X_train.dot(w))
        differnces = y_train * (1 - pred)
        grad = -1/N * X_train.T.dot(differnces) + l2_reg * w
        w -= lr * grad
    return w

def evaluate(w, X, y, name):
    y_preds = predict(w, X)
    acc = np.mean(y_preds == y)
    print('    {} Accuracy: {}'.format(name, acc))
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=2)
    parser.add_argument('--num-iters', '-T', type=int, default=10000)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot-weights')
    return parser.parse_args()

def main():
    # Split data
    df = pd.read_csv('secondary_data.csv', sep=';')

    le = LabelEncoder()
    # removed "cap-diameter","stem-height","stem-width"
    categorical_cols = ["class","cap-shape","cap-surface","cap-color","does-bruise-or-bleed","gill-attachment","gill-spacing","gill-color","stem-root","stem-surface","stem-color","veil-type","veil-color","has-ring","ring-type","spore-print-color","habitat","season"]
    # Apply LabelEncoder to the specified column
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

    # Train with gradient descent
    w = train(X_train, y_train, lr=OPTS.learning_rate, num_iters=OPTS.num_iters, l2_reg=OPTS.l2)

    # Evaluate model
    train_acc = evaluate(w, X_train, y_train, 'Train')
    dev_acc = evaluate(w, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(w, X_test, y_test, 'Test')

    # Plot the weights
    if OPTS.plot_weights:
        img = w.reshape(28, 28)
        vmax = max(np.max(w), -np.min(w))
        plt.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.savefig(f'plot_{OPTS.plot_weights}.png')
        plt.show()


if __name__ == '__main__':
    OPTS = parse_args()
    main()