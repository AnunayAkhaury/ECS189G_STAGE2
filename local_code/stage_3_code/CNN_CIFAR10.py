'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class CNN_CIFAR10(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    target_accuracy = 0.85
    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, num_classes=10):
        self.channels_last = True

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        x = x.to(self.device)

        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []

        X_array = np.array(X)
        y_array = np.array(y)

        batch_size = 128
        n_samples = len(X_array)

        self.to(self.device)

        for epoch in range(self.max_epoch):
            epoch_loss = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X_array[indices]
            y_shuffled = y_array[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                X_tensor = torch.FloatTensor(X_batch).to(self.device)
                y_tensor = torch.LongTensor(y_batch).to(self.device)

                y_pred = self.forward(X_tensor)

                train_loss = loss_function(y_pred, y_tensor)
                epoch_loss += train_loss.item() * len(X_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)

            if epoch % 10 == 0:
                eval_size = min(n_samples, 128)
                eval_indices = np.random.choice(n_samples, eval_size, replace=False)
                X_eval = X_array[eval_indices]
                y_eval = y_array[eval_indices]

                with torch.no_grad():
                    X_eval_tensor = torch.FloatTensor(X_eval).to(self.device)
                    y_pred = self.forward(X_eval_tensor)
                    pred_labels = y_pred.max(1)[1].cpu()

                accuracy_evaluator.data = {
                    'true_y': torch.LongTensor(y_eval),
                    'pred_y': pred_labels
                }
                results = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Results:', results, 'Loss:', avg_loss)

                if results['accuracy'] >= self.target_accuracy and epoch > 50:
                    print(f"Threshold reached ({results['accuracy']:.4f}), saving model and stopping.")
                    print(self.loss_history)
                    break

    def test(self, X):
        self.to(self.device)
        X_array = np.array(X)
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.max(1)[1].cpu()


    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
