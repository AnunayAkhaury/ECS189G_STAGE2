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
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    target_accuracy = 0.85
    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.batch_size = 128

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.4)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, X):
        '''Forward propagation'''
        x = X.to(self.device)

        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        y_pred = self.softmax(x)

        return y_pred
    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

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
                    y_pred = self.forward(torch.FloatTensor(X_eval).to(self.device))
                    pred_labels = y_pred.max(1)[1].cpu()

                accuracy_evaluator.data = {
                    'true_y': torch.LongTensor(y_eval),
                    'pred_y': pred_labels
                }
                results = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Results:', results, 'Loss:', avg_loss)

                if results['accuracy'] >= self.target_accuracy and epoch >= 50:
                    print(f"Threshold reached ({results['accuracy']:.4f}), saving model and stopping.")
                    break

    def test(self, X):
        # do the testing, and result the result
        self.to(self.device)
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1].cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
