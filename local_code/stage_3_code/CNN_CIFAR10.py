from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

class CNN_CIFAR10(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)

        x = x.view(batch_size, 3, 32, 32)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

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
                eval_indices = np.random.choice(n_samples, n_samples, replace=False)
                X_eval = X_array[eval_indices]
                y_eval = y_array[eval_indices]

                with torch.no_grad():
                    y_pred = self.forward(torch.FloatTensor(X_eval).to(self.device))
                    pred_labels = y_pred.max(1)[1].cpu()

                accuracy_evaluator.data = {
                    'true_y': torch.LongTensor(y_eval),
                    'pred_y': pred_labels
                }
                accuracy = accuracy_evaluator.evaluate()
                print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {avg_loss:.6f}')

    def test(self, X):
        self.to(self.device)
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1].cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
