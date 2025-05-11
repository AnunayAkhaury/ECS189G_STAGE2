import pickle
from matplotlib import pyplot as plt
from local_code.stage_3_code.CNN_CIFAR10 import CNN_CIFAR10  # make sure to save the modified class here
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

with open('../../data/stage_3_data/CIFAR', 'rb') as f:
    data = pickle.load(f)

train_labels = [instance['label'] for instance in data['train']]
test_labels = [instance['label'] for instance in data['test']]

train_images = [instance['image'] for instance in data['train']]
test_images = [instance['image'] for instance in data['test']]


# Initialize model
cnn = CNN_CIFAR10('CNN', 'CNN for CIFAR-10 Classification')

cnn.data = {
    'train': {
        'X': train_images,
        'y': train_labels
    },
    'test': {
        'X': test_images,
        'y': test_labels
    }
}

evaluate_obj = Evaluate_Accuracy('accuracy', '')

print('************ Start ************')
print('Training CNN on CIFAR-10...')
result = cnn.run()

loss_history = cnn.loss_history

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Convergence on CIFAR-10')
plt.grid(True)
plt.legend()
plt.show()

# Evaluate
evaluate_obj.data = result
metrics = evaluate_obj.evaluate()
