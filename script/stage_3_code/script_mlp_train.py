import pickle
from matplotlib import pyplot as plt
from local_code.stage_3_code.CNN_MNIST import CNN_MNIST
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

f = open('../../data/stage_3_data/MNIST', 'rb')
data = pickle.load(f)
f.close()

train = data['train']
test = data['test']

train_labels = [instance['label'] for instance in data['train']]
test_labels = [instance['label'] for instance in data['test']]

train_image = [instance['image'] for instance in data['train']]
test_image = [instance['image'] for instance in data['test']]


mlp = CNN_MNIST('MLP', 'Multi-Layer Perceptron for MNIST Classification')

mlp.data = {
    'train': {
        'X': train_image,
        'y': train_labels
    },
    'test': {
        'X': test_image,
        'y': test_labels
    }
}

evaluate_obj = Evaluate_Accuracy('accuracy', '')

print('************ Start ************')
print('Training MLP...')
result = mlp.run()

loss_history = mlp.loss_history

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Convergence')
plt.grid(True)
plt.legend()

#plt.savefig('../../result/stage_2_result/training_convergence.png')
#plt.close()

evaluate_obj.data = result
metrics = evaluate_obj.evaluate()
