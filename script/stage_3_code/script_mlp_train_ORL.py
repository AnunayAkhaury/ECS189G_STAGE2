import pickle
from matplotlib import pyplot as plt
from local_code.stage_3_code.CNN_ORL import CNN_ORL
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

f = open('../../data/stage_3_data/ORL', 'rb')
data = pickle.load(f)
f.close()

train = data['train']
test = data['test']

train_labels = [instance['label'] for instance in data['train']]
test_labels = [instance['label'] for instance in data['test']]

train_image = [instance['image'] for instance in data['train']]
test_image = [instance['image'] for instance in data['test']]


mlp = CNN_ORL('MLP', 'Multi-Layer Perceptron for ORL Classification')

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

plt.savefig('../../result/stage_3_result/training_convergence_ORL.png')
plt.close()

evaluate_obj.data = result
metrics = evaluate_obj.evaluate()

print('************ Overall Performance ************')
print(f"Accuracy: {metrics['accuracy']}")

print('\nWeighted Metrics:')
print(f"Precision: {metrics['weighted']['precision']}")
print(f"Recall: {metrics['weighted']['recall']}")
print(f"F1-score: {metrics['weighted']['f1']}")

print('\nMacro Metrics:')
print(f"Precision: {metrics['macro']['precision']}")
print(f"Recall: {metrics['macro']['recall']}")
print(f"F1-score: {metrics['macro']['f1']}")

print('\nMicro Metrics:')
print(f"Precision: {metrics['micro']['precision']}")
print(f"Recall: {metrics['micro']['recall']}")
print(f"F1-score: {metrics['micro']['f1']}")

print('************ Finish ************')
