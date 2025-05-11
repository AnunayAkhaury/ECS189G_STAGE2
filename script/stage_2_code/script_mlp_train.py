from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

train_path = '../../data/stage_2_data/train.csv'
test_path = '../../data/stage_2_data/test.csv'

train_data_loader = Dataset_Loader(train_path, 'train', 'Training data for MLP')
test_data_loader = Dataset_Loader(test_path, 'test', 'Test data for MLP')

train_data = train_data_loader.load()
test_data = test_data_loader.load()

mlp = Method_MLP('MLP', 'Multi-Layer Perceptron for Classification')

mlp.data = {
    'train': {
        'X': train_data['X'],
        'y': train_data['y']
    },
    'test': {
        'X': test_data['X'],
        'y': test_data['y']
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

plt.savefig('../../result/stage_2_result/training_convergence.png')
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
