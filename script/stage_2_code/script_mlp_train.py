from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

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

evaluate_obj.data = result
accuracy = evaluate_obj.evaluate()

print('************ Overall Performance ************')
print('MLP Accuracy:', accuracy)
print('************ Finish ************')
