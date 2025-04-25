from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP

train_path = '../../data/stage_2_data/train.csv'
test_path = '../../data/stage_2_data/test.csv'

train_data_loader = Dataset_Loader(train_path, 'train', 'Training data for MLP')
test_data_loader = Dataset_Loader(test_path, 'test', 'Test data for MLP')

train_data = train_data_loader.load()
test_data = test_data_loader.load()

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

print('Training data shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape:', y_test.shape)

method_obj = Method_MLP('multi-layer perceptron', '')

