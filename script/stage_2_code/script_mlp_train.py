from local_code.stage_2_code.Dataset_Loader import Dataset_Loader

# Initialize the data loader
train_path = 'data/stage_2_data/train.csv'
test_path = 'data/stage_2_data/test.csv'  

# Create data loader instances
train_data_loader = Dataset_Loader(train_path, 'train', 'Training data for MLP')
test_data_loader = Dataset_Loader(test_path, 'test', 'Test data for MLP')

# Load the data
train_data = train_data_loader.load()
test_data = test_data_loader.load()

# Access the features and labels
X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

# Print the shapes to verify the data loading
print('Training data shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape:', y_test.shape)