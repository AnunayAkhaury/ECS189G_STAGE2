import pickle
from matplotlib import pyplot as plt

f = open('../../data/stage_3_data/MNIST', 'rb')
data = pickle.load(f)
f.close()

train = data['train']
test = data['test']

train_labels = [instance['label'] for instance in data['train']]
test_labels = [instance['label'] for instance in data['test']]

train_image = [instance['image'] for instance in data['train']]
test_image = [instance['image'] for instance in data['test']]




