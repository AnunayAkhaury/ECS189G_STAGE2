'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os
import csv


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        train_data = self._load_csv_file('train.csv')
        test_data = self._load_csv_file('test.csv')
        return {
            'train': train_data,
            'test': test_data
        }
    
    def _load_csv_file(self, filename):
        X = []
        y = []
        filepath = os.path.join(self.dataset_source_folder_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist!")
            return {'X': X, 'y': y}
            
        print(f'Loading {filename}...')
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                # Convert all elements to integers
                elements = [int(x) for x in row]
                # First element is the label
                y.append(elements[0])
                # Remaining elements are features
                X.append(elements[1:])
        
        print(f'Loaded {len(X)} instances from {filename}')
        return {'X': X, 'y': y}