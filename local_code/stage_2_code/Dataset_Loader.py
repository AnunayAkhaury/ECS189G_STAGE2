'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self,file_path , dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.file_path = file_path

    def load(self):
        print('loading data...')
        X = []
        y = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                elements = [int(i) for i in line.strip().split(',')]
                
                y.append(elements[0])
                
                X.append(elements[1:])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f'Loaded {len(X)} instances with {X.shape[1]} features')
        return {'X': X, 'y': y}