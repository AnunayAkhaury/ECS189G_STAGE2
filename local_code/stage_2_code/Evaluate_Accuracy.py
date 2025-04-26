'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Calculate all metrics
        accuracy = accuracy_score(true_y, pred_y)
        
        # Weighted metrics
        weighted_precision = precision_score(true_y, pred_y, average='weighted')
        weighted_recall = recall_score(true_y, pred_y, average='weighted')
        weighted_f1 = f1_score(true_y, pred_y, average='weighted')
        
        # Macro metrics
        macro_precision = precision_score(true_y, pred_y, average='macro')
        macro_recall = recall_score(true_y, pred_y, average='macro')
        macro_f1 = f1_score(true_y, pred_y, average='macro')
        
        # Micro metrics
        micro_precision = precision_score(true_y, pred_y, average='micro')
        micro_recall = recall_score(true_y, pred_y, average='micro')
        micro_f1 = f1_score(true_y, pred_y, average='micro')
        
        return {
            'accuracy': accuracy,
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'micro': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1
            }
        }
        
