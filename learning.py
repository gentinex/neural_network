import numpy.random as random

class LearningMethod:
    def __init__(self, name, paramsDict={}):
        self.name = name
        self.paramsDict = paramsDict
    
''' select a subset of the data for training '''
def select_data(inputs, outputs, batch_pct=1.):
    assert batch_pct >= 0. and batch_pct <= 1.
    if batch_pct == 1.:
        used_inputs = inputs
        used_outputs = outputs
    else:
        num_inputs = len(inputs)
        num_selected = max(1, int(num_inputs * batch_pct))
        all_indices = xrange(num_inputs)
        random_indices = tuple(random.choice(all_indices, num_selected, False))
        used_inputs = inputs[random_indices, ...]
        used_outputs = outputs[random_indices, ...]
    return used_inputs, used_outputs
