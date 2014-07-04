import datetime
import numpy as np
import numpy.random as random
from learning import LearningMethod, select_data
from mnist import load_mnist
from scipy.optimize import fmin_l_bfgs_b

EPSILON = 1e-10

class Softmax:
    def __init__(self, num_labels, num_features, regularization=0.):
        self.weights = random.randn(num_labels, num_features + 1)
        self.regularization = regularization

    ''' calculate probabilities of each possibility, given the inputs '''
    def probabilities(self, inputs):
        with_one = np.insert(inputs.T, 0, 1., axis=0)
        rel_probs = np.exp(-self.weights.dot(with_one))
        net_probs = np.sum(rel_probs, 0)
        return rel_probs / net_probs
        
    ''' cost function is defined as negative log-likelihood of the data. thus,
        minimizing the cost is equivalent to maximizing the log-likelihood '''
    def cost(self, inputs, outputs):
        probs = self.probabilities(inputs)
        with_outputs = outputs.T * probs
        cost_standard = -np.sum(np.log(np.sum(with_outputs, 0))) / len(inputs)
        weight_squared = self.weights[:, 1:] ** 2.
        cost_regularized = np.sum(weight_squared) * self.regularization / 2.
        return cost_standard + cost_regularized

    def unflatten_params(self, unrolled):
        self.weights = unrolled.reshape(self.weights.shape)

    ''' need unrolled cost to pass into L-BFGS-B minimization algorithm '''
    def cost_unrolled(self, unrolled, inputs, outputs):
        self.unflatten_params(unrolled)
        return self.cost(inputs, outputs)
        
    def backpropagate(self, inputs, outputs):
        with_one = np.insert(inputs.T, 0, 1., axis=0)
        probs = self.probabilities(inputs)
        cost_deriv_standard = -(probs - outputs.T).dot(with_one.T) / len(inputs)
        return cost_deriv_standard + self.regularization * self.weights
        
    ''' need unrolled cost deriv to pass into L-BFGS-B minimization algorithm '''
    def cost_deriv_unrolled(self, unrolled, inputs, outputs):
        self.unflatten_params(unrolled)
        return self.backpropagate(inputs, outputs).ravel()
        
    ''' numerical cost deriv to validate backpropagation '''
    def cost_deriv(self, inputs, outputs):
        base_cost = self.cost(inputs, outputs)
        weight_derivs = np.zeros(self.weights.shape)
        for i, weight in enumerate(self.weights):
            for j, weight_row in enumerate(weight):
                self.weights[i][j] += EPSILON
                weight_derivs[i][j] = \
                    (self.cost(inputs, outputs) - base_cost) / EPSILON
                self.weights[i][j] -= EPSILON
        return weight_derivs

    def gradient_descent(self, inputs, outputs, learning_rate):
        weight_deriv = self.backpropagate(inputs, outputs)
        self.weights -= learning_rate * weight_deriv
    
    def l_bfgs_b(self, inputs, outputs, max_iter):
        unrolled = self.weights.ravel()
        bound_cost = lambda x: self.cost_unrolled(x, inputs, outputs)
        bound_cost_deriv = lambda x: self.cost_deriv_unrolled(x, inputs, outputs)
        optimal_unrolled, _, _ = fmin_l_bfgs_b(bound_cost, unrolled, bound_cost_deriv, maxiter=max_iter)
        self.weights = optimal_unrolled.reshape(self.weights.shape)
    
    def evaluate(self, inputs, outputs):
        probs = self.probabilities(inputs)
        predicted_outputs = probs.argmax(0)
        actual_outputs = outputs.argmax(1)
        comparison = [a == b for a, b in zip(predicted_outputs, actual_outputs)]
        num_correct = len(list(x for x in comparison if x))
        return float(num_correct) / float(len(inputs))
        
    def train(self, \
              training, \
              test, \
              batch_pct, \
              num_per_epoch, \
              num_epochs, \
              learning_method=LearningMethod('SGD', {'learning_rate' : 0.1}) \
             ):
        print 'Training started at', str(datetime.datetime.now())
        inputs, outputs = training
        for epoch in xrange(num_epochs):
            for _ in xrange(num_per_epoch):
                used_inputs, used_outputs = select_data(inputs, outputs, batch_pct)
                if learning_method.name == 'SGD':
                    self.gradient_descent(used_inputs, used_outputs, learning_method.paramsDict['learning_rate'])
                elif learning_method.name == 'L-BFGS-B':
                    self.l_bfgs_b(used_inputs, used_outputs, learning_method.paramsDict['max_iter'])
                else:
                    raise ValueError, 'SGD or L-BFGS-B are the only supported learning methods'
            pct_correct = self.evaluate(inputs, outputs) * 100.
            print 'Training epoch', str(epoch), ':', str(pct_correct), 'correct'
        print 'Training finished at', str(datetime.datetime.now())
        if test:
            pct_correct_test = self.evaluate(test[0], test[1]) * 100.
            print 'Test:', str(pct_correct_test), 'correct'
        
def mnist_softmax():
    training, test = load_mnist()
    random.seed(1)
    softmax = Softmax(10, 784, regularization = 1e-4)
    # L-BFGS-B seems to train more quickly than SGD, in that in the same amount
    # of time (100 iters for L-BFGS-B vs. 150 for SGD), it has higher accuracy
    # for training / validation / test
    softmax.train(training, \
                  test, \
                  1., \
                  1, \
                  1, \
                  LearningMethod('L-BFGS-B', {'max_iter' : 100})
                 )

if __name__ == '__main__':
    mnist_softmax()
