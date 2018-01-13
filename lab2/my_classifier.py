from numpy import *
from skmultiflow.core.utils.utils import *
from skmultiflow.core.utils.data_structures import InstanceWindow
from sklearn.tree import DecisionTreeClassifier

class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        self.window_size = window_size
        self.max_models = max_models
        self.window = InstanceWindow(window_size)
        self.counter = 0

        # TODO

    def partial_fit(self, X, y=None, classes=None):
        # TODO 
        #if not initialized ...
            # Setup 
        # N.B.: The 'classes' option is not important for this classifier
        # HINT: You can build a decision tree model on a set of data like this:
        #       h = DecisionTreeClassifier()
        #       h.fit(X_batch,y_batch)
        #       self.H.append(h) # <-- and append it to the ensemble
       
        r, c = X.shape;
        #Feature matrix of a single sample
        #print("r : " + str(r));
        #Labels matrix of a single sample
        #print("c : " + str(c));
        
        for i in range(r):
            # if window is not created
            if self.window is None :
                # create the window
                self.window = InstanceWindow(self.window_size)
            
            #def add_element(self, X, y)
            #X: numpy.ndarray of shape (1, 1) 
            #Feature matrix of a single sample.
            #y: numpy.ndarray of shape (1, 1) 
            #Labels matrix of a single sample.
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))

            self.counter += 1

            #if there is no model for h, we create the decision tree for it
            if (self.h) is None:
                self.h = DecisionTreeClassifier()

            #if the window is full
            #we use the decision tree model to fit X and y
            if self.counter == self.window_size:
                #reinitialize the counter
                self.counter=0

                X_batch = self.window.get_attributes_matrix()
                y_batch = self.window.get_targets_matrix()
                #print(X_batch)
                #print(y_batch)
                self.h.fit(X_batch,y_batch)

                #if H is full
                #we take off the oldest model from it
                #and we push a the new one into H
                if(len(self.H) == self.max_models):
                    self.H.pop(0)
                self.H.append(self.h)

        return self



    def predict(self, X):
        # TODO 
        N,D = X.shape

        #we create the vector with the length of H
        if len(self.H) > 0 :
            res = zeros(len(self.H)) 
        else:
            0

        #do the prediction for every model in H
        for i in range(len(self.H)):
            res[i] = self.H[i].predict(X)

        # You also need to change this line to return your prediction instead of 0s:
        return res

















