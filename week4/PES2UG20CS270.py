import numpy as np
from decimal import Decimal

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self
        
    def minkowski(self, x, y, p_value):
        total=sum(pow(abs(m-n), p_value) for m, n in zip(x, y))
        rootval=1/float(p_value)
        final=Decimal(total)**Decimal(rootval)
        mink_dist=float(round(final,3))    
        return mink_dist

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        dist=[]

        for i in range(x.shape[0]):
            a=x[i]
            val=[]
            for j in range(self.data.shape[0]): 
                b=self.data[j]
                val.append(self.minkowski(a,b,self.p))
            dist.append(val)

        return dist

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        dist=self.find_distance(x)
        knn = [[],[]]
        
        for i in range(len(dist)):
            index=[i for i in range(self.data.shape[0])]

            dist1=list(list(zip(*list(sorted(zip(dist[i],index)))))[0])
            dist2=list(list(zip(*list(sorted(zip(dist[i],index)))))[1])

            knn[0].append(dist1[0:self.k_neigh])
            knn[1].append(dist2[0:self.k_neigh])

        return knn

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        pred=[]
        index=self.k_neighbours(x)[1]
        
        for i in range(len(index)):
            f={}
            for j in range(len(index[i])):
                if self.target[index[i][j]] in f:
                    f[self.target[index[i][j]]] += 1
                else:
                    f[self.target[index[i][j]]] = 1 
            maxF=0
            maxK=None

            for i in range(min(f), max(f)+1):
                if f[i]>maxF:
                    maxF=f[i]
                    maxK=i                
            pred.append(maxK)

        return pred

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        predicted=self.predict(x)
        count=0
        l=len(predicted)

        for i in range(l):
            if(predicted[i]==y[i]):
                count+=1
        accuracy=round((count/l)*100,2)
        
        return accuracy
