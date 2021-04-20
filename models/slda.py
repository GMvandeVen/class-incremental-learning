import torch
from torch import nn


## NOTE: The implementation of SLDA was extended from: https://github.com/tyler-hayes/Deep_SLDA (retrieved: 17 Dec 2020)


####################################################################################################################


class StreamingLDA(nn.Module):
    '''Streaming LDA model (only first episode is not learned in a streaming fashion).'''

    def __init__(self, num_features, classes,
                 # -slda parameters
                 epsilon=1e-4, device='cpu', covariance='fixed'):
        '''Set up the model.

        Args:
            epsilon:        <float> value of the shrinkage parameter (default: 1e-4)
            covariance:     <str>: [fixed|identity|streaming|pure_streaming]
        '''
        super(StreamingLDA, self).__init__()

        # Model configurations
        self.label = "SLDA"
        self.device = device
        self.cuda = (self.device=='cuda')

        # SLDA parameters
        self.num_features = num_features
        self.classes = classes
        self.epsilon = epsilon
        self.covariance_type = covariance

        # Initialize SLDA class-means & counts
        self.muK = torch.zeros((classes, self.num_features)).to(self.device)
        self.cK = torch.zeros(classes).to(self.device)
        self.num_updates = 0
        self.new_data = True

        # If covariance-matrix is selected to be identity-matrix, already specify the precision matrix
        if self.covariance_type=="identity":
            self.Precision = torch.eye(self.num_features).to(self.device)
        # If covariance-matrix is to be calculated in pure streaming fashion, start with identity-matrix
        if self.covariance_type=="pure_streaming":
            self.Covariance = torch.eye(self.num_features).to(self.device)


    def _device(self):
        return self.device

    def _is_on_cuda(self):
        return self.cuda

    @property
    def name(self):
        return "slda-{}-c{}".format(self.covariance_type, self.classes)


    def fit_slda(self, x, y):
        '''Update the SLDA model based on a new sample (x,y).'''
        x = x.to(self.device)
        y = y.long().to(self.device)

        # Make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():
            # Update covariance-matrix (if requested)
            if self.covariance_type in ("streaming", "pure_streaming"):
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Covariance = (self.num_updates * self.Covariance + delta) / (self.num_updates + 1)

            # Update class-means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1


    def fit_slda_base(self, X, y):
        '''Compute & store for each class its mean. If requested, also compute covariance matrix.

        Args:
            X (torch.FloatTensor: [batch]x[num_features])   input features
            y (torch.LongTensor: [batch])                   class-labels
        '''
        X = X.to(self.device)

        # Compute & store means for each class
        for k in torch.unique(y):
            class_mean = X[y==k].mean(dim=0)  #-> mean (shape: [num_features])
            self.muK[k] = class_mean
            self.cK[k] = X[y==k].shape[0]     #-> keep count number of examples per class
        self.num_updates = X.shape[0]         #-> keep count total number of samples

        # If requested, compute & store covariance (and precision) matrix (which is shared between all classes)
        if self.covariance_type in ("fixed", "streaming"):
            from sklearn.covariance import OAS
            cov_estimator = OAS(assume_centered=True)
            cov_estimator.fit((X - self.muK[y]).cpu().numpy())
            self.Covariance = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)
            if self.covariance_type=="fixed":
                self.Precision = torch.pinverse(
                    (1-self.epsilon)*self.Covariance + self.epsilon*torch.eye(self.num_features).to(self.device)
                )


    def classify(self, x):
        '''For input [x], return predicted scores for all classes.

        Args:
            x (torch.FloatTensor: [batch]x[num_features] OR [batch]x[channels]x[image_size]x[image_size])
        '''
        # Turn generative model into linear discriminative classifier (if new train data arrived since last prediction)
        if self.new_data:
            self.new_data = False  #-> only need to convert gen model into classifier again if new data arrives
            # -if needed, compute the precision-matrix (= inverse of covariance-matrix)
            if self.covariance_type in ("streaming", "pure_streaming"):
                self.Precision = torch.pinverse(
                    (1-self.epsilon)*self.Covariance + self.epsilon*torch.eye(self.num_features).to(self.device)
                )
            # -parameters for linear classifier implied by the generative model:
            M = self.muK.transpose(1, 0)                             #-> shape: [num_features]x[classes]
            self.Weights = torch.matmul(self.Precision, M)           #-> shape: [num_features]x[classes]
            self.biases = 0.5 * torch.sum(M * self.Weights, dim=0)   #-> shape: [classes]

        # Make predictions for the provided samples, and return them
        with torch.no_grad():
            batch_size = x.shape[0]
            scores = torch.matmul(x.view(batch_size, -1), self.Weights) - self.biases     #-> shape: [batch]x[classes]
        return scores