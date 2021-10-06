
import numpy as np
import torch
import torch.nn as nn
#from knockoff_threshold import knockoff_threshold
from knockpy.knockoff_stats import data_dependent_threshhold

class DeepYknock(nn.Module):
    def __init__(self, 
                 Xfeatures, 
                 Yfeatures, 
                 hidden_sizes=[32,32], 
                 batchsize=100,
                 num_epochs=200,
                 learningRate=1e-2,
                 lambda1=None,
                 lambda2=None,
                 bias=False,
                 initW=None,
                 verbose=True,
                 dropout=False,
                 normalize=False,
                 binaryX=False):


        super().__init__()

        # Infer n, p, set default lambda1, lambda2
        n = Xfeatures.shape[0]
        p = Xfeatures.shape[1]
        r = int(Yfeatures.shape[1] / 2)
        if not isinstance(Xfeatures, torch.Tensor):
            Xfeatures = torch.FloatTensor(Xfeatures)
        if not isinstance(Yfeatures, torch.Tensor):
            Yfeatures = torch.FloatTensor(Yfeatures)   
        self.Xfeatures = Xfeatures
        self.Yfeatures = Yfeatures
        self.p = p
        self.r = r
        self.n = n        
        
        if lambda1 is None:
            lambda1 = 10 * np.sqrt(np.log(r) / n)
        if lambda2 is None:
            lambda2 = 0
        self.lambda1=lambda1
        self.lambda2=lambda2
        
        if(initW is None):
            initW=torch.ones(2 * r)
        else:
            initW=torch.tensor(initW)
  
        self.Z_weight = nn.Parameter(initW)
        self.bias = bias
        self.verbose=verbose
        self.normalize=normalize
        self.binaryX = binaryX
        
        # Create MLP layers
        mlp_layers = [nn.Linear(r, hidden_sizes[0],bias = self.bias)]
        mlp_layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            mlp_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1],bias = self.bias))
            mlp_layers.append(nn.ReLU())

        mlp_layers.append(nn.Linear(hidden_sizes[-1], p,bias = self.bias))        
        self.mlp = nn.Sequential(*mlp_layers)
  
        
        self.num_epochs = num_epochs
        self.batchsize = min(r,batchsize)
        self.learningRate = learningRate
        
        
        if binaryX:
            self.criterion = nn.BCELoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5) #,  momentum=0.9)
        else:
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
            
        self.dropoutAct = dropout
        if dropout:
            self.dropout = nn.Dropout(0.25)
        
    def forward(self, Yfeatures):
        """
        NOTE: FEATURES CANNOT BE SHUFFLED
        """

        if not isinstance(Yfeatures, torch.Tensor):
            Yfeatures = torch.tensor(Yfeatures).float()     
                     
        # First pairwise weights
        features = self.normalize_Z_weight().unsqueeze(dim=0) * Yfeatures
        features = features[:, 0 : self.r] - features[:, self.r :]
        # Then apply MLP
        result = self.mlp(features.float())
        if self.dropoutAct:
            result = self.dropout(result)
        if self.binaryX:
            result = torch.sigmoid(result)

        return result   
        
 

    def normalize_Z_weight(self):

        # # First normalize
        if self.normalize:
            normalizer = torch.abs(self.Z_weight[0 : self.r]) + torch.abs(self.Z_weight[self.r :])
            return torch.cat([
                    torch.abs(self.Z_weight[0 : self.r]) / normalizer,
                    torch.abs(self.Z_weight[self.r :]) / normalizer,
                    ],dim=0,)
        else:
            return self.Z_weight



    def predict(self, features):
        """
        Wraps forward method, for compatibility
        with sklearn classes.
        """
        with torch.no_grad():
            return self.forward(features).numpy()

    def l1norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += torch.abs(parameter).sum()
        out += torch.abs(self.Z_weight).sum()  # This is just for stability
        return out

    def l2norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += (parameter ** 2).sum()
        out += (self.Z_weight ** 2).sum()
        return out

    def Z_regularizer(self):
        normZ = self.normalize_Z_weight()
        return -0.5 * torch.log(normZ).sum()

       
    def trainModel(self):
        n = self.n
        batchsize = self.batchsize
        for j in range(self.num_epochs):
            # Create batches, loop through
            ## Create random indices            
            inds = torch.randperm(n)      
            ## Iterate through and create batches
            i = 0
            batches = []            
            while i < self.n:
                batches.append([self.Xfeatures[inds][i : i + batchsize], self.Yfeatures[inds][i : i + batchsize]])
                i += batchsize
            ## 

            for Xbatch, Ybatch in batches:
                Xpred = self.forward(Ybatch)
                
                if self.binaryX:
                    loss = 0
                    for i in range(self.r):
                        loss += self.criterion(Xpred[:,i], Xbatch[:,i])
                else:
                    loss = self.criterion(Xpred, Xbatch)
                    


                # Add l1 and l2 regularization
                loss += self.lambda1 * self.l1norm()
                loss += self.lambda2 * self.l2norm()
                #loss += self.lambda1 * model.Z_regularizer()
                # Step
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # wandb.log({"loss": loss})
            if self.verbose and j % 10 == 0:
                print(f"At epoch {j}, mean predictive_loss is {loss / n}")
                
    def feature_importances(self, weight_scores=True):
       # weight_scores whether use 
       # the relative importance of the jth feature among all p features as weights
        with torch.no_grad():
            # Calculate weights from MLP
            if weight_scores:
                layers = list(self.mlp.named_children())
                W = layers[0][1].weight.detach().numpy().T   # p* hiddensize[0]
                for layer in layers[1:]:
                    if isinstance(layer[1], nn.ReLU):
                        continue
                    weight = layer[1].weight.detach().numpy().T
                    W = np.matmul(W, weight)
                    #W = W.squeeze(-1)
                #W = W.squeeze(-1)
                #print(W.shape)
                W = np.mean(W**2,axis=1)
            else:
                W = np.ones(self.r)
            # Multiply by Z weights
            feature_imp = self.normalize_Z_weight()[0 : self.r] * W
            knockoff_imp = self.normalize_Z_weight()[self.r :] * W
            return np.concatenate([feature_imp, knockoff_imp])
        
    def filter(self,fdr=0.2):
        Z = self.feature_importances()
        W = Z[0:self.r] - Z[self.r:]
        tau=data_dependent_threshhold(W,fdr=fdr)
        S = set(np.where(W>tau)[0]+1)
        return S