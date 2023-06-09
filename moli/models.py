import torch
import torch.nn as nn

class AEE(nn.Module):
            def __init__(self):
                super(AEE, self).__init__()
                self.EnE = torch.nn.Sequential(
                    nn.Linear(IE_dim, h_dim1),
                    nn.BatchNorm1d(h_dim1),
                    nn.ReLU(),
                    nn.Dropout(rate1))
            def forward(self, x):
                output = self.EnE(x)
                return output

class AEM(nn.Module):
    def __init__(self):
        super(AEM, self).__init__()
        self.EnM = torch.nn.Sequential(
            nn.Linear(IM_dim, h_dim2),
            nn.BatchNorm1d(h_dim2),
            nn.ReLU(),
            nn.Dropout(rate2))
    def forward(self, x):
        output = self.EnM(x)
        return output    


class AEC(nn.Module):
    def __init__(self):
        super(AEC, self).__init__()
        self.EnC = torch.nn.Sequential(
            nn.Linear(IM_dim, h_dim3),
            nn.BatchNorm1d(h_dim3),
            nn.ReLU(),
            nn.Dropout(rate3))
    def forward(self, x):
        output = self.EnC(x)
        return output    

class OnlineTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTriplet, self).__init__()
        self.marg = marg
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets

class OnlineTestTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.marg = marg
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.FC = torch.nn.Sequential(
            nn.Linear(Z_in, 1),
            nn.Dropout(rate4),
            nn.Sigmoid())
    def forward(self, x):
        return self.FC(x)