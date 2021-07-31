import torch.nn as nn
from functions import ReverseLayerF
import torch
import coral
import mmd_distance
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.Sfeature = nn.Sequential()
        self.Sfeature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.Sfeature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.Sfeature.add_module('f_pool1', nn.MaxPool2d(2))
        self.Sfeature.add_module('f_relu1', nn.ReLU(True))
        self.Sfeature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.Sfeature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.Sfeature.add_module('f_drop1', nn.Dropout2d())
        self.Sfeature.add_module('f_pool2', nn.MaxPool2d(2))
        self.Sfeature.add_module('f_relu2', nn.ReLU(True))

        self.Tfeature = nn.Sequential()
        self.Tfeature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.Tfeature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.Tfeature.add_module('f_pool1', nn.MaxPool2d(2))
        self.Tfeature.add_module('f_relu1', nn.ReLU(True))
        self.Tfeature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.Tfeature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.Tfeature.add_module('f_drop1', nn.Dropout2d())
        self.Tfeature.add_module('f_pool2', nn.MaxPool2d(2))
        self.Tfeature.add_module('f_relu2', nn.ReLU(True))

        # self.SfeatureNN = nn.Sequential()
        # self.SfeatureNN.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.SfeatureNN.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.SfeatureNN.add_module('c_relu1', nn.ReLU(True))
        # #
        # self.TfeatureNN = nn.Sequential()
        # self.TfeatureNN.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.TfeatureNN.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.TfeatureNN.add_module('c_relu1', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, Sinput_data,Tinput_data,alpha):
        Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
        Sfeature = self.Sfeature(Sinput_data)
        Sfeature = Sfeature.view(-1, 50 * 4 * 4)
        # Sfeature=self.SfeatureNN(Sfeature)

        Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
        Tfeature = self.Tfeature(Tinput_data)
        Tfeature = Tfeature.view(-1, 50 * 4 * 4)
        # Tfeature = self.TfeatureNN(Tfeature)

        feature=torch.cat((Sfeature,Tfeature),0)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(Sfeature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def MMD(self,sourceData,targetData):
        tMMD = mmd_distance.MMD_loss()
        sourceData = sourceData.expand(sourceData.data.shape[0], 3, 28, 28)
        targetData = targetData.expand(targetData.data.shape[0], 3, 28, 28)
        sourceFeature=self.Sfeature(sourceData)
        targetFeature = self.Tfeature(targetData)
        sourceFeature = sourceFeature.view(-1, 50 * 4 * 4)
        targetFeature = targetFeature.view(-1, 50 * 4 * 4)
        # sourceFeature=self.SfeatureNN(sourceFeature)
        # targetFeature = self.TfeatureNN(targetFeature)
        #MMDloss=torch.norm(sourceFeature-targetFeature)/(2*sourceFeature.shape[0])
        return tMMD(sourceFeature,targetFeature)

    def testS(self,Sinput_data):
        Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
        Sfeature = self.Sfeature(Sinput_data)
        Sfeature = Sfeature.view(-1, 50 * 4 * 4)
        # Sfeature = self.SfeatureNN(Sfeature)

        class_output = self.class_classifier(Sfeature)

        return class_output

    def testT(self,Tinput_data):
        Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
        Tfeature = self.Tfeature(Tinput_data)
        Tfeature = Tfeature.view(-1, 50 * 4 * 4)
        # Tfeature = self.TfeatureNN(Tfeature)

        class_output = self.class_classifier(Tfeature)

        return class_output

    def deepcoral(self,Sinput_data,Tinput_data):
        Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
        Sfeature = self.Sfeature(Sinput_data)
        Sfeature = Sfeature.view(-1, 50 * 4 * 4)
        # Sfeature = self.SfeatureNN(Sfeature)

        Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
        Tfeature = self.Tfeature(Tinput_data)
        Tfeature = Tfeature.view(-1, 50 * 4 * 4)
        # Tfeature = self.TfeatureNN(Tfeature)

        return coral.coral(Sfeature, Tfeature)

    def Observe(self,Sinput_data,Tinput_data):
        with torch.no_grad():
            MMD=mmd_distance.MMD_loss()
            Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
            Sfeature = self.Sfeature(Sinput_data)
            Sfeature = Sfeature.view(-1,50 * 4 * 4)
            # Sfeature = self.SfeatureNN(Sfeature)

            Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
            Tfeature = self.Tfeature(Tinput_data)
            Tfeature = Tfeature.view(-1,50 * 4 * 4)
            # Tfeature = self.TfeatureNN(Tfeature)

            # SfeatureMean = torch.mean(Sfeature,0)
            # TfeatureMean = torch.mean(Tfeature, 0)
            # temp1=torch.norm(Sfeature-SfeatureMean,dim=1)**2
            # temp2 = torch.norm(Tfeature - TfeatureMean, dim=1)**2
            #
            # SfeatureVar = torch.sqrt(torch.mean(temp1))
            # TfeatureVar = torch.sqrt(torch.mean(temp2))
            # return SfeatureMean,SfeatureVar,TfeatureMean,TfeatureVar
            dis=MMD(Sfeature,Tfeature)
            return dis,Sfeature.cpu().numpy(),Tfeature.cpu().numpy()

    def Visuialize(self,Sinput_data,Tinput_data):
        with torch.no_grad():
            Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
            Sfeature = self.Sfeature(Sinput_data)
            Sfeature = Sfeature.view(-1,50 * 4 * 4)
            # Sfeature = self.SfeatureNN(Sfeature)

            Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
            Tfeature = self.Tfeature(Tinput_data)
            Tfeature = Tfeature.view(-1,50 * 4 * 4)
            # Tfeature = self.TfeatureNN(Tfeature)

            Sfeature=Sfeature.cpu().numpy()
            Tfeature=Tfeature.cpu().numpy()

            S_tsne = TSNE(learning_rate=500).fit_transform(Sfeature)
            T_tsne = TSNE(learning_rate=500).fit_transform(Tfeature)
            return S_tsne,T_tsne

    def VisuializeS(self,Sinput_data):
        with torch.no_grad():
            Sinput_data = Sinput_data.expand(Sinput_data.data.shape[0], 3, 28, 28)
            Sfeature = self.Sfeature(Sinput_data)
            Sfeature = Sfeature.view(-1,50 * 4 * 4)
            # Sfeature = self.SfeatureNN(Sfeature)

            Sfeature=Sfeature.cpu().numpy()
            return Sfeature

    def VisuializeT(self,Tinput_data):
        with torch.no_grad():
            Tinput_data = Tinput_data.expand(Tinput_data.data.shape[0], 3, 28, 28)
            Tfeature = self.Tfeature(Tinput_data)
            Tfeature = Tfeature.view(-1,50 * 4 * 4)
            # Tfeature = self.TfeatureNN(Tfeature)

            Tfeature = Tfeature.cpu().numpy()

            return Tfeature