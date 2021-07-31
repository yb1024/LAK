import torch.nn as nn
from functions import ReverseLayerF
import torch
import coral
import mmd_distance
from sklearn.manifold import TSNE


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

        self.class_adversarial0 = nn.Sequential()
        self.class_adversarial0.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial0.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial0.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial0.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial0.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial1 = nn.Sequential()
        self.class_adversarial1.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial1.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial1.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial1.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial1.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial2 = nn.Sequential()
        self.class_adversarial2.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial2.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial2.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial2.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial2.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial3 = nn.Sequential()
        self.class_adversarial3.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial3.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial3.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial3.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial3.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial4 = nn.Sequential()
        self.class_adversarial4.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial4.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial4.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial4.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial4.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial5 = nn.Sequential()
        self.class_adversarial5.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial5.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial5.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial5.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial5.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial6 = nn.Sequential()
        self.class_adversarial6.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial6.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial6.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial6.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial6.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial7 = nn.Sequential()
        self.class_adversarial7.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial7.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial7.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial7.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial7.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial8 = nn.Sequential()
        self.class_adversarial8.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial8.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial8.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial8.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial8.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.class_adversarial9 = nn.Sequential()
        self.class_adversarial9.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_adversarial9.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_adversarial9.add_module('c_relu1', nn.ReLU(True))
        self.class_adversarial9.add_module('c_fc2', nn.Linear(100, 2))
        self.class_adversarial9.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, Sinput_data,Tinput_data,ads0,ads1,ads2,ads3,ads4,ads5,ads6,ads7,ads8,ads9,adt0,adt1,adt2,adt3,adt4,adt5,adt6,adt7,adt8,adt9,alpha):
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
        Sclass_output = self.class_classifier(Sfeature)
        domain_output = self.domain_classifier(reverse_feature)

        ads0 = ads0.expand(ads0.data.shape[0], 3, 28, 28)
        ads1 = ads1.expand(ads1.data.shape[0], 3, 28, 28)
        ads2 = ads2.expand(ads2.data.shape[0], 3, 28, 28)
        ads3 = ads3.expand(ads3.data.shape[0], 3, 28, 28)
        ads4 = ads4.expand(ads4.data.shape[0], 3, 28, 28)
        ads5 = ads5.expand(ads5.data.shape[0], 3, 28, 28)
        ads6 = ads6.expand(ads6.data.shape[0], 3, 28, 28)
        ads7 = ads7.expand(ads7.data.shape[0], 3, 28, 28)
        ads8 = ads8.expand(ads8.data.shape[0], 3, 28, 28)
        ads9 = ads9.expand(ads9.data.shape[0], 3, 28, 28)
        adfs0=self.Sfeature(ads0)
        adfs1 = self.Sfeature(ads1)
        adfs2 = self.Sfeature(ads2)
        adfs3 = self.Sfeature(ads3)
        adfs4 = self.Sfeature(ads4)
        adfs5 = self.Sfeature(ads5)
        adfs6 = self.Sfeature(ads6)
        adfs7 = self.Sfeature(ads7)
        adfs8 = self.Sfeature(ads8)
        adfs9 = self.Sfeature(ads9)
        adfs0 = adfs0.view(-1, 50 * 4 * 4)
        adfs1 = adfs1.view(-1, 50 * 4 * 4)
        adfs2 = adfs2.view(-1, 50 * 4 * 4)
        adfs3 = adfs3.view(-1, 50 * 4 * 4)
        adfs4 = adfs4.view(-1, 50 * 4 * 4)
        adfs5 = adfs5.view(-1, 50 * 4 * 4)
        adfs6 = adfs6.view(-1, 50 * 4 * 4)
        adfs7 = adfs7.view(-1, 50 * 4 * 4)
        adfs8 = adfs8.view(-1, 50 * 4 * 4)
        adfs9 = adfs9.view(-1, 50 * 4 * 4)
        adt0 = adt0.expand(adt0.data.shape[0], 3, 28, 28)
        adt1 = adt1.expand(adt1.data.shape[0], 3, 28, 28)
        adt2 = adt2.expand(adt2.data.shape[0], 3, 28, 28)
        adt3 = adt3.expand(adt3.data.shape[0], 3, 28, 28)
        adt4 = adt4.expand(adt4.data.shape[0], 3, 28, 28)
        adt5 = adt5.expand(adt5.data.shape[0], 3, 28, 28)
        adt6 = adt6.expand(adt6.data.shape[0], 3, 28, 28)
        adt7 = adt7.expand(adt7.data.shape[0], 3, 28, 28)
        adt8 = adt8.expand(adt8.data.shape[0], 3, 28, 28)
        adt9 = adt9.expand(adt9.data.shape[0], 3, 28, 28)
        adft0 = self.Tfeature(adt0)
        adft1 = self.Tfeature(adt1)
        adft2 = self.Tfeature(adt2)
        adft3 = self.Tfeature(adt3)
        adft4 = self.Tfeature(adt4)
        adft5 = self.Tfeature(adt5)
        adft6 = self.Tfeature(adt6)
        adft7 = self.Tfeature(adt7)
        adft8 = self.Tfeature(adt8)
        adft9 = self.Tfeature(adt9)
        adft0 = adft0.view(-1, 50 * 4 * 4)
        adft1 = adft1.view(-1, 50 * 4 * 4)
        adft2 = adft2.view(-1, 50 * 4 * 4)
        adft3 = adft3.view(-1, 50 * 4 * 4)
        adft4 = adft4.view(-1, 50 * 4 * 4)
        adft5 = adft5.view(-1, 50 * 4 * 4)
        adft6 = adft6.view(-1, 50 * 4 * 4)
        adft7 = adft7.view(-1, 50 * 4 * 4)
        adft8 = adft8.view(-1, 50 * 4 * 4)
        adft9 = adft9.view(-1, 50 * 4 * 4)
        adfeature0 = torch.cat((adfs0, adft0), 0)
        adfeature1 = torch.cat((adfs1, adft1), 0)
        adfeature2 = torch.cat((adfs2, adft2), 0)
        adfeature3 = torch.cat((adfs3, adft3), 0)
        adfeature4 = torch.cat((adfs4, adft4), 0)
        adfeature5 = torch.cat((adfs5, adft5), 0)
        adfeature6 = torch.cat((adfs6, adft6), 0)
        adfeature7 = torch.cat((adfs7, adft7), 0)
        adfeature8 = torch.cat((adfs8, adft8), 0)
        adfeature9 = torch.cat((adfs9, adft9), 0)
        ref0 = ReverseLayerF.apply(adfeature0, alpha)
        ref1 = ReverseLayerF.apply(adfeature1, alpha)
        ref2 = ReverseLayerF.apply(adfeature2, alpha)
        ref3 = ReverseLayerF.apply(adfeature3, alpha)
        ref4 = ReverseLayerF.apply(adfeature4, alpha)
        ref5 = ReverseLayerF.apply(adfeature5, alpha)
        ref6 = ReverseLayerF.apply(adfeature6, alpha)
        ref7 = ReverseLayerF.apply(adfeature7, alpha)
        ref8 = ReverseLayerF.apply(adfeature8, alpha)
        ref9 = ReverseLayerF.apply(adfeature9, alpha)
        adversarial_output0 = self.class_adversarial0(ref0)
        adversarial_output1 = self.class_adversarial1(ref1)
        adversarial_output2 = self.class_adversarial2(ref2)
        adversarial_output3 = self.class_adversarial3(ref3)
        adversarial_output4 = self.class_adversarial4(ref4)
        adversarial_output5 = self.class_adversarial5(ref5)
        adversarial_output6 = self.class_adversarial6(ref6)
        adversarial_output7 = self.class_adversarial7(ref7)
        adversarial_output8 = self.class_adversarial8(ref8)
        adversarial_output9 = self.class_adversarial9(ref9)

        return Sclass_output, domain_output, adversarial_output0,adversarial_output1,adversarial_output2,adversarial_output3,\
               adversarial_output4,adversarial_output5,adversarial_output6,adversarial_output7,adversarial_output8,adversarial_output9

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