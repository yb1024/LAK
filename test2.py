import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def test(dataset_name, epoch):
    assert dataset_name in ['USPS', 'SVHN']

    model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 100
    image_size = 28
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'USPS':
        test_list = os.path.join(image_root, 'USPS_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'USPS_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        test_list = os.path.join(image_root, 'SVHN_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'SVHN_test'),
            data_list=test_list,
            transform=img_transform_source
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'SVHN_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    data=[]
    label=[]

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        if dataset_name=='SVHN':
            class_output= my_net.testS(Sinput_data=input_img)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
            # if (epoch % 10 == 0 or epoch==99):
            #     temp = my_net.VisuializeS(Sinput_data=input_img)
            #     classlabels = class_label.cpu().numpy()
            #     classlabels=classlabels.reshape(-1,1)
            #     if i == 0:
            #         data = temp
            #         label=classlabels
            #     else:
            #         data = np.vstack((data, temp))
            #         label=np.vstack((label,classlabels))

        if dataset_name=='USPS':
            class_output = my_net.testT(Tinput_data=input_img)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
            # if(epoch % 10==0 or epoch==99):
            #     temp = my_net.VisuializeS(Sinput_data=input_img)
            #     classlabelt = class_label.cpu().numpy()
            #     classlabelt= classlabelt.reshape(-1,1)
            #     if i == 0:
            #         data = temp
            #         label=classlabelt
            #     else:
            #         data = np.vstack((data, temp))
            #         label=np.vstack((label,classlabelt))

        i += 1

    # if (dataset_name=='MNIST' and epoch % 10==0) or epoch==99:
    #     S_tsne = TSNE(learning_rate=500).fit_transform(data)
    #     plt.figure(epoch)
    #     plt.scatter(S_tsne[:, 0], S_tsne[:, 1], c=np.squeeze(label))
    #     plt.show()
    #
    # if (dataset_name=='mnist_m' and epoch % 10==0) or epoch==99:
    #     T_tsne = TSNE(learning_rate=500).fit_transform(data)
    #     plt.figure(epoch+1)
    #     plt.scatter(T_tsne[:, 0], T_tsne[:, 1], c=np.squeeze(label))
    #     plt.show()


    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))

    if epoch % 10==0:
        return data,label
    else:
        return [],[]
