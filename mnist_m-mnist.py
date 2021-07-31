import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model3 import CNNModel
import numpy as np
from USPS_mnist_test import test
import coral
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as scio

if __name__ == "__main__":
    source_dataset_name = 'mnist_m'
    target_dataset_name = 'MNIST'
    source_image_root = os.path.join('dataset', source_dataset_name)
    target_image_root = os.path.join('dataset', target_dataset_name)
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 100
    image_size = 28
    n_epoch = 100

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_target = datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_target,
        download=False
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    # train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    #
    # dataset_target = GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     data_list=train_list,
    #     transform=img_transform_target
    # )
    train_list = os.path.join(source_image_root, 'mnist_m_train_labels.txt')

    dataset_source = GetLoader(
        data_root=os.path.join(source_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    # load model

    my_net = CNNModel()
    # my_net=torch.load('models/mnist_mnistm_model_epoch_1.pth')

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training

    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader - 1:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            input_imgs = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_labels = torch.LongTensor(batch_size)
            domain_labels = torch.zeros(batch_size)
            domain_labels = domain_labels.long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_imgs = input_imgs.cuda()
                class_labels = class_labels.cuda()
                domain_labels = domain_labels.cuda()

            input_imgs.resize_as_(s_img).copy_(s_img)
            class_labels.resize_as_(s_label).copy_(s_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            input_imgt = torch.FloatTensor(batch_size, 3, image_size, image_size)
            domain_labelt = torch.ones(batch_size)
            domain_labelt = domain_labelt.long()

            if cuda:
                t_img = t_img.cuda()
                input_imgt = input_imgt.cuda()
                domain_labelt = domain_labelt.cuda()

            input_imgt.resize_as_(t_img).copy_(t_img)

            class_output, domain_output = my_net(Sinput_data=input_imgs, Tinput_data=input_imgt, alpha=alpha)
            err_domain = loss_domain(domain_output, torch.cat((domain_labels, domain_labelt), 0))
            err_s_label = loss_class(class_output, class_labels)
            # 加MMD
            err = err_domain + err_s_label + my_net.MMD(input_imgs, input_imgt) + 0.5 * my_net.deepcoral(input_imgs,
                                                                                                             input_imgt)
            # DANN
            # err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            ###用于观察映射后的均值和方差a,c为均值，bd为方差
            # a,b,c,d=my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)
            # with torch.no_grad():
            #     distance=torch.norm(a-c)
            dis, _, _ = my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)

            i += 1

            # if epoch == n_epoch-1:
            #     S_tsne, T_tsne = my_net.Visuialize(Sinput_data=input_imgs, Tinput_data=input_imgt)
            #     classlabels=class_labels.cpu().numpy()
            #     plt.scatter(S_tsne[:, 0], S_tsne[:, 1], c=classlabels)
            #     plt.show()
            #     #plt.scatter(T_tsne[:, 0], T_tsne[:, 1], c=domain_labelt)

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_domain: %f, SM: %f, TM: %f, DS: %f' \
                  % (
                  epoch, i, len_dataloader, err_s_label.data.cpu().numpy(), err_domain.data.cpu().numpy(), 0, 0, dis))

        torch.save(my_net, '{0}/mnist_m_mnist_model_epoch_{1}.pth'.format(model_root, epoch))
        datas, labels = test(source_dataset_name, epoch)
        datat, labelt = test(target_dataset_name, epoch)
        if epoch % 10 == 0:
            data = np.vstack((datas, datat))
            label = np.vstack((labels, labelt))
            mytsne = TSNE(learning_rate=500).fit_transform(data)
            scio.savemat('FigureData4/data' + str(epoch) + '.mat', {'data': data})
            scio.savemat('FigureData4/label' + str(epoch) + '.mat', {'data': label})
            plt.figure(epoch + 2)
            plt.scatter(mytsne[:, 0], mytsne[:, 1], c=np.squeeze(label))
            plt.show()
            datas = None
            labels = None
            datat = None
            labelt = None
            data = None
            label = None

    # _, Sfeature, Tfeature = my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)
    # S_tsne = TSNE(learning_rate=500).fit_transform(Sfeature)
    # T_tsne = TSNE(learning_rate=500).fit_transform(Tfeature)
    print('done')