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
from proxy_a_distance import proxy_a_distance

def select_source():
    source_dataset1_name = 'mnist_m'
    source_dataset2_name = 'USPS'
    source_dataset3_name = 'SVHN'
    target_dataset_name = 'MNIST'
    source_image_root1 = os.path.join('dataset', source_dataset1_name)
    source_image_root2 = os.path.join('dataset', source_dataset2_name)
    source_image_root3 = os.path.join('dataset', source_dataset3_name)
    target_image_root = os.path.join('dataset', target_dataset_name)

    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 100
    image_size = 28
    n_epoch = 3

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
    train_list = os.path.join(source_image_root1, 'mnist_m_train_labels.txt')

    dataset_source1 = GetLoader(
        data_root=os.path.join(source_image_root1, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_source
    )

    dataloader_source1 = torch.utils.data.DataLoader(
        dataset=dataset_source1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    train_list = os.path.join(source_image_root2, 'USPS_train_labels.txt')

    dataset_source2 = GetLoader(
        data_root=os.path.join(source_image_root2, 'USPS_train'),
        data_list=train_list,
        transform=img_transform_source
    )

    dataloader_source2 = torch.utils.data.DataLoader(
        dataset=dataset_source2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    train_list = os.path.join(source_image_root3, 'SVHN_train_labels.txt')

    dataset_source3 = GetLoader(
        data_root=os.path.join(source_image_root3, 'SVHN_train'),
        data_list=train_list,
        transform=img_transform_source
    )

    dataloader_source3 = torch.utils.data.DataLoader(
        dataset=dataset_source3,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    minindex=0
    minvalue=99999999
    for i in range(3):
        my_net = CNNModel()

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

            len_dataloader = min(len(eval("dataloader_source"+str(i+1))), len(dataloader_target))
            data_source_iter = iter(eval("dataloader_source"+str(i+1)))
            data_target_iter = iter(dataloader_target)

            i = 0
            while i < len_dataloader:

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
                # dis, s, t = my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)

                i += 1

        data_source = data_source_iter.next()
        s_img, s_label = data_source
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
        dis, s, t = my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)
        Adis=proxy_a_distance(s,t)
        if(Adis<minvalue):
            minvalue=Adis
            minindex=i

    return i