import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from newmodel import CNNModel
import numpy as np
from mnist_mnist_m_newtest import test
import coral
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as scio

if __name__ == "__main__":
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('dataset', source_dataset_name)
    target_image_root = os.path.join('dataset', target_dataset_name)
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 220
    image_size = 28
    n_epoch = 150

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_source = datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=False
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
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
    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    adversarial_train_list = os.path.join(target_image_root, 'mnist_m_adversarial_train_labels.txt')

    adversarial_dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_adversarial_train'),
        data_list=adversarial_train_list,
        transform=img_transform_target
    )

    adversarial_dataloader = torch.utils.data.DataLoader(
        dataset=adversarial_dataset_target,
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
    loss_adversarial=torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    data_adversarial_iter=iter(adversarial_dataloader)
    data_adversarial_target = data_adversarial_iter.next()
    adversarial_img, adversarial_label = data_adversarial_target
    input_adversarial_t_0 = []
    input_adversarial_t_1 = []
    input_adversarial_t_2 = []
    input_adversarial_t_3 = []
    input_adversarial_t_4 = []
    input_adversarial_t_5 = []
    input_adversarial_t_6 = []
    input_adversarial_t_7 = []
    input_adversarial_t_8 = []
    input_adversarial_t_9 = []

    label_adversarial_t_0 = []
    label_adversarial_t_1 = []
    label_adversarial_t_2 = []
    label_adversarial_t_3 = []
    label_adversarial_t_4 = []
    label_adversarial_t_5 = []
    label_adversarial_t_6 = []
    label_adversarial_t_7 = []
    label_adversarial_t_8 = []
    label_adversarial_t_9 = []

    adversarial_batch_size = len(adversarial_img)

    # input_img_adversarial = torch.FloatTensor(adversarial_batch_size, 3, image_size, image_size)
    # adversarial_labelt = torch.LongTensor(adversarial_batch_size)
    adversarial_label_temp=adversarial_label.numpy()
    adversarial_img_temp=adversarial_img.numpy()
    for i,label in enumerate(adversarial_label_temp):
        temp='input_adversarial_t_'+str(label)
        if eval(temp)==[]:
            a = adversarial_img_temp[i]
            a=a.reshape(1,3,28,28)
            temp = 'input_adversarial_t_' + str(label) + '=a'
            exec(temp)
        else:
            a=np.concatenate((eval(temp),adversarial_img_temp[i].reshape(1,3,28,28)),0)
            temp = 'input_adversarial_t_' + str(label)+'=a'
            exec(temp)

    for i,label in enumerate(adversarial_label_temp):
        temp='label_adversarial_t_'+str(label)
        a = adversarial_label_temp[i]
        temp = 'label_adversarial_t_' + str(label) + '.append(1)'
        exec(temp)
        # else:
        #     a=np.concatenate((eval(temp),adversarial_img_temp[i].reshape(1,3,28,28)),0)
        #     temp = 'input_adversarial_t_' + str(label)+'=a'
        #     exec(temp)

    for i in range(10):
        temp='input_adversarial_t_' + str(i)+'=torch.tensor(input_adversarial_t_'+str(i)+')'
        exec(temp)
        temp = 'input_adversarial_t_' + str(i)+'='+'input_adversarial_t_' + str(i)+'.cuda()'
        exec(temp)

    for i in range(10):
        temp='label_adversarial_t_' + str(i)+'=torch.tensor(label_adversarial_t_'+str(i)+')'
        exec(temp)
        temp = 'label_adversarial_t_' + str(i)+'='+'label_adversarial_t_' + str(i)+'.cuda()'
        exec(temp)

    # if cuda:
    #     adversarial_img = adversarial_img.cuda()
    #     adversarial_label = adversarial_label.cuda()
    #     input_img_adversarial = input_img_adversarial.cuda()
    #     adversarial_labelt = adversarial_labelt.cuda()
    #
    # input_img_adversarial.resize_as_(adversarial_img).copy_(adversarial_img)
    # adversarial_labelt.resize_as_(adversarial_label).copy_(adversarial_label)

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

            input_adversarial_s_0 = []
            input_adversarial_s_1 = []
            input_adversarial_s_2 = []
            input_adversarial_s_3 = []
            input_adversarial_s_4 = []
            input_adversarial_s_5 = []
            input_adversarial_s_6 = []
            input_adversarial_s_7 = []
            input_adversarial_s_8 = []
            input_adversarial_s_9 = []

            label_adversarial_s_0 = []
            label_adversarial_s_1 = []
            label_adversarial_s_2 = []
            label_adversarial_s_3 = []
            label_adversarial_s_4 = []
            label_adversarial_s_5 = []
            label_adversarial_s_6 = []
            label_adversarial_s_7 = []
            label_adversarial_s_8 = []
            label_adversarial_s_9 = []

            adversarial_batch_size = len(adversarial_img)

            # input_img_adversarial = torch.FloatTensor(adversarial_batch_size, 3, image_size, image_size)
            # adversarial_labelt = torch.LongTensor(adversarial_batch_size)
            adversarial_label_temp = s_label.numpy()
            adversarial_img_temp = s_img.numpy()
            for j, label in enumerate(adversarial_label_temp):
                temp = 'input_adversarial_s_' + str(label)
                if eval(temp) == []:
                    a = adversarial_img_temp[j]
                    a = a.reshape(1, 1, 28, 28)
                    temp = 'input_adversarial_s_' + str(label) + '=a'
                    exec(temp)
                else:
                    a = np.concatenate((eval(temp), adversarial_img_temp[j].reshape(1, 1, 28, 28)), 0)
                    temp = 'input_adversarial_s_' + str(label) + '=a'
                    exec(temp)

            for j, label in enumerate(adversarial_label_temp):
                temp = 'label_adversarial_s_' + str(label)
                a = adversarial_label_temp[j]
                temp = 'label_adversarial_s_' + str(label) + '.append(0)'
                exec(temp)
                # else:
                #     a=np.concatenate((eval(temp),adversarial_img_temp[i].reshape(1,3,28,28)),0)
                #     temp = 'input_adversarial_t_' + str(label)+'=a'
                #     exec(temp)

            for j in range(10):
                temp = 'input_adversarial_s_' + str(j) + '=torch.tensor(input_adversarial_s_' + str(j) + ')'
                exec(temp)
                temp = 'input_adversarial_s_' + str(j) + '=' + 'input_adversarial_s_' + str(j) + '.cuda()'
                exec(temp)

            for j in range(10):
                temp = 'label_adversarial_s_' + str(j) + '=torch.tensor(label_adversarial_s_' + str(j) + ')'
                exec(temp)
                temp = 'label_adversarial_s_' + str(j) + '=' + 'label_adversarial_s_' + str(j) + '.cuda()'
                exec(temp)

            if label_adversarial_s_0.shape[0]==0 or label_adversarial_s_1.shape[0]==0 or label_adversarial_s_2.shape[0]==0 or label_adversarial_s_3.shape[0]==0 or label_adversarial_s_4.shape[0]==0 or label_adversarial_s_5.shape[0]==0 or label_adversarial_s_6.shape[0] == 0 or label_adversarial_s_7.shape[0]==0 or label_adversarial_s_8.shape[0]==0 or label_adversarial_s_9.shape[0]==0:
                i += 1
                continue

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



            # adversarial_dataloader
            # data_adversarial_target = data_adversarial_iter.next()
            # adversarial_img, adversarial_label = data_adversarial_target
            #
            # adversarial_batch_size = len(adversarial_img)
            #
            # input_img_adversarial = torch.FloatTensor(adversarial_batch_size, 3, image_size, image_size)
            # adversarial_labelt = torch.LongTensor(adversarial_batch_size)
            #
            # if cuda:
            #     adversarial_img = adversarial_img.cuda()
            #     adversarial_label = adversarial_label.cuda()
            #     input_img_adversarial = input_img_adversarial.cuda()
            #     adversarial_labelt=adversarial_labelt.cuda()
            #
            # input_img_adversarial.resize_as_(adversarial_img).copy_(adversarial_img)
            # adversarial_labelt.resize_as_(adversarial_label).copy_(adversarial_label)
            # ST0 = torch.cat((input_adversarial_s_0, input_adversarial_t_0), 0)
            # ST1 = torch.cat((input_adversarial_s_1, input_adversarial_t_1), 0)
            # ST2 = torch.cat((input_adversarial_s_2, input_adversarial_t_2), 0)
            # ST3 = torch.cat((input_adversarial_s_3, input_adversarial_t_3), 0)
            # ST4 = torch.cat((input_adversarial_s_4, input_adversarial_t_4), 0)
            # ST5 = torch.cat((input_adversarial_s_5, input_adversarial_t_5), 0)
            # ST6 = torch.cat((input_adversarial_s_6, input_adversarial_t_6), 0)
            # ST7 = torch.cat((input_adversarial_s_7, input_adversarial_t_7), 0)
            # ST8 = torch.cat((input_adversarial_s_8, input_adversarial_t_8), 0)
            # ST9 = torch.cat((input_adversarial_s_9, input_adversarial_t_9), 0)

            for j in range(10):
                temp1='label_adversarial_s_'+str(j)
                temp2='label_adversarial_t_'+str(j)
                if eval(temp1).shape[0]==0 and eval(temp2).shape[0]!=0:
                    exec('lST'+str(j)+'='+temp2)
                if eval(temp1).shape[0]!=0 and eval(temp2).shape[0]==0:
                    exec('lST'+str(j)+'='+temp1)
                if eval(temp1).shape[0]!=0 and eval(temp2).shape[0]!=0:
                    exec('lST'+str(j)+'=torch.cat(('+temp1+','+temp2+'),0)')
            lST0 = torch.cat((label_adversarial_s_0, label_adversarial_t_0), 0)
            lST1 = torch.cat((label_adversarial_s_1, label_adversarial_t_1), 0)
            lST2 = torch.cat((label_adversarial_s_2, label_adversarial_t_2), 0)
            lST3 = torch.cat((label_adversarial_s_3, label_adversarial_t_3), 0)
            lST4 = torch.cat((label_adversarial_s_4, label_adversarial_t_4), 0)
            lST5 = torch.cat((label_adversarial_s_5, label_adversarial_t_5), 0)
            lST6 = torch.cat((label_adversarial_s_6, label_adversarial_t_6), 0)
            lST7 = torch.cat((label_adversarial_s_7, label_adversarial_t_7), 0)
            lST8 = torch.cat((label_adversarial_s_8, label_adversarial_t_8), 0)
            lST9 = torch.cat((label_adversarial_s_9, label_adversarial_t_9), 0)


            class_output, domain_output, adversarial_output0,adversarial_output1,adversarial_output2,adversarial_output3,\
            adversarial_output4,adversarial_output5,adversarial_output6,adversarial_output7,adversarial_output8,\
            adversarial_output9 = my_net(input_imgs, input_imgt,input_adversarial_s_0,input_adversarial_s_1,input_adversarial_s_2,input_adversarial_s_3
                                         ,input_adversarial_s_4,input_adversarial_s_5,input_adversarial_s_6,input_adversarial_s_7,input_adversarial_s_8,
                                         input_adversarial_s_9,input_adversarial_t_0,input_adversarial_t_1,input_adversarial_t_2,input_adversarial_t_3,
                                         input_adversarial_t_4,input_adversarial_t_5,input_adversarial_t_6,input_adversarial_t_7,input_adversarial_t_8,
                                         input_adversarial_t_9,alpha=alpha)

            err_domain = loss_domain(domain_output, torch.cat((domain_labels, domain_labelt), 0))
            err_adversarial=0
            for j in range(10):
                temp1 = 'adversarial_output' + str(j)
                temp2 = 'lST' + str(j)
                err_adversarial += loss_adversarial(eval(temp1),eval(temp2))
            err_s_label = loss_class(class_output, class_labels)
            # 加MMD
            if epoch<30 or epoch>=120:
                err = err_domain + err_s_label + my_net.MMD(input_imgs, input_imgt) + my_net.deepcoral(input_imgs,input_imgt)
            elif (epoch>=30) and (epoch<60):
                err = err_domain + 0.01 * err_adversarial + err_s_label + my_net.MMD(input_imgs,input_imgt) + my_net.deepcoral(input_imgs, input_imgt)
            elif (epoch>=60) and (epoch<90):
                err = err_domain + err_s_label + my_net.MMD(input_imgs, input_imgt) + my_net.deepcoral(input_imgs,input_imgt)
            elif (epoch >= 90) and (epoch < 120):
                err = err_domain + 0.01 * err_adversarial + err_s_label + my_net.MMD(input_imgs, input_imgt) + my_net.deepcoral(input_imgs,input_imgt)
            else:
                err = err_domain +0.01*err_adversarial+ err_s_label + my_net.MMD(input_imgs,input_imgt) + my_net.deepcoral(input_imgs, input_imgt)
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
                  epoch, i, len_dataloader, err_s_label.data.cpu().numpy(), err_domain.data.cpu().numpy(), 0, err_adversarial.data.cpu().numpy(), dis))

        torch.save(my_net, '{0}/mnist_mnist_m_model_epoch_{1}.pth'.format(model_root, epoch))
        datas, labels = test(source_dataset_name, epoch)
        datat, labelt = test(target_dataset_name, epoch)
        if epoch % 10 == 0:
            data = np.vstack((datas, datat))
            label = np.vstack((labels, labelt))
            mytsne = TSNE(learning_rate=500).fit_transform(data)
            scio.savemat('FigureData3/data' + str(epoch) + '.mat', {'data': data})
            scio.savemat('FigureData3/label' + str(epoch) + '.mat', {'data': label})
            plt.figure(epoch + 2)
            plt.scatter(mytsne[:, 0], mytsne[:, 1], c=np.squeeze(label))
            plt.show()
            datas = None
            labels = None
            datat = None
            labelt = None
            data = None
            label = None

    #
    # _, Sfeature, Tfeature = my_net.Observe(Sinput_data=input_imgs, Tinput_data=input_imgt)
    # S_tsne = TSNE(learning_rate=500).fit_transform(Sfeature)
    # T_tsne = TSNE(learning_rate=500).fit_transform(Tfeature)
    print('done')