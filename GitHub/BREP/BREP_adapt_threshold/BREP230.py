import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import random
from torch import optim
from torch.autograd import Variable


#random.seed(0)
#np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

device = 'cuda:0'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('~/data1/', train=False, download=True, transform=transform)#10000
trainset = tv.datasets.CIFAR10(root='~/data1/', train=True, download=True, transform=transform)#50000

B = 1
Node = 10
Label = 10
batchsize = 50
mmm = 0
pr_c = 10
stepsize = 0.002
alpha = 0.002
penal = 0.001
regu = 0.01
beta = 0.2
L_Huber = 5 * pow(0.1, 3)
Iter = 200000
p = 1/20
kth = Node // 2
iteration = Iter
count_point = 100#画图要多少个点
count_iter = int(iteration/count_point)#每多少个epoch计算一次acc和H
criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数
Blist = [0, 1, 2, 3]


#测试集数据
data_test_temp = []
label_test_temp = []
for i in range(len(testset)):
    data_test_temp.append(testset[i][0])
    label_test_temp.append(testset[i][1])

data_test = []
label_test = []
for i in range(int(len(data_test_temp)/batchsize)):
    data_batchtemp = []
    label_batchtemp = []
    for bs in range(batchsize):
        data_batchtemp.append(data_test_temp[i*batchsize+bs].numpy().tolist())
        label_batchtemp.append(label_test_temp[i*batchsize+bs])
    data_test.append(torch.tensor(data_batchtemp, device=device, requires_grad=False))
    label_test.append(torch.tensor(np.array(label_batchtemp), device=device, requires_grad=False))


#数据在生成时已经做了混合打乱，程序主体中每一轮迭代相当于做无放回抽样
data_len = len(trainset)
data_temp = [[] for _ in range(Label)]#10
label_temp = [[] for _ in range(Label)]
for i in range(data_len):
    label_temp[trainset[i][1]].append(trainset[i][1])#5000
    data_temp[trainset[i][1]].append(trainset[i][0])

#——————iid数据集——————
#先从每个标签中随机抽样放进每个节点里
data_iid_temp = [[] for _ in range(Node)]
label_iid_temp = [[] for _ in range(Node)]
rand_index = list(range(len(data_temp[0])))#每个标签下的数据个数，默认5000
random.shuffle(rand_index)
for k in range(Node):
    for l in range(Label):
        for j in range(data_len // (Label*Node)):#250
            data_iid_temp[k].append(data_temp[l][rand_index[j + k*(data_len//(Label*Node))]].to(device))
            label_iid_temp[k].append(label_temp[l][rand_index[j + k*(data_len//(Label*Node))]])
#打乱，按batchsize分组打包
data_iid = [[] for _ in range(Node)]
label_iid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_iid_temp[0])))#依次取随机index的十个段
    random.shuffle(rand_index)
    for j in range(len(data_iid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_iid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_iid_temp[k][rand_index[j*batchsize+bs]])
        data_iid[k].append(torch.cat(data_batchtemp, dim=0))
        label_iid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))

#——————noniid数据集——————
#从每个标签中随机抽样放进每个节点里
data_noniid_temp = [[] for _ in range(Node)]
label_noniid_temp = [[] for _ in range(Node)]
#先放一半
for k in range(Node):
    for j in range(len(data_temp[0])//2):#2500
        data_noniid_temp[k].append(data_temp[k][j].to(device))
        label_noniid_temp[k].append(label_temp[k][j])
    #再放一半
data_noniid_mix = []
label_noniid_mix = []
for k in range(Node):
    for j in range(len(data_temp[0])//2):              #先混合
        data_noniid_mix.append(data_temp[k][len(data_temp)//2+j])
        label_noniid_mix.append(label_temp[k][len(data_temp)//2+j])
rand_index = list(range(len(data_noniid_mix)))      #打乱
random.shuffle(rand_index)
for k in range(10):
    for i in range(len(data_temp[0])//2):               #分配
        data_noniid_temp[k].append(data_noniid_mix[rand_index[(len(data_temp[0])//2)*k + i]].to(device))
        label_noniid_temp[k].append(label_noniid_mix[rand_index[(len(label_temp[0])//2)*k + i]])

#打乱，按batchsize分组打包
data_noniid = [[] for _ in range(Node)]
label_noniid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_noniid_temp[0])))#依次取随机index的十个段,5000
    random.shuffle(rand_index)
    for j in range(len(data_noniid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_noniid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_noniid_temp[k][rand_index[j*batchsize+bs]])
        data_noniid[k].append(torch.cat(data_batchtemp, dim=0))
        label_noniid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))


#Predefine
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64,5)
        self.fc1 = torch.nn.Linear(64*4*4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Accuracy(net, data_test, label_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(data_test)):
            images = data_test[i]
            labels = label_test[i]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # 第一个变量不重要，用-代替
            correct +=(predicted == labels).sum()      # 更新正确分类的图片的数量
            total += len(data_test[i])                  # 更新测试图片的数量
    return correct/total

def GetSign(wk,w0):
    w1 = wk-w0
    w1 = w1.sign()
    return w1

def GetSign_B(wk,w0):
    w2 = (-1)*wk-w0#Byzantine Attack
    w2 = w2.sign()
    return w2

def GetSub(wk,w0):
    w1 = wk-w0
    return w1

def GetSub_B(wk,w0):
    w2 = (-1)*wk-w0#Byzantine Attack
    return w2

def DistinguishHuber(msg, lh):
    ind_small = torch.where(msg.abs() - lh <= 0)
    ind_large = torch.where(msg.abs() - lh > 0)
    msg[ind_large] = (-1) * beta * lh * msg[ind_large].sign()
    msg[ind_small] = (-1) * beta * msg[ind_small]
    #H_diff = [penal*diff[i] if diff[i]>L_Huber else (1/L_Huber)*beta*diff[i] for i in range(len(diff))]
    # H_diff = H_diff.view(a_size)

def DistinguishHuber1(msg, k):
    for data in msg:
        data_abs = data.abs()
        lh = torch.topk(data_abs, k, dim=0)[0][-1]
        ind_small = torch.where(data_abs - lh <= 0)
        ind_large = torch.where(data_abs - lh > 0)
        data[ind_large] = (-1) * beta * lh[ind_large[1:]] * data[ind_large].sign()
        data[ind_small] = (-1) * beta * data[ind_small]


def DistinguishHuber_B(a, b, lh):
    a_size = a.size()
    a_line = a.view([-1])
    b_line = b.view([-1])
    diff = (-1)*a_line-b_line#Byzantine Attack
    H_diff = penal * diff
    H_diff[diff.abs() > lh] = diff[diff.abs() > lh].sign() * beta * lh
    H_diff[diff.abs() <= lh] = diff[diff.abs() <= lh] * beta
    #H_diff = [penal*diff[i] if diff[i]>L_Huber else (1/L_Huber)*beta*diff[i] for i in range(len(diff))]
    H_diff = H_diff.view(a_size)
    return H_diff

def ParaNorm(net0,netk):
    net0para_list = []
    netkparak_list = []
    paranorm_list = []
    for name, para in net0.named_parameters():
        net0para_list.append(para)
    for name, para in netk.named_parameters():
        netkparak_list.append(para)
    for pl in range(len(net0para_list)):
        paranorm_list.append(torch.norm(net0para_list[pl]-netkparak_list[pl]))
    return sum(paranorm_list)

def ParaNorm1(net):
    ret = 0.
    for para in net.parameters():
        ret += torch.norm(para.data).item()
    return ret

def ParaNorm2(net0, netk):
    distance = 0.
    for (para1, para2) in zip(net0.parameters(), netk.parameters()):
        d = torch.norm(para1.data-para2.data, p=float('inf')).item()
        distance = max(d, distance)
    return distance


# Blist = [1, 3]
for pr_b in Blist:
    B = pr_b+1
    # kth = Node - B

    #Initialization
    net0 = Net().to(device)
    net = []
    for k in range(Node):
        net.append(Net().to(device))
    for k in range(Node):
        net[k].load_state_dict(net0.state_dict())

    optimizer0 = optim.SGD(net0.parameters(), lr=alpha, momentum=mmm)
    optimizer = []
    for k in range(Node):
        optimizer.append(optim.SGD(net[k].parameters(), lr=alpha, momentum=mmm))


    #开始训练
    accuracy_list = []
    loss_list = []
    outputs = [[] for k in range(Node)]
    loss = [[] for k in range(Node)]
    outputs0 = [[] for k in range(Node)]
    loss0 = [[] for k in range(Node)]
    inputs = [[] for k in range(Node)]
    labels = [[] for k in range(Node)]
    dist = [[] for k in range(Node)]
    msg = []
    for para in net0.parameters():
        size = np.array(para.size()).tolist()
        size.insert(0, Node)
        msg.append(torch.zeros(size).to(device))

    diff = np.zeros(Node)
    lh = 0.
    comm = 0
    for epoch in range(iteration):
        running_loss = 0
        xi = np.random.binomial(1, p)
        rand_i = random.choice(list(range(len(data_iid[0]))))
        for k in range(Node):
            inputs[k] = data_noniid[k][rand_i]
            labels[k] = label_noniid[k][rand_i]

        optimizer0.zero_grad()
        for k in range(Node):
            optimizer[k].zero_grad()
        if xi == 0:
            for k in range(Node):
                outputs[k] = net[k](inputs[k])
                loss[k] = criterion(outputs[k], labels[k]) / (1 - p)
                loss[k].backward()
                optimizer[k].step()

            for para in net0.parameters():
                para.grad = regu / (1 - p) * para.data.clone().detach()
                # para.data.mul_(1 - regu * alpha / (1 - p))

        else:
            # lh = L_Huber
            # msg = [[] for k in range(Node)]
            # for k in range(B, Node):
            #     d = 0.
            #     for (para, para0) in zip(net[k].parameters(), net0.parameters()):
            #         msg[k].append(para.data.clone() - para0.data.clone())
            #         d = max(d, torch.norm(msg[k][-1], p=float('inf')).item())
            #     diff[k] = d
            # for k in range(B):
            #     d = 0.
            #     for (para, para0) in zip(net[k].parameters(), net0.parameters()):
            #         msg[k].append(10000*(para.data.clone() - para0.data.clone()))
            #         d = max(d, torch.norm(msg[k][-1], p=float('inf')).item())
            #     diff[k] = d
            for k in range(B, Node):
                for pi, (para, para0) in enumerate(zip(net[k].parameters(), net0.parameters())):
                    msg[pi][k] = (para.data - para0.data)
            for k in range(B):
                for pi, (para, para0) in enumerate(zip(net[k].parameters(), net0.parameters())):
                    msg[pi][k] = (10000 * (para.data - para0.data))
            DistinguishHuber1(msg, Node-kth+1)
            for k in range(Node):
                for (para0, para) in zip(net0.parameters(), net[k].parameters()):
                    para.grad = beta / p * (para.data.clone() - para0.data.clone()).detach()
                    # para.data.sub_(para.data - para0.data, alpha=alpha*beta/p)
                optimizer[k].step()

            for pi, para in enumerate(net0.parameters()):
                inc = msg[pi].sum(dim=0)
                para.grad = inc.data.clone() / p
                # para.data.sub_(inc.data, alpha=alpha/p)

            # lh = diff[np.argpartition(diff, kth)[kth]]
            # # lh = L_Huber
            # for k in range(Node):
            #     for data in msg[k]:
            #         DistinguishHuber(data, lh)
            #
            #     for (para0, para) in zip(net0.parameters(), net[k].parameters()):
            #         para.data.sub_(para.data - para0.data, alpha=alpha*beta/p)

            comm += 1

        optimizer0.step()
        for k in range(Node):
            netk_norm = ParaNorm1(net[k])
            if netk_norm > 1000:
                for para in net[k].parameters():
                    para.data.mul_(1000. / netk_norm)

        net0_norm = ParaNorm1(net0)
        if net0_norm > 1000:
            for para in net0.parameters():
                para.data.mul_(1000. / net0_norm)

        if epoch % count_iter == 0:
            accur = Accuracy(net0, data_test, label_test)
            accuracy_list.append([accur.cpu().numpy().tolist(), comm])
            print(comm, accuracy_list[-1][0], lh)
            print(epoch, Accuracy(net[9], data_test, label_test).cpu().numpy().tolist())
            for k in range(Node):
                outputs0[k] = net0(inputs[k])
                loss0[k] = criterion(outputs0[k], labels[k])
                dist[k].append(float(ParaNorm(net0, net[k])))
            running_loss += sum([loss0[k].item() for k in range(Node)])
            loss_list.append(running_loss/(len(labels[0])*Node))

    print("----------finished training---------")
    pr_c = 10 + pr_b
    pd.DataFrame(accuracy_list).to_csv('./result/Acc_BREPN231{}_P{}_M{}_N{}.csv'.format(pr_c, 1 / p, mmm, Node))


