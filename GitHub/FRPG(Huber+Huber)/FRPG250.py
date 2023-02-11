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

device = 'cuda:2'
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('~/data1/',train=False,download=True,transform=transform)#10000
trainset = tv.datasets.CIFAR10(root='~/data1/',train = True,download=True,transform=transform)#50000


B = 2
Node = 10
Label = 10
batchsize = 50
mmm = 0
pr_c = 10
stepsize = 0.002
alpha = 200
beta = 0.9
delta = 0.01
L_Huber = pow(0.1,3)
Lambda = 0.001
Iter = 100000
lc = 10
iteration = int(Iter/lc)
count_point = 100#画图要多少个点
count_iter = int(iteration/count_point)#每多少个epoch计算一次acc和H
criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数


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
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.conv2 = torch.nn.Conv2d(64,64,5)
        self.fc1 = torch.nn.Linear(64*4*4,384)
        self.fc2 = torch.nn.Linear(384,192)
        self.fc3 = torch.nn.Linear(192,10)
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


def PenalHuber(w0,wk):
    if torch.norm(w0-wk) > L_Huber:
        diff = (w0-wk).sign()
    else:
        diff = (1/L_Huber) * (w0-wk)
    return diff


for prB in range(4):
    B = prB+1

    #Initialization
    netU0 = Net().to(device)
    netW0 = Net().to(device)
    netV0 = Net().to(device)
    netUk = []
    netWk = []
    netVk = []
    netUBk = []
    netWBk = []
    netVBk = []
    for k in range(Node):
        netUk.append(Net().to(device))
        netWk.append(Net().to(device))
        netVk.append(Net().to(device))
        netUBk.append(Net().to(device))
        netWBk.append(Net().to(device))
        netVBk.append(Net().to(device))
    for k in range(Node):
        netUk[k].load_state_dict(netU0.state_dict())
        netWk[k].load_state_dict(netW0.state_dict())
        netVk[k].load_state_dict(netV0.state_dict())
        netUBk[k].load_state_dict(netW0.state_dict())
        netWBk[k].load_state_dict(netW0.state_dict())
        netVBk[k].load_state_dict(netW0.state_dict())


    optimizerU0 = optim.SGD(netU0.parameters(),lr = 1/(alpha),momentum=mmm)
    optimizerW0 = optim.SGD(netW0.parameters(),lr = stepsize,momentum=mmm)
    optimizerV0 = optim.SGD(netV0.parameters(),lr = 1/(delta+alpha*beta),momentum=mmm)
    optimizerUk = []
    optimizerWk = []
    optimizerVk = []
    optimizerUBk = []
    optimizerWBk = []
    optimizerVBk = []
    for k in range(Node):
        optimizerUk.append(optim.SGD(netUk[k].parameters(),lr = 1/(alpha),momentum=mmm))
        optimizerWk.append(optim.SGD(netWk[k].parameters(),lr = stepsize,momentum=mmm))
        optimizerVk.append(optim.SGD(netVk[k].parameters(),lr = 1/(delta+alpha*beta),momentum=mmm))
        optimizerUBk.append(optim.SGD(netUBk[k].parameters(),lr = stepsize,momentum=mmm))
        optimizerWBk.append(optim.SGD(netUBk[k].parameters(),lr = stepsize,momentum=mmm))
        optimizerVBk.append(optim.SGD(netUBk[k].parameters(),lr = stepsize,momentum=mmm))


    #开始训练
    accuracy_list = []
    loss_list = []
    inputs = [[] for k in range(Node)]
    labels = [[] for k in range(Node)]
    inputsB = [[] for k in range(Node)]
    labelsB = [[] for k in range(Node)]
    dist = [[] for k in range(Node)]
    outputsU0 = [[] for k in range(Node)]
    outputsW0 = [[] for k in range(Node)]
    outputsV0 = [[] for k in range(Node)]
    outputsUk = [[] for k in range(Node)]
    outputsWk = [[] for k in range(Node)]
    outputsVk = [[] for k in range(Node)]
    outputsUBk = [[] for k in range(Node)]
    outputsWBk = [[] for k in range(Node)]
    outputsVBk = [[] for k in range(Node)]
    lossU0 = [[] for k in range(Node)]
    lossW0 = [[] for k in range(Node)]
    lossV0 = [[] for k in range(Node)]
    lossUk = [[] for k in range(Node)]
    lossWk = [[] for k in range(Node)]
    lossVk = [[] for k in range(Node)]
    lossUBk = [[] for k in range(Node)]
    lossWBk = [[] for k in range(Node)]
    lossVBk = [[] for k in range(Node)]

    hb_fc1_w_0_N = [[] for k in range(Node)]
    hb_fc2_w_0_N = [[] for k in range(Node)]
    hb_fc3_w_0_N = [[] for k in range(Node)]
    hb_conv1_w_0_N = [[] for k in range(Node)]
    hb_conv2_w_0_N = [[] for k in range(Node)]
    hb_fc1_b_0_N = [[] for k in range(Node)]
    hb_fc2_b_0_N = [[] for k in range(Node)]
    hb_fc3_b_0_N = [[] for k in range(Node)]
    hb_conv1_b_0_N = [[] for k in range(Node)]
    hb_conv2_b_0_N = [[] for k in range(Node)]
    hb_fc1_w_0_B = [[] for k in range(Node)]
    hb_fc2_w_0_B = [[] for k in range(Node)]
    hb_fc3_w_0_B = [[] for k in range(Node)]
    hb_conv1_w_0_B = [[] for k in range(Node)]
    hb_conv2_w_0_B = [[] for k in range(Node)]
    hb_fc1_b_0_B = [[] for k in range(Node)]
    hb_fc2_b_0_B = [[] for k in range(Node)]
    hb_fc3_b_0_B = [[] for k in range(Node)]
    hb_conv1_b_0_B = [[] for k in range(Node)]
    hb_conv2_b_0_B = [[] for k in range(Node)]


    for pre in range(1):
        rand_i = random.choice(list(range(len(data_iid[0]))))
        for k in range(Node):
            inputs[k] = data_noniid[k][rand_i]
            labels[k] = label_noniid[k][rand_i]
        for k in range(Node):
            optimizerUk[k].zero_grad()
            optimizerWk[k].zero_grad()
            optimizerVk[k].zero_grad()
            optimizerUBk[k].zero_grad()
            optimizerWBk[k].zero_grad()
            optimizerVBk[k].zero_grad()
            outputsUk[k] = netUk[k](inputs[k])
            outputsWk[k] = netWk[k](inputs[k])
            outputsVk[k] = netVk[k](inputs[k])
            outputsUBk[k] = netWk[k](inputs[k])
            outputsWBk[k] = netWk[k](inputs[k])
            outputsVBk[k] = netWk[k](inputs[k])
            lossUk[k] = criterion(outputsUk[k], labels[k])
            lossWk[k] = criterion(outputsWk[k], labels[k])
            lossVk[k] = criterion(outputsVk[k], labels[k])
            lossUBk[k] = criterion(outputsUBk[k], labels[k])
            lossWBk[k] = criterion(outputsWBk[k], labels[k])
            lossVBk[k] = criterion(outputsVBk[k], labels[k])
            lossUk[k].backward()
            lossWk[k].backward()
            lossVk[k].backward()
            lossUBk[k].backward()
            lossWBk[k].backward()
            lossVBk[k].backward()
        for k in range(Node):
            optimizerUk[k].step()
            optimizerWk[k].step()
            optimizerVk[k].step()
            optimizerUBk[k].step()
            optimizerWBk[k].step()
            optimizerVBk[k].step()
        optimizerU0.zero_grad()
        optimizerW0.zero_grad()
        optimizerV0.zero_grad()
        outputsU0 = netU0(inputs[k])
        outputsW0 = netW0(inputs[k])
        outputsV0 = netV0(inputs[k])
        lossU0 = criterion(outputsU0, labels[k])
        lossW0 = criterion(outputsW0, labels[k])
        lossV0 = criterion(outputsV0, labels[k])
        lossU0.backward()
        lossW0.backward()
        lossV0.backward()
        optimizerU0.step()
        optimizerW0.step()
        optimizerV0.step()




    for epoch in range(iteration):
        running_loss = 0


        #先更新server
        for k in range(Node):
            hb_fc1_w_0_N[k] = Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWk[k].fc1.weight.data.clone())
            hb_fc2_w_0_N[k] = Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWk[k].fc2.weight.data.clone())
            hb_fc3_w_0_N[k] = Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWk[k].fc3.weight.data.clone())
            hb_conv1_w_0_N[k] = Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWk[k].conv1.weight.data.clone())
            hb_conv2_w_0_N[k] = Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWk[k].conv2.weight.data.clone())
            hb_fc1_b_0_N[k] = Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWk[k].fc1.bias.data.clone())
            hb_fc2_b_0_N[k] = Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWk[k].fc2.bias.data.clone())
            hb_fc3_b_0_N[k] = Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWk[k].fc3.bias.data.clone())
            hb_conv1_b_0_N[k] = Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWk[k].conv1.bias.data.clone())
            hb_conv2_b_0_N[k] = Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWk[k].conv2.bias.data.clone())
            hb_fc1_w_0_B[k] = Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWBk[k].fc1.weight.data.clone())
            hb_fc2_w_0_B[k] = Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWBk[k].fc2.weight.data.clone())
            hb_fc3_w_0_B[k] = Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWBk[k].fc3.weight.data.clone())
            hb_conv1_w_0_B[k] = Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWBk[k].conv1.weight.data.clone())
            hb_conv2_w_0_B[k] = Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWBk[k].conv2.weight.data.clone())
            hb_fc1_b_0_B[k] = Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWBk[k].fc1.bias.data.clone())
            hb_fc2_b_0_B[k] = Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWBk[k].fc2.bias.data.clone())
            hb_fc3_b_0_B[k] = Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWBk[k].fc3.bias.data.clone())
            hb_conv1_b_0_B[k] = Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWBk[k].conv1.bias.data.clone())
            hb_conv2_b_0_B[k] = Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWBk[k].conv2.bias.data.clone())

        netU0.fc1.weight.data = (1-beta)*netW0.fc1.weight.data.clone() + beta*netV0.fc1.weight.data.clone()
        netU0.fc2.weight.data = (1-beta)*netW0.fc2.weight.data.clone() + beta*netV0.fc2.weight.data.clone()
        netU0.fc3.weight.data = (1-beta)*netW0.fc3.weight.data.clone() + beta*netV0.fc3.weight.data.clone()
        netU0.conv1.weight.data = (1-beta)*netW0.conv1.weight.data.clone() + beta*netV0.conv1.weight.data.clone()
        netU0.conv2.weight.data = (1-beta)*netW0.conv2.weight.data.clone() + beta*netV0.conv2.weight.data.clone()
        netU0.fc1.bias.data = (1-beta)*netW0.fc1.bias.data.clone() + beta*netV0.fc1.bias.data.clone()
        netU0.fc2.bias.data = (1-beta)*netW0.fc2.bias.data.clone() + beta*netV0.fc2.bias.data.clone()
        netU0.fc3.bias.data = (1-beta)*netW0.fc3.bias.data.clone() + beta*netV0.fc3.bias.data.clone()
        netU0.conv1.bias.data = (1-beta)*netW0.conv1.bias.data.clone() + beta*netV0.conv1.bias.data.clone()
        netU0.conv2.bias.data = (1-beta)*netW0.conv2.bias.data.clone() + beta*netV0.conv2.bias.data.clone()

        netW0.fc1.weight.data = netU0.fc1.weight.data.clone()
        netW0.fc2.weight.data = netU0.fc2.weight.data.clone()
        netW0.fc3.weight.data = netU0.fc3.weight.data.clone()
        netW0.conv1.weight.data = netU0.conv1.weight.data.clone()
        netW0.conv2.weight.data = netU0.conv2.weight.data.clone()
        netW0.fc1.bias.data = netU0.fc1.bias.data.clone()
        netW0.fc2.bias.data = netU0.fc2.bias.data.clone()
        netW0.fc3.bias.data = netU0.fc3.bias.data.clone()
        netW0.conv1.bias.data = netU0.conv1.bias.data.clone()
        netW0.conv2.bias.data = netU0.conv2.bias.data.clone()
        netW0.fc1.weight.grad = delta * netU0.fc1.weight.data.clone()
        netW0.fc2.weight.grad = delta * netU0.fc2.weight.data.clone()
        netW0.fc3.weight.grad = delta * netU0.fc3.weight.data.clone()
        netW0.conv1.weight.grad = delta * netU0.conv1.weight.data.clone()
        netW0.conv2.weight.grad = delta * netU0.conv2.weight.data.clone()
        netW0.fc1.bias.grad = delta * netU0.fc1.bias.data.clone()
        netW0.fc2.bias.grad = delta * netU0.fc2.bias.data.clone()
        netW0.fc3.bias.grad = delta * netU0.fc3.bias.data.clone()
        netW0.conv1.bias.grad = delta * netU0.conv1.bias.data.clone()
        netW0.conv2.bias.grad = delta * netU0.conv2.bias.data.clone()
        optimizerW0.step()

        netV0.fc1.weight.grad = delta*(netV0.fc1.weight.data.clone()-netU0.fc1.weight.data.clone()) + delta*netU0.fc1.weight.data.clone() + torch.stack([hb_fc1_w_0_B[k] if k<B else hb_fc1_w_0_N[k] for k in range(10)],2).sum(axis=2)
        netV0.fc2.weight.grad = delta*(netV0.fc2.weight.data.clone()-netU0.fc2.weight.data.clone()) + delta*netU0.fc2.weight.data.clone() + torch.stack([hb_fc2_w_0_B[k] if k<B else hb_fc2_w_0_N[k] for k in range(10)],2).sum(axis=2)
        netV0.fc3.weight.grad = delta*(netV0.fc3.weight.data.clone()-netU0.fc3.weight.data.clone()) + delta*netU0.fc3.weight.data.clone() + torch.stack([hb_fc3_w_0_B[k] if k<B else hb_fc3_w_0_N[k] for k in range(10)],2).sum(axis=2)
        netV0.conv1.weight.grad = delta*(netV0.conv1.weight.data.clone()-netU0.conv1.weight.data.clone()) + delta*netU0.conv1.weight.data.clone() + torch.stack([hb_conv1_w_0_B[k] if k<B else hb_conv1_w_0_N[k] for k in range(10)],4).sum(axis=4)
        netV0.conv2.weight.grad = delta*(netV0.conv2.weight.data.clone()-netU0.conv2.weight.data.clone()) + delta*netU0.conv2.weight.data.clone() + torch.stack([hb_conv2_w_0_B[k] if k<B else hb_conv2_w_0_N[k] for k in range(10)],4).sum(axis=4)
        netV0.fc1.bias.grad = delta*(netV0.fc1.bias.data.clone()-netU0.fc1.bias.data.clone()) + delta*netU0.fc1.bias.data.clone() + torch.stack([hb_fc1_b_0_B[k] if k<B else hb_fc1_b_0_N[k] for k in range(10)],1).sum(axis=1)
        netV0.fc2.bias.grad = delta*(netV0.fc2.bias.data.clone()-netU0.fc2.bias.data.clone()) + delta*netU0.fc2.bias.data.clone() + torch.stack([hb_fc2_b_0_B[k] if k<B else hb_fc2_b_0_N[k] for k in range(10)],1).sum(axis=1)
        netV0.fc3.bias.grad = delta*(netV0.fc3.bias.data.clone()-netU0.fc3.bias.data.clone()) + delta*netU0.fc3.bias.data.clone() + torch.stack([hb_fc3_b_0_B[k] if k<B else hb_fc3_b_0_N[k] for k in range(10)],1).sum(axis=1)
        netV0.conv1.bias.grad = delta*(netV0.conv1.bias.data.clone()-netU0.conv1.bias.data.clone()) + delta*netU0.conv1.bias.data.clone() + torch.stack([hb_conv1_b_0_B[k] if k<B else hb_conv1_b_0_N[k] for k in range(10)],1).sum(axis=1)
        netV0.conv2.bias.grad = delta*(netV0.conv2.bias.data.clone()-netU0.conv2.bias.data.clone()) + delta*netU0.conv2.bias.data.clone() + torch.stack([hb_conv2_b_0_B[k] if k<B else hb_conv2_b_0_N[k] for k in range(10)],1).sum(axis=1)
        optimizerV0.step()

        optimizerU0.zero_grad()
        optimizerW0.zero_grad()
        optimizerV0.zero_grad()

        #再更新worker
        for local_epoch in range(lc):
            rand_i = random.choice(list(range(len(data_iid[0]))))
            for k in range(Node):
                inputs[k] = data_noniid[k][rand_i]
                labels[k] = label_noniid[k][rand_i]
            for k in range(Node):
                inputsB[k] = data_noniid[k][rand_i]
                labelsB[k] = 9-label_noniid[k][rand_i]
            for k in range(Node):
                optimizerUk[k].zero_grad()
                optimizerWk[k].zero_grad()
                optimizerVk[k].zero_grad()
                optimizerUBk[k].zero_grad()
                outputsUk[k] = netUk[k](inputs[k])
                outputsUBk[k] = netUBk[k](inputsB[k])
                #outputsWk[k] = netWk[k](inputs[k])
                #outputsVk[k] = netVk[k](inputs[k])
                lossUk[k] = criterion(outputsUk[k], labels[k])
                lossUBk[k] = criterion(outputsUBk[k], labelsB[k])
                #lossWk[k] = criterion(outputsWk[k], labels[k])
                #lossVk[k] = criterion(outputsVk[k], labels[k])
                lossUk[k].backward()
                lossUBk[k].backward()
                #lossWk[k].backward()
                #lossVk[k].backward()

            for k in range(Node):
                netUk[k].fc1.weight.data = (1-beta)*netWk[k].fc1.weight.data.clone() + beta*netVk[k].fc1.weight.data.clone()
                netUk[k].fc2.weight.data = (1-beta)*netWk[k].fc2.weight.data.clone() + beta*netVk[k].fc2.weight.data.clone()
                netUk[k].fc3.weight.data = (1-beta)*netWk[k].fc3.weight.data.clone() + beta*netVk[k].fc3.weight.data.clone()
                netUk[k].conv1.weight.data = (1-beta)*netWk[k].conv1.weight.data.clone() + beta*netVk[k].conv1.weight.data.clone()
                netUk[k].conv2.weight.data = (1-beta)*netWk[k].conv2.weight.data.clone() + beta*netVk[k].conv2.weight.data.clone()
                netUk[k].fc1.bias.data = (1-beta)*netWk[k].fc1.bias.data.clone() + beta*netVk[k].fc1.bias.data.clone()
                netUk[k].fc2.bias.data = (1-beta)*netWk[k].fc2.bias.data.clone() + beta*netVk[k].fc2.bias.data.clone()
                netUk[k].fc3.bias.data = (1-beta)*netWk[k].fc3.bias.data.clone() + beta*netVk[k].fc3.bias.data.clone()
                netUk[k].conv1.bias.data = (1-beta)*netWk[k].conv1.bias.data.clone() + beta*netVk[k].conv1.bias.data.clone()
                netUk[k].conv2.bias.data = (1-beta)*netWk[k].conv2.bias.data.clone() + beta*netVk[k].conv2.bias.data.clone()
            for k in range(Node):
                netUBk[k].fc1.weight.data = (1-beta)*netWBk[k].fc1.weight.data.clone() + beta*netVBk[k].fc1.weight.data.clone()
                netUBk[k].fc2.weight.data = (1-beta)*netWBk[k].fc2.weight.data.clone() + beta*netVBk[k].fc2.weight.data.clone()
                netUBk[k].fc3.weight.data = (1-beta)*netWBk[k].fc3.weight.data.clone() + beta*netVBk[k].fc3.weight.data.clone()
                netUBk[k].conv1.weight.data = (1-beta)*netWBk[k].conv1.weight.data.clone() + beta*netVBk[k].conv1.weight.data.clone()
                netUBk[k].conv2.weight.data = (1-beta)*netWBk[k].conv2.weight.data.clone() + beta*netVBk[k].conv2.weight.data.clone()
                netUBk[k].fc1.bias.data = (1-beta)*netWBk[k].fc1.bias.data.clone() + beta*netVBk[k].fc1.bias.data.clone()
                netUBk[k].fc2.bias.data = (1-beta)*netWBk[k].fc2.bias.data.clone() + beta*netVBk[k].fc2.bias.data.clone()
                netUBk[k].fc3.bias.data = (1-beta)*netWBk[k].fc3.bias.data.clone() + beta*netVBk[k].fc3.bias.data.clone()
                netUBk[k].conv1.bias.data = (1-beta)*netWBk[k].conv1.bias.data.clone() + beta*netVBk[k].conv1.bias.data.clone()
                netUBk[k].conv2.bias.data = (1-beta)*netWBk[k].conv2.bias.data.clone() + beta*netVBk[k].conv2.bias.data.clone()

            for k in range(Node):
                netWk[k].fc1.weight.grad = netUk[k].fc1.weight.grad.clone() - Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWk[k].fc1.weight.data.clone())
                netWk[k].fc2.weight.grad = netUk[k].fc2.weight.grad.clone() - Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWk[k].fc2.weight.data.clone())
                netWk[k].fc3.weight.grad = netUk[k].fc3.weight.grad.clone() - Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWk[k].fc3.weight.data.clone())
                netWk[k].conv1.weight.grad = netUk[k].conv1.weight.grad.clone() - Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWk[k].conv1.weight.data.clone())
                netWk[k].conv2.weight.grad = netUk[k].conv2.weight.grad.clone() - Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWk[k].conv2.weight.data.clone())
                netWk[k].fc1.bias.grad = netUk[k].fc1.bias.grad.clone() - Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWk[k].fc1.bias.data.clone())
                netWk[k].fc2.bias.grad = netUk[k].fc2.bias.grad.clone() - Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWk[k].fc2.bias.data.clone())
                netWk[k].fc3.bias.grad = netUk[k].fc3.bias.grad.clone() - Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWk[k].fc3.bias.data.clone())
                netWk[k].conv1.bias.grad = netUk[k].conv1.bias.grad.clone() - Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWk[k].conv1.bias.data.clone())
                netWk[k].conv2.bias.grad = netUk[k].conv2.bias.grad.clone() - Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWk[k].conv2.bias.data.clone())
            for k in range(Node):
                netWk[k].fc1.weight.data = netUk[k].fc1.weight.data.clone()
                netWk[k].fc2.weight.data = netUk[k].fc2.weight.data.clone()
                netWk[k].fc3.weight.data = netUk[k].fc3.weight.data.clone()
                netWk[k].conv1.weight.data = netUk[k].conv1.weight.data.clone()
                netWk[k].conv2.weight.data = netUk[k].conv2.weight.data.clone()
                netWk[k].fc1.bias.data = netUk[k].fc1.bias.data.clone()
                netWk[k].fc2.bias.data = netUk[k].fc2.bias.data.clone()
                netWk[k].fc3.bias.data = netUk[k].fc3.bias.data.clone()
                netWk[k].conv1.bias.data = netUk[k].conv1.bias.data.clone()
                netWk[k].conv2.bias.data = netUk[k].conv2.bias.data.clone()
            for k in range(Node):
                optimizerWk[k].step()
            for k in range(Node):
                netWBk[k].fc1.weight.grad = netUBk[k].fc1.weight.grad.clone() - Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWBk[k].fc1.weight.data.clone())
                netWBk[k].fc2.weight.grad = netUBk[k].fc2.weight.grad.clone() - Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWBk[k].fc2.weight.data.clone())
                netWBk[k].fc3.weight.grad = netUBk[k].fc3.weight.grad.clone() - Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWBk[k].fc3.weight.data.clone())
                netWBk[k].conv1.weight.grad = netUBk[k].conv1.weight.grad.clone() - Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWBk[k].conv1.weight.data.clone())
                netWBk[k].conv2.weight.grad = netUBk[k].conv2.weight.grad.clone() - Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWBk[k].conv2.weight.data.clone())
                netWBk[k].fc1.bias.grad = netUBk[k].fc1.bias.grad.clone() - Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWBk[k].fc1.bias.data.clone())
                netWBk[k].fc2.bias.grad = netUBk[k].fc2.bias.grad.clone() - Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWBk[k].fc2.bias.data.clone())
                netWBk[k].fc3.bias.grad = netUBk[k].fc3.bias.grad.clone() - Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWBk[k].fc3.bias.data.clone())
                netWBk[k].conv1.bias.grad = netUBk[k].conv1.bias.grad.clone() - Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWBk[k].conv1.bias.data.clone())
                netWBk[k].conv2.bias.grad = netUBk[k].conv2.bias.grad.clone() - Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWBk[k].conv2.bias.data.clone())
            for k in range(Node):
                netWBk[k].fc1.weight.data = netUBk[k].fc1.weight.data.clone()
                netWBk[k].fc2.weight.data = netUBk[k].fc2.weight.data.clone()
                netWBk[k].fc3.weight.data = netUBk[k].fc3.weight.data.clone()
                netWBk[k].conv1.weight.data = netUBk[k].conv1.weight.data.clone()
                netWBk[k].conv2.weight.data = netUBk[k].conv2.weight.data.clone()
                netWBk[k].fc1.bias.data = netUBk[k].fc1.bias.data.clone()
                netWBk[k].fc2.bias.data = netUBk[k].fc2.bias.data.clone()
                netWBk[k].fc3.bias.data = netUBk[k].fc3.bias.data.clone()
                netWBk[k].conv1.bias.data = netUBk[k].conv1.bias.data.clone()
                netWBk[k].conv2.bias.data = netUBk[k].conv2.bias.data.clone()
            for k in range(Node):
                optimizerWBk[k].step()

            for k in range(Node):
                netVk[k].fc1.weight.grad = delta*(netVk[k].fc1.weight.data.clone()-netUk[k].fc1.weight.data.clone()) + netUk[k].fc1.weight.grad.clone() - Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWk[k].fc1.weight.data.clone())
                netVk[k].fc2.weight.grad = delta*(netVk[k].fc2.weight.data.clone()-netUk[k].fc2.weight.data.clone()) + netUk[k].fc2.weight.grad.clone() - Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWk[k].fc2.weight.data.clone())
                netVk[k].fc3.weight.grad = delta*(netVk[k].fc3.weight.data.clone()-netUk[k].fc3.weight.data.clone()) + netUk[k].fc3.weight.grad.clone() - Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWk[k].fc3.weight.data.clone())
                netVk[k].conv1.weight.grad = delta*(netVk[k].conv1.weight.data.clone()-netUk[k].conv1.weight.data.clone()) + netUk[k].conv1.weight.grad.clone() - Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWk[k].conv1.weight.data.clone())
                netVk[k].conv2.weight.grad = delta*(netVk[k].conv2.weight.data.clone()-netUk[k].conv2.weight.data.clone()) + netUk[k].conv2.weight.grad.clone() - Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWk[k].conv2.weight.data.clone())
                netVk[k].fc1.bias.grad = delta*(netVk[k].fc1.bias.data.clone()-netUk[k].fc1.bias.data.clone()) + netUk[k].fc1.bias.grad.clone() - Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWk[k].fc1.bias.data.clone())
                netVk[k].fc2.bias.grad = delta*(netVk[k].fc2.bias.data.clone()-netUk[k].fc2.bias.data.clone()) + netUk[k].fc2.bias.grad.clone() - Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWk[k].fc2.bias.data.clone())
                netVk[k].fc3.bias.grad = delta*(netVk[k].fc3.bias.data.clone()-netUk[k].fc3.bias.data.clone()) + netUk[k].fc3.bias.grad.clone() - Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWk[k].fc3.bias.data.clone())
                netVk[k].conv1.bias.grad = delta*(netVk[k].conv1.bias.data.clone()-netUk[k].conv1.bias.data.clone()) + netUk[k].conv1.bias.grad.clone() - Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWk[k].conv1.bias.data.clone())
                netVk[k].conv2.bias.grad = delta*(netVk[k].conv2.bias.data.clone()-netUk[k].conv2.bias.data.clone()) + netUk[k].conv2.bias.grad.clone() - Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWk[k].conv2.bias.data.clone())
            for k in range(Node):
                optimizerVk[k].step()
            for k in range(Node):
                netVBk[k].fc1.weight.grad = delta*(netVBk[k].fc1.weight.data.clone()-netUBk[k].fc1.weight.data.clone()) + netUBk[k].fc1.weight.grad.clone() - Lambda * PenalHuber(netW0.fc1.weight.data.clone(),netWBk[k].fc1.weight.data.clone())
                netVBk[k].fc2.weight.grad = delta*(netVBk[k].fc2.weight.data.clone()-netUBk[k].fc2.weight.data.clone()) + netUBk[k].fc2.weight.grad.clone() - Lambda * PenalHuber(netW0.fc2.weight.data.clone(),netWBk[k].fc2.weight.data.clone())
                netVBk[k].fc3.weight.grad = delta*(netVBk[k].fc3.weight.data.clone()-netUBk[k].fc3.weight.data.clone()) + netUBk[k].fc3.weight.grad.clone() - Lambda * PenalHuber(netW0.fc3.weight.data.clone(),netWBk[k].fc3.weight.data.clone())
                netVBk[k].conv1.weight.grad = delta*(netVBk[k].conv1.weight.data.clone()-netUBk[k].conv1.weight.data.clone()) + netUBk[k].conv1.weight.grad.clone() - Lambda * PenalHuber(netW0.conv1.weight.data.clone(),netWBk[k].conv1.weight.data.clone())
                netVBk[k].conv2.weight.grad = delta*(netVBk[k].conv2.weight.data.clone()-netUBk[k].conv2.weight.data.clone()) + netUBk[k].conv2.weight.grad.clone() - Lambda * PenalHuber(netW0.conv2.weight.data.clone(),netWBk[k].conv2.weight.data.clone())
                netVBk[k].fc1.bias.grad = delta*(netVBk[k].fc1.bias.data.clone()-netUBk[k].fc1.bias.data.clone()) + netUBk[k].fc1.bias.grad.clone() - Lambda * PenalHuber(netW0.fc1.bias.data.clone(),netWBk[k].fc1.bias.data.clone())
                netVBk[k].fc2.bias.grad = delta*(netVBk[k].fc2.bias.data.clone()-netUBk[k].fc2.bias.data.clone()) + netUBk[k].fc2.bias.grad.clone() - Lambda * PenalHuber(netW0.fc2.bias.data.clone(),netWBk[k].fc2.bias.data.clone())
                netVBk[k].fc3.bias.grad = delta*(netVBk[k].fc3.bias.data.clone()-netUBk[k].fc3.bias.data.clone()) + netUBk[k].fc3.bias.grad.clone() - Lambda * PenalHuber(netW0.fc3.bias.data.clone(),netWBk[k].fc3.bias.data.clone())
                netVBk[k].conv1.bias.grad = delta*(netVBk[k].conv1.bias.data.clone()-netUBk[k].conv1.bias.data.clone()) + netUBk[k].conv1.bias.grad.clone() - Lambda * PenalHuber(netW0.conv1.bias.data.clone(),netWBk[k].conv1.bias.data.clone())
                netVBk[k].conv2.bias.grad = delta*(netVBk[k].conv2.bias.data.clone()-netUBk[k].conv2.bias.data.clone()) + netUBk[k].conv2.bias.grad.clone() - Lambda * PenalHuber(netW0.conv2.bias.data.clone(),netWBk[k].conv2.bias.data.clone())
            for k in range(Node):
                optimizerVBk[k].step()


        if epoch%count_iter==0:
            accur = Accuracy(netW0, data_test, label_test)
            accuracy_list.append(accur.cpu().numpy().tolist())
            H_count = 0
            for k in range(Node):
                if ParaNorm(netW0,netWk[k]) > L_Huber:
                    H_count +=1
            print(epoch,Accuracy(netU0, data_test, label_test).cpu().numpy().tolist())
            print(epoch,Accuracy(netW0, data_test, label_test).cpu().numpy().tolist())
            print(epoch,Accuracy(netV0, data_test, label_test).cpu().numpy().tolist())
            print(epoch,Accuracy(netUk[9], data_test, label_test).cpu().numpy().tolist())
            print(epoch,Accuracy(netWk[9], data_test, label_test).cpu().numpy().tolist())
            print(epoch,Accuracy(netVk[9], data_test, label_test).cpu().numpy().tolist())
            print(H_count,'======')

    print("----------finished training---------")
    pd.DataFrame(accuracy_list).to_csv('/home/hexchao/Simulation_CIFAR10/Acc_FRPGN251%d.csv'%pr_c)
    pr_c += 1



