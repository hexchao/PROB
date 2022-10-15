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
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('~/data1/',train=False,download=True,transform=transform)#10000
trainset = tv.datasets.CIFAR10(root='~/data1/',train = True,download=True,transform=transform)#50000

B = 1
Node = 10
Label = 10
batchsize = 50
mmm = 0.9
pr_c = 10
xi_s_list = [0]
xi_w_list = [0]
xi_s = xi_s_list[0]
xi_w = xi_w_list[0]
L_Huber = pow(0.1,6)
stepsize = 0.002
penal = 0.001
regu = 0.01
beta = 0.2
iteration = 30000
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
    net_para_list = []
    paranorm_list = []
    for name, para in net.named_parameters():
        net_para_list.append(para)
    for pl in range(len(net_para_list)):
        paranorm_list.append(torch.norm(net_para_list[pl]))
    return sum(paranorm_list)

def SignComp(k):
    sign_comp = 0
    sign_comp += len ((sign_fc1_w_0_N[k] - sign_fc1_w_old_N[k]).view([-1]).nonzero())
    sign_comp += len ((sign_fc2_w_0_N[k] - sign_fc2_w_old_N[k]).view([-1]).nonzero())
    sign_comp += len ((sign_fc3_w_0_N[k] - sign_fc3_w_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_conv1_w_0_N[k] - sign_conv1_w_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_conv2_w_0_N[k] - sign_conv2_w_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_fc1_b_0_N[k] - sign_fc1_b_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_fc2_b_0_N[k] - sign_fc2_b_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_fc3_b_0_N[k] - sign_fc3_b_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_conv1_b_0_N[k] - sign_conv1_b_old_N[k]).view([-1]).nonzero())
    sign_comp += len((sign_conv2_b_0_N[k] - sign_conv2_b_old_N[k]).view([-1]).nonzero())
    return sign_comp

def Para_len(net):
    lenth = 0
    for para in net.parameters():
        lenth += len(para.view([-1]))
    return lenth




for pr1 in range(len(xi_s_list)):
    xi_s = xi_s_list[pr1]
    xi_w = xi_w_list[pr1]

    for prb in range(4):
        B = prb+1

        #Initialization
        net0 = Net().to(device)
        net = []
        for k in range(Node):
            net.append(Net().to(device))
        for k in range(Node):
            net[k].load_state_dict(net0.state_dict())

        net0_old = Net().to(device)
        net0_old.load_state_dict(net0.state_dict())
        net_old = []
        for k in range(Node):
            net_old.append(Net().to(device))
        for k in range(Node):
            net_old[k].load_state_dict(net0.state_dict())

        optimizer0 = optim.SGD(net0.parameters(),lr = stepsize,momentum=mmm)
        optimizer = []
        for k in range(Node):
            optimizer.append(optim.SGD(net[k].parameters(),lr = stepsize,momentum=mmm))


        #开始训练
        para_len = Para_len(net0)
        HS_sum = 0
        HW_sum = 0
        HS_list = []
        HW_list = []
        HS_sum_list = []
        HS_sign = [1]
        HW_sum_list = []
        HW_sign = [1 for k in range(Node)]
        accuracy_list = []
        loss_list = []
        outputs = [[] for k in range(Node)]
        loss = [[] for k in range(Node)]
        outputs0 = [[] for k in range(Node)]
        loss0 = [[] for k in range(Node)]
        inputs = [[] for k in range(Node)]
        labels = [[] for k in range(Node)]
        dist = [[] for k in range(Node)]
        sign_fc1_w_0_N = [[] for k in range(Node)]
        sign_fc2_w_0_N = [[] for k in range(Node)]
        sign_fc3_w_0_N = [[] for k in range(Node)]
        sign_conv1_w_0_N = [[] for k in range(Node)]
        sign_conv2_w_0_N = [[] for k in range(Node)]
        sign_fc1_b_0_N = [[] for k in range(Node)]
        sign_fc2_b_0_N = [[] for k in range(Node)]
        sign_fc3_b_0_N = [[] for k in range(Node)]
        sign_conv1_b_0_N = [[] for k in range(Node)]
        sign_conv2_b_0_N = [[] for k in range(Node)]
        sign_fc1_w_0_B = [[] for k in range(Node)]
        sign_fc2_w_0_B = [[] for k in range(Node)]
        sign_fc3_w_0_B = [[] for k in range(Node)]
        sign_conv1_w_0_B = [[] for k in range(Node)]
        sign_conv2_w_0_B = [[] for k in range(Node)]
        sign_fc1_b_0_B = [[] for k in range(Node)]
        sign_fc2_b_0_B = [[] for k in range(Node)]
        sign_fc3_b_0_B = [[] for k in range(Node)]
        sign_conv1_b_0_B = [[] for k in range(Node)]
        sign_conv2_b_0_B = [[] for k in range(Node)]
        sign_fc1_w_old_N =[[] for k in range(Node)]
        sign_fc2_w_old_N =[[] for k in range(Node)]
        sign_fc3_w_old_N =[[] for k in range(Node)]
        sign_conv1_w_old_N =[[] for k in range(Node)]
        sign_conv2_w_old_N =[[] for k in range(Node)]
        sign_fc1_b_old_N =[[] for k in range(Node)]
        sign_fc2_b_old_N =[[] for k in range(Node)]
        sign_fc3_b_old_N =[[] for k in range(Node)]
        sign_conv1_b_old_N =[[] for k in range(Node)]
        sign_conv2_b_old_N =[[] for k in range(Node)]
        #完成赋值
        for k in range(Node):
            sign_fc1_w_0_N[k] = GetSign(net_old[k].fc1.weight.data.clone(),net0_old.fc1.weight.data.clone())
            sign_fc2_w_0_N[k] = GetSign(net_old[k].fc2.weight.data.clone(),net0_old.fc2.weight.data.clone())
            sign_fc3_w_0_N[k] = GetSign(net_old[k].fc3.weight.data.clone(),net0_old.fc3.weight.data.clone())
            sign_conv1_w_0_N[k] = GetSign(net_old[k].conv1.weight.data.clone(),net0_old.conv1.weight.data.clone())
            sign_conv2_w_0_N[k] = GetSign(net_old[k].conv2.weight.data.clone(),net0_old.conv2.weight.data.clone())
            sign_fc1_b_0_N[k] = GetSign(net_old[k].fc1.bias.data.clone(),net0_old.fc1.bias.data.clone())
            sign_fc2_b_0_N[k] = GetSign(net_old[k].fc2.bias.data.clone(),net0_old.fc2.bias.data.clone())
            sign_fc3_b_0_N[k] = GetSign(net_old[k].fc3.bias.data.clone(),net0_old.fc3.bias.data.clone())
            sign_conv1_b_0_N[k] = GetSign(net_old[k].conv1.bias.data.clone(),net0_old.conv1.bias.data.clone())
            sign_conv2_b_0_N[k] = GetSign(net_old[k].conv2.bias.data.clone(),net0_old.conv2.bias.data.clone())

        for epoch in range(iteration):
            running_loss = 0
            rand_i = random.choice(list(range(len(data_iid[0]))))
            for k in range(Node):
                inputs[k] = data_noniid[k][rand_i]
                labels[k] = label_noniid[k][rand_i]
            optimizer0.zero_grad()
            for k in range(Node):
                optimizer[k].zero_grad()
                outputs[k] = net[k](inputs[k])
                loss[k] = criterion(outputs[k], labels[k])
                loss[k].backward()

            for k in range(Node):
                sign_fc1_w_old_N[k] =sign_fc1_w_0_N[k].clone()
                sign_fc2_w_old_N[k] =sign_fc2_w_0_N[k].clone()
                sign_fc3_w_old_N[k] =sign_fc3_w_0_N[k].clone()
                sign_conv1_w_old_N[k] =sign_conv1_w_0_N[k].clone()
                sign_conv2_w_old_N[k] =sign_conv2_w_0_N[k].clone()
                sign_fc1_b_old_N[k] =sign_fc1_b_0_N[k].clone()
                sign_fc2_b_old_N[k] =sign_fc2_b_0_N[k].clone()
                sign_fc3_b_old_N[k] =sign_fc3_b_0_N[k].clone()
                sign_conv1_b_old_N[k] =sign_conv1_b_0_N[k].clone()
                sign_conv2_b_old_N[k] =sign_conv2_b_0_N[k].clone()
            for k in range(Node):
                sign_fc1_w_0_N[k] = beta * GetSub(net_old[k].fc1.weight.data.clone(),net0_old.fc1.weight.data.clone())
                sign_fc2_w_0_N[k] = beta * GetSub(net_old[k].fc2.weight.data.clone(),net0_old.fc2.weight.data.clone())
                sign_fc3_w_0_N[k] = beta * GetSub(net_old[k].fc3.weight.data.clone(),net0_old.fc3.weight.data.clone())
                sign_conv1_w_0_N[k] = beta * GetSub(net_old[k].conv1.weight.data.clone(),net0_old.conv1.weight.data.clone())
                sign_conv2_w_0_N[k] = beta * GetSub(net_old[k].conv2.weight.data.clone(),net0_old.conv2.weight.data.clone())
                sign_fc1_b_0_N[k] = beta * GetSub(net_old[k].fc1.bias.data.clone(),net0_old.fc1.bias.data.clone())
                sign_fc2_b_0_N[k] = beta * GetSub(net_old[k].fc2.bias.data.clone(),net0_old.fc2.bias.data.clone())
                sign_fc3_b_0_N[k] = beta * GetSub(net_old[k].fc3.bias.data.clone(),net0_old.fc3.bias.data.clone())
                sign_conv1_b_0_N[k] = beta * GetSub(net_old[k].conv1.bias.data.clone(),net0_old.conv1.bias.data.clone())
                sign_conv2_b_0_N[k] = beta * GetSub(net_old[k].conv2.bias.data.clone(),net0_old.conv2.bias.data.clone())
                sign_fc1_w_0_B[k] = beta * GetSub_B(net_old[k].fc1.weight.data.clone(),net0_old.fc1.weight.data.clone())
                sign_fc2_w_0_B[k] = beta * GetSub_B(net_old[k].fc2.weight.data.clone(),net0_old.fc2.weight.data.clone())
                sign_fc3_w_0_B[k] = beta * GetSub_B(net_old[k].fc3.weight.data.clone(),net0_old.fc3.weight.data.clone())
                sign_conv1_w_0_B[k] = beta * GetSub_B(net_old[k].conv1.weight.data.clone(),net0_old.conv1.weight.data.clone())
                sign_conv2_w_0_B[k] = beta * GetSub_B(net_old[k].conv2.weight.data.clone(),net0_old.conv2.weight.data.clone())
                sign_fc1_b_0_B[k] = beta * GetSub_B(net_old[k].fc1.bias.data.clone(),net0_old.fc1.bias.data.clone())
                sign_fc2_b_0_B[k] = beta * GetSub_B(net_old[k].fc2.bias.data.clone(),net0_old.fc2.bias.data.clone())
                sign_fc3_b_0_B[k] = beta * GetSub_B(net_old[k].fc3.bias.data.clone(),net0_old.fc3.bias.data.clone())
                sign_conv1_b_0_B[k] = beta * GetSub_B(net_old[k].conv1.bias.data.clone(),net0_old.conv1.bias.data.clone())
                sign_conv2_b_0_B[k] = beta * GetSub_B(net_old[k].conv2.bias.data.clone(),net0_old.conv2.bias.data.clone())

            net0.fc1.weight.grad = regu * net0.fc1.weight.data.clone() - torch.stack([sign_fc1_w_0_B[k] if k<B else sign_fc1_w_0_N[k] for k in range(10)],2).sum(axis=2)
            net0.fc2.weight.grad = regu * net0.fc2.weight.data.clone() - torch.stack([sign_fc2_w_0_B[k] if k<B else sign_fc2_w_0_N[k] for k in range(10)],2).sum(axis=2)
            net0.fc3.weight.grad = regu * net0.fc3.weight.data.clone() - torch.stack([sign_fc3_w_0_B[k] if k<B else sign_fc3_w_0_N[k] for k in range(10)],2).sum(axis=2)
            net0.conv1.weight.grad = regu * net0.conv1.weight.data.clone() - torch.stack([sign_conv1_w_0_B[k] if k<B else sign_conv1_w_0_N[k] for k in range(10)],4).sum(axis=4)
            net0.conv2.weight.grad = regu * net0.conv2.weight.data.clone() - torch.stack([sign_conv2_w_0_B[k] if k<B else sign_conv2_w_0_N[k] for k in range(10)],4).sum(axis=4)
            net0.fc1.bias.grad = regu * net0.fc1.bias.data.clone() - torch.stack([sign_fc1_b_0_B[k] if k<B else sign_fc1_b_0_N[k] for k in range(10)],1).sum(axis=1)
            net0.fc2.bias.grad = regu * net0.fc2.bias.data.clone() - torch.stack([sign_fc2_b_0_B[k] if k<B else sign_fc2_b_0_N[k] for k in range(10)],1).sum(axis=1)
            net0.fc3.bias.grad = regu * net0.fc3.bias.data.clone() - torch.stack([sign_fc3_b_0_B[k] if k<B else sign_fc3_b_0_N[k] for k in range(10)],1).sum(axis=1)
            net0.conv1.bias.grad = regu * net0.conv1.bias.data.clone() - torch.stack([sign_conv1_b_0_B[k] if k<B else sign_conv1_b_0_N[k] for k in range(10)],1).sum(axis=1)
            net0.conv2.bias.grad = regu * net0.conv2.bias.data.clone() - torch.stack([sign_conv2_b_0_B[k] if k<B else sign_conv2_b_0_N[k] for k in range(10)],1).sum(axis=1)
            optimizer0.step()
            HS_sign = [1 if float(ParaNorm(net0_old,net0)) >= xi_s else 0]
            if HS_sign[0]==1:#censoring
                net0_old.load_state_dict(net0.state_dict())
            HS_sum += sum(HS_sign)

            #worker中x0更新
            for k in range(Node):
                net[k].fc1.weight.grad += beta * (net[k].fc1.weight.data.clone() -net0_old.fc1.weight.data.clone())
                net[k].fc2.weight.grad += beta * (net[k].fc2.weight.data.clone() - net0_old.fc2.weight.data.clone())
                net[k].fc3.weight.grad += beta * (net[k].fc3.weight.data.clone() - net0_old.fc3.weight.data.clone())
                net[k].conv1.weight.grad += beta * (net[k].conv1.weight.data.clone() - net0_old.conv1.weight.data.clone())
                net[k].conv2.weight.grad += beta * (net[k].conv2.weight.data.clone() - net0_old.conv2.weight.data.clone())
                net[k].fc1.bias.grad += beta * (net[k].fc1.bias.data.clone() - net0_old.fc1.bias.data.clone())
                net[k].fc2.bias.grad += beta * (net[k].fc2.bias.data.clone() - net0_old.fc2.bias.data.clone())
                net[k].fc3.bias.grad += beta * (net[k].fc3.bias.data.clone() - net0_old.fc3.bias.data.clone())
                net[k].conv1.bias.grad += beta * (net[k].conv1.bias.data.clone() - net0_old.conv1.bias.data.clone())
                net[k].conv2.bias.grad += beta * (net[k].conv2.bias.data.clone() - net0_old.conv2.bias.data.clone())

            for k in range(Node):
                optimizer[k].step()
            if epoch==0:
                HW_sign = [1 for k in range(Node)]
            else:
                HW_sign = [1 if float(SignComp(k)) >= para_len*xi_w else 0 for k in range(Node)]
            for bb in range(B):
                HW_sign[bb] = 1
            for k in range(Node):
                if HW_sign[k]==1:#censoring
                    net_old[k].load_state_dict(net[k].state_dict())
            HW_sum += sum(HW_sign)-B


            net0_norm = ParaNorm1(net0)
            if net0_norm>1000:
                net0.fc1.weight.data = 1000*net0.fc1.weight.data.clone()/net0_norm
                net0.fc2.weight.data = 1000*net0.fc2.weight.data.clone()/net0_norm
                net0.fc3.weight.data = 1000*net0.fc3.weight.data.clone()/net0_norm
                net0.conv1.weight.data = 1000*net0.conv1.weight.data.clone()/net0_norm
                net0.conv2.weight.data = 1000*net0.conv2.weight.data.clone()/net0_norm
                net0.fc1.bias.data = 1000*net0.fc1.bias.data.clone()/net0_norm
                net0.fc2.bias.data = 1000*net0.fc2.bias.data.clone()/net0_norm
                net0.fc3.bias.data = 1000*net0.fc3.bias.data.clone()/net0_norm
                net0.conv1.bias.data = 1000*net0.conv1.bias.data.clone()/net0_norm
                net0.conv2.bias.data = 1000*net0.conv2.bias.data.clone()/net0_norm
            for k in range(Node):
                netk_norm = ParaNorm1(net[k])
                if netk_norm>1000:
                    net[k].fc1.weight.data = 1000*net[k].fc1.weight.data.clone()/netk_norm
                    net[k].fc2.weight.data = 1000*net[k].fc2.weight.data.clone()/netk_norm
                    net[k].fc3.weight.data = 1000*net[k].fc3.weight.data.clone()/netk_norm
                    net[k].conv1.weight.data = 1000*net[k].conv1.weight.data.clone()/netk_norm
                    net[k].conv2.weight.data = 1000*net[k].conv2.weight.data.clone()/netk_norm
                    net[k].fc1.bias.data = 1000*net[k].fc1.bias.data.clone()/netk_norm
                    net[k].fc2.bias.data = 1000*net[k].fc2.bias.data.clone()/netk_norm
                    net[k].fc3.bias.data = 1000*net[k].fc3.bias.data.clone()/netk_norm
                    net[k].conv1.bias.data = 1000*net[k].conv1.bias.data.clone()/netk_norm
                    net[k].conv2.bias.data = 1000*net[k].conv2.bias.data.clone()/netk_norm

            if epoch%count_iter==0:
                #xi = alpha*pow(rho,int(epoch/count_iter))
                HS_sum_list.append(HS_sum)
                HW_sum_list.append(HW_sum)
                accur = Accuracy(net0, data_test, label_test)
                accuracy_list.append(accur.cpu().numpy().tolist())
                print(epoch,accuracy_list[int(epoch/count_iter)])
                print(epoch,Accuracy(net[9], data_test, label_test).cpu().numpy().tolist())
                print(epoch, HS_sum, HW_sum)
                for k in range(Node):
                    outputs0[k] = net0(inputs[k])
                    loss0[k] = criterion(outputs0[k], labels[k])
                    dist[k].append(float(ParaNorm(net0,net[k])))
                running_loss += sum([loss0[k].item() for k in range(Node)])
                loss_list.append(running_loss/(len(labels[0])*Node))

        print("----------finished training---------")
        pd.DataFrame(accuracy_list).to_csv('/home/hexchao/Simulation_CIFAR10/Acc_LL210%d.csv'%pr_c)
        pd.DataFrame(HS_sum_list).to_csv('/home/hexchao/Simulation_CIFAR10/HSsum_LL210%d.csv'%pr_c)
        pd.DataFrame(HW_sum_list).to_csv('/home/hexchao/Simulation_CIFAR10/HWsum_LL210%d.csv'%pr_c)
        pd.DataFrame(loss_list).to_csv('/home/hexchao/Simulation_CIFAR10/Loss_LL210%d.csv'%pr_c)
        pd.DataFrame(dist).to_csv('/home/hexchao/Simulation_CIFAR10/Dist_LL210%d.csv'%pr_c)
        pr_c += 1





