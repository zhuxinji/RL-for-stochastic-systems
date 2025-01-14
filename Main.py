import scipy.io
import itertools
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import copy
import debugpy
debugpy.debug_this_thread()
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sympy import *
from modules import NetF, NetBF, Counter, Initialize_weight
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

def checkCLBF(modelCLF, modelBF):
    # find the largest invariant set in the stable region

    global invR1,invR2,u1b,u2b,uncertainty
 
    # uncertainty=1
    if uncertainty == 0.9:
        CAs =   0.675056342111348
        Ts =   351.209701753034
    elif uncertainty == 1:
        CAs = 0.5734
        Ts = 395.3268

    CF = 100e-3 * uncertainty
    CV = 0.1 * uncertainty
    Ck0 = 72e+9 * uncertainty
    CE = 8.314e+4 * uncertainty
    CR = 8.314 
    CT0 = 310 * uncertainty
    CDh = -4.78e+4 * uncertainty
    Ccp = 0.239 * uncertainty
    Crho = 1000 * uncertainty

    Q = torch.tensor([[9.35, 0.41], [0.41, 0.02]]).to(device)
    R = np.array([[1/invR1, 0], [0, 1/invR2]])

    CA0s = 1
    Qs = 0
    ua = np.array([[0.1], [0.1]])
    dx = np.array([[0.0], [0.0]])
 
    xx1_safe_ = np.linspace(-0.57, 0.57, 100)
    xx2_safe_ = np.linspace(-15, 15, 100)

    xx1_safe, xx2_safe = np.meshgrid(xx1_safe_, xx2_safe_)
    a = len(xx1_safe)
    V = np.zeros([a, a])

    predict_CLF = net_predict(torch.tensor(np.array([[0.1, -2]])), modelCLF)
    initial_C = predict_CLF[0]
    flagC = 0
    a0 = 0
    unstable_list = []
    stable_list = []
    unstable_array = np.array([[[0, 0]]])
    stable_array = np.array([[[0, 0]]])

    while flagC == 0:

        for i in range(len(xx1_safe)):
            for j in range(len(xx2_safe)):
                x1 = xx1_safe[i][j]
                x2 = xx2_safe[i][j]

                test_data_V = torch.tensor(np.array([[(x1)/1, (x2)/1]]))
                predict_CLF = net_predict(test_data_V, modelCLF)
                V_x1x2 = predict_CLF[0]
                V[i][j] = V_x1x2
                dVdX = predict_CLF[1].view(1, -1).to(device)

                if V_x1x2 < initial_C:

                    f1 = (CF / CV) * (- x1) - Ck0 * np.exp(-CE / (CR * (x2 + Ts))) * \
                        (x1 + CAs)+(CF / CV) * (CA0s-CAs)
                    f2 = (CF / CV) * (-x2) + (-CDh / (Crho * Ccp)) * Ck0 * \
                        np.exp(-CE / (CR * (x2 + Ts))) * \
                        (x1 + CAs) + CF*(CT0-Ts)/CV
                    f = torch.tensor([[f1], [f2]]).to(torch.float32).to(device)
                    g1 = CF / CV
                    g2 = 1 / (Crho * Ccp * CV)

                    LfV = dVdX@f
                    LgV1 = g1 * dVdX[0][0]
                    LgV2 = g2 * dVdX[0][1]

                    if (abs(x1) < 0.1) and (abs(x2) < 1):
                        deta_u = 0.001*(abs(x1)+abs(x2))
                    else:
                        deta_u = 0.0011
                    # QX = torch.bmm(torch.bmm(X.view(-1, 1, 2), Q), X)
                    xx = torch.tensor([[x1, x2]]).to(torch.float32).to(device)
                    xxt = xx.t()
                    QX = xx  @ Q @ xxt

                    kx1 = (LfV+math.sqrt(math.pow(LfV, 2) + invR1*math.pow(LgV1, 2)*(QX))) / \
                        (math.pow(LgV1, 2)+deta_u)
                    ub1 = -kx1*LgV1
                    kx2 = (LfV+math.sqrt(math.pow(LfV, 2) + invR2*math.pow(LgV2, 2)*(QX))) / \
                        (math.pow(LgV2, 2)+deta_u)
                    ub2 = -kx2*LgV2
                    #-- control input---#

                    if (abs(LgV1) > 1e-5):
                        ub1 = ub1[0][0]
                    else:
                        ub1 = 0

                    if (abs(LgV2) > 1e-5):
                        ub2 = ub2[0][0]
                    else:
                        ub2 = 0

                    ua_ = torch.tensor([[ub1], [ub2]]).to(device)

                    if (ua_[0][0] > u1b):
                        ua_[0][0] = u1b
                    elif (ua_[0][0] < -u1b):
                        ua_[0][0] = -u1b

                    if ua_[1][0] > u2b:
                        ua_[1][0] = u2b
                    elif ua_[1][0] < -u2b:
                        ua_[1][0] = -u2b

                    dx0 = f + torch.tensor([[g1*ua_[0][0]], [g2*ua_[1][0]]]).to(device)
                    dVdt = dVdX @ dx0
                    dVdt = dVdt[0][0]

                    if dVdt < 0:
                        flagC = 0
                        flag_CLBF = 1
                    else:
                        flagC = 1
                        flag_CLBF = 0

                if flagC == 1:
                    final_C = initial_C
                    break
            if flagC == 1:
                final_C = initial_C
                break
        if flagC == 1:
            final_C = initial_C
            flag_CLBF = 0
            break

        for k in range(int((15+15)/1)):
            x2 = -15+1*k
            test_data_V = torch.tensor(np.array([[(-0.57)/1, x2/1]]))
            predict_CLF = net_predict(test_data_V, modelCLF)
            V_x1x2 = predict_CLF[0]
            if V_x1x2 < initial_C:
                final_C = initial_C
                a0 = 1
                break
        if a0 == 1:
            break

        initial_C = initial_C+initial_C/25

    if flag_CLBF == 1:
        # check whether the unstable region D is within the invariant set
        xx1_D_ = np.linspace(-0.25, -0.19, 50)
        xx2_D_ = np.linspace(2, 7, 50)
        xx1_D, xx2_D = np.meshgrid(xx1_D_, xx2_D_)
        flag_CLBF0 = 1
        for i in range(len(xx1_D)):
            for j in range(len(xx2_D)):
                x1 = xx1_D[i][j]
                x2 = xx2_D[i][j]

                test_data_V = torch.tensor(np.array([[(x1)/1, (x2)/1]])).to(device)
                predict_CLF = net_predict(test_data_V, modelCLF)
                V_x1x2 = predict_CLF[0]

                if V_x1x2 >= final_C:
                    test_data_BF = torch.tensor(np.array([[(x1)/1, (x2)/1]])).to(device)

                    predict_BF0 = net_predict_BF(test_data_BF, modelBF)
                    BF0 = predict_BF0[0]
                    if BF0 > 0:
                        flag_CLBF0 = 0
                        flag_CLBF = 0
                        unstable_data = np.array([[x1, x2]])
                        unstable_list.append(unstable_data)
                        unstable_array = np.array(unstable_list)
                    else:
                        flag_CLBF0 = 1
                        stable_data = np.array([[x1, x2]])
                        stable_list.append(stable_data)
                        stable_array = np.array(stable_list)

                if flag_CLBF0 == 0:
                    flag_CLBF = 0
                    break
            if flag_CLBF0 == 0:
                flag_CLBF = 0
                break

    return flag_CLBF, final_C


def systemGenerateData(modelF, x1_0, x2_0, Flag_train, t_final, FlagCLF0):

    global kdVdt, ksumV, knonzero, allnum
    global invR1,invR2,u1b,u2b,uncertainty, sigma1, sigma2
    
    allnum = allnum+1

    Qa = 9.35
    Qb = 0.41
    Qc = 0.41
    Qd = 0.02
    Q = np.array([[Qa, Qb], [Qc, Qd]])
    
    # invR1 = 4
    # invR2 = 100

    R = np.array([[1/invR1, 0], [0, 1/invR2]])

    lambda0 = 5000
    alpha0 = 0.1

    # uncertainty=0.9

    if uncertainty == 0.9:
        CAs =   0.675056342111348 #0.999355860850161 # uncertainty=0.9
        Ts =   351.209701753034 #279.143142033298 # uncertainty=0.9
    elif uncertainty == 1:
        CAs = 0.5734
        Ts = 395.3268

    CF = 100e-3 * uncertainty
    CV = 0.1 * uncertainty
    Ck0 = 72e+9 * uncertainty
    CE = 8.314e+4 * uncertainty
    CR = 8.314 
    CT0 = 310 * uncertainty
    CDh = -4.78e+4 * uncertainty
    Ccp = 0.239 * uncertainty
    Crho = 1000 * uncertainty

    CA0s = 1
    Qs = 0

    ua = np.array([[0.1], [0.1]])
    dx = np.array([[0.0], [0.0]])

    t_step = 0.001
    t_final = t_final
    hold_step = 100
    
    sumV = 0

    x1, x2 = x1_0, x2_0

    x1_list = list()  # evolution of state over time
    x2_list = list()  # evolution of state time
    u1_list = list()
    u2_list = list()
    hat_V_list = list()
    LV_list = list()
    dw1_list = list()
    dw2_list = list()

    t_list = list()
    save_data = []
    NNoutput_list10 = []
    NNoutput_list20 = []
    dVdX1_list = []
    dVdX2_list = []

    lastV = 0
    dVdt = 0
    dw1 = 0
    dw2 = 0
    
    for i in range(int(t_final / t_step)):

        f1 = (CF / CV) * (- x1) - Ck0 * np.exp(-CE / (CR * (x2 + Ts))) * \
            (x1 + CAs)+(CF / CV) * (CA0s-CAs)
        f2 = (CF / CV) * (-x2) + (-CDh / (Crho * Ccp)) * Ck0 * \
            np.exp(-CE / (CR * (x2 + Ts))) * (x1 + CAs) + CF*(CT0-Ts)/CV

        g1 = CF / CV
        g2 = 1 / (Crho * Ccp * CV)

        f = np.array([[f1], [f2]])
        g = np.array([[g1, 0], [0, g2]])

        dt = 0.001
        dw = np.array([[
        np.random.normal(scale=np.sqrt(dt))],
        [np.random.normal(scale=np.sqrt(dt))]
        ])  # Independent Gaussian noise increments for each dimension
        hw = torch.tensor([[[sigma1*x1], [sigma2*x2]]],  dtype=torch.float).to(device)

        dBFdX = np.array([[0], [0]])
        BF = 0
            
        # ------------------------------------------
        # NN prediction
        # -------------------------------------------
        testdata = torch.tensor(np.array([[(x1)/1, (x2)/1]])).to(device)
        predict = net_predict(testdata, modelF)

        NNV = predict[0]
        dVdX = predict[1].view(-1, 1)
        hat_VV = NNV[0][0]
        h = predict[-1].to(device)

        LfV = (dVdX[0][0] * f1 + dVdX[1][0] * f2).to(device)
        trace = torch.bmm(hw.view(-1, 1, 2), torch.bmm(h, hw))
        LfV += 0.5*trace[0][0][0]
        
        LgV1 = g1 * dVdX[0][0]
        LgV2 = g2 * dVdX[1][0]

        QX = np.array([[x1, x2]])@Q@np.array([[x1], [x2]])

        if (abs(x1) < 0.1) and (abs(x2) < 1):
            deta_u = 0.001*(abs(x1)+abs(x2))
        else:
            deta_u = 0.001

        if i%hold_step == 0:
            #-- control input---#

            if (abs(LgV1) > 1e-5):
                kx1 = (LfV+math.sqrt(math.pow(LfV, 2)+invR1*math.pow(LgV1, 2)*(QX+BF*0))) / \
                    (math.pow(LgV1, 2)+deta_u)
                ub1 = -kx1*LgV1
                ub1 = ub1.cpu()
            else:
                ub1 = 0
    
            if (abs(LgV2) > 1e-5):
                kx2 = (LfV+math.sqrt(math.pow(LfV, 2)+invR2*math.pow(LgV2, 2)*(QX+BF*0))) / \
                    (math.pow(LgV2, 2))
                ub2 = -kx2*LgV2
                ub2 = ub2.cpu()
            else:
                ub2 = 0
            
            ua_ = np.array([[ub1], [ub2]])
    
            ua[0] = ua_[0][0]
            ua[1] = ua_[1][0]
    
            if (ua_[0][0] > u1b):
                ua_[0][0] = u1b
            elif (ua_[0][0] < -u1b):
                ua_[0][0] = -u1b
    
            if ua_[1][0] > u2b:
                ua_[1][0] = u2b
            elif ua_[1][0] < -u2b:
                ua_[1][0] = -u2b
            ua = ua_

        hw = np.array([[sigma1*x1], [sigma2*x2]])
        dx = f+g @ ua + hw*dw

        dVdt = np.transpose(dVdX)@dx
        dVdt = dVdt[0][0]

        U = np.transpose(ua) @ R @ ua

        # ---------------------------------
        # NN training data
        # ---------------------------------

        save_data0 = np.array([[(x1)/1, (x2)/1, dx[0][0], dx[1][0], U[0][0]]])
        save_data.append(save_data0)
        traindata = np.array(save_data)

        # ---------------------------------
        # check
        # ---------------------------------
        sumV = sumV  + np.array([[x1, x2]]) @ Q @ np.array([[x1], [x2]]) + U

        if Flag_train == 1:
            if sumV > 500:  # 1e16:f
                FlagCLF = 0
                ksumV = ksumV+1

        if Flag_train == 1:
            FlagCLF = 1
            
        else:
            if FlagCLF0 == 0:
                if dVdt <= 0:
                    FlagCLF = i
                else:
                    FlagCLF = i
                    print(f"i={i};dVdt > 0")
                    FlagCLF0 = 1

        NNoutput = predict[2]
        NNoutput_list10.append(NNoutput[0][0])
        NNoutput_list1 = np.array(NNoutput_list10)
        NNoutput_list20.append(NNoutput[0][1])
        NNoutput_list2 = np.array(NNoutput_list20)

        if i % 100 == 0:
            print(f"---i= {i} ---")
            print(f'Predition: u=[{ua[0][0]:.4f},{ua[1][0]:.4f}]', f'hat_V={NNV[0][0]:.5f}', f'dVdt={dVdt:.4f}',
                  f'x=[{x1:.2f},{x2:.2f}]', f'dVdX=[{dVdX[0][0]:.4f},{dVdX[1][0]:.4f}]', f'f=[{f1:.4f},{f2:.4f}]', f'g=[{g1:.4f},{g2:.4f}]')
            print(f"sumV={sumV[0][0]}")

        # update
        dx = f+g @ ua + hw*dw
        x1 = x1+dx[0][0]*t_step
        x2 = x2+dx[1][0]*t_step

        dw1 += dw[0]*t_step
        dw2 += dw[1]*t_step
        # save data
        # if i % 10 == 0:
        x1_list.append(x1)
        x2_list.append(x2)
        u1_list.append(ua[0][0])
        u2_list.append(ua[1][0])
        dw1_list.append(dw1[0])
        dw2_list.append(dw2[0])
        hat_V_list.append(hat_VV)
        LV_list.append(dVdt)
        dVdX1_list.append(dVdX[0][0])
        dVdX2_list.append(dVdX[1][0])
        t_list.append(i*t_step)

    return modelF, sumV, traindata, x1_list, x2_list, u1_list, u2_list, hat_V_list, t_list, FlagCLF, LV_list, dw1_list, dw2_list


# ----------------------------------
# prediction
# ----------------------------------
def net_predict(testdata, modelF):

    global allnum

    testdata = testdata.to(torch.float32).to(device)
    X = testdata[:, 0:2].to(device)
    X.requires_grad = True

    V_hat_ = modelF(X[:, 0:1],  X[:, 1:2]/12)
    V_hat = V_hat_ @  V_hat_.view(-1, 1) + \
        X[:, 0:1]*X[:, 0:1]*1 + X[:, 0:1] * \
        X[:, 1:2]*0 + X[:, 1:2]*X[:, 1:2]*0  # + \

    grad_VX = torch.autograd.grad(outputs=V_hat, inputs=X, grad_outputs=torch.ones_like(V_hat), create_graph=True)
    dVdX = grad_VX[0].view(-1, 1, 2)
    h = torch.zeros(len(X), 2, 2).to(device)
    for k in range(len(X)):
        for i in range(2):
            for j in range(2):
                h[k, i, j] = torch.autograd.grad(dVdX[k][0][i], X, retain_graph=True, allow_unused=True)[0][k][j]

    return np.array(V_hat.cpu().detach()), dVdX.cpu().detach(), np.array(V_hat_.cpu().detach()), h.cpu().detach()


def net_predict_BF(testdata, modelBF):

    global allnum

    testdata = testdata.to(torch.float32).to(device)
    X = testdata[:, 0:2]
    X.requires_grad = True

    V_hat = modelBF(X[:, 0:1],  X[:, 1:2])

    V_hat = (V_hat)*10000
    grad_VX = torch.autograd.grad(
        outputs=V_hat, inputs=X, grad_outputs=torch.ones_like(V_hat), create_graph=True)
    dVdX = grad_VX[0].view(-1, 1, 2)*1

    if V_hat < 0:
        V_hat = torch.Tensor(np.array([[0]]))
        dVdX = torch.Tensor(np.array([[0], [0]]))

    else:
        V_hat = V_hat
        dVdX = dVdX.detach()

    return np.array(V_hat.detach()), dVdX.detach()
# ----------------------------------
# train
# ----------------------------------


def train_online(dataset,  learning_rate, modelF):

    global uncertainty, sigma1, sigma2

    if uncertainty == 0.9:
        CAs =   0.675056342111348 #0.999355860850161 # uncertainty=0.9
        Ts =   351.209701753034 #279.143142033298 # uncertainty=0.9
    elif uncertainty == 1:
        CAs = 0.5734
        Ts = 395.3268

    batch_size = 16
    epochs = 90

    loss1 = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(itertools.chain(modelF.parameters()), lr=learning_rate)

    min_valid_loss = np.inf

    ecArr = dataset[:, 0, 0]
    ec = np.zeros_like(ecArr)
    loss_dict = []
    train_loss_list = []

    lr_list = []
    lr0 = np.array([[0.0, 0.0, 0.0]])

    nn0_list = []
    nn1_list = []
    nn2_list = []
    nn3_list = []

    inputdata = dataset
    X_train, X_val, y_train, y_val = train_test_split(
        inputdata, ec, test_size=0.1, random_state=123)

    dataloader_train = DataLoader(
        X_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(X_val, batch_size=batch_size, shuffle=True)

    for t in range(epochs):

        size = len(dataloader_train.dataset)
        train_loss_recorder = Counter()
        val_loss_recorder = Counter()

        modelF.train()  # prep model for training
        for step, (traindata) in enumerate(dataloader_train):

            traindata = traindata.to(torch.float32).to(device)

            X = traindata[:, 0, 0:2]
            X.requires_grad = True
            X_size = len(X)
            dX = traindata[:, 0, 2:4]
            U = traindata[:, 0, 4:5]
            U = U.to(torch.float32)

            X1 = X[:, 0:1]
            X2 = X[:, 1:2]

            V_ = modelF(X1,  X2/12)
            V_hat = V_.view(-1, 1, 2)
            V_hat = torch.bmm(V_hat, V_.view(-1, 2, 1))

            grad_VX = torch.autograd.grad(outputs=V_hat, inputs=X, grad_outputs=torch.ones_like(V_hat), create_graph=True)
            dVdX = grad_VX[0].view(-1, 1, 2)
            h = torch.zeros(X_size, 2, 2).to(device)
            for k in range(X_size):
                for i in range(2):
                    for j in range(2):
                        h[k, i, j] = torch.autograd.grad(dVdX[k][0][i], X, retain_graph=True, allow_unused=True)[0][k][j]

            Q = np.array([[9.35, 0.41], [0.41, 0.02]])
            Q = torch.tensor(Q).view(-1, 2, 2)
            Q = Q.repeat(X_size, 1, 1)
            Q = Q.to(torch.float32).to(device)

            # X = X.view(-1, 2, 1)
            dX = dX.view(-1, 2, 1)
            U = U.view(-1, 1, 1)

            x1 = X1.view(-1, 1, 1)*1
            x2 = X2.view(-1, 1, 1)*1

            X = torch.cat([x1, x2], 1).view(-1, 2, 1)

            #-- Define barrier function---#
            BF_FD = (x1+0.22)*(x1+0.22)+(x2-4.6)*(x2-4.6)/10000
            lambda0 = 5000  # dfgd
            alpha0 = 0.1

            BF = torch.zeros(len(BF_FD), 1, 1)
            dBFdX = torch.zeros(len(BF_FD), 2, 1).to(device)
            BF[i][0][0] = 0
            dBFdX[i] = torch.tensor([[0], [0]])

            nn0 = torch.bmm(torch.bmm(X.view(-1, 1, 2), Q), X)
            nn1 = torch.bmm(dVdX, dX)
            nn2 = BF
            nn3 = U
            nn00 = np.array(nn0_list.append(nn0))
            nn10 = np.array(nn1_list.append(nn1))
            nn20 = np.array(nn2_list.append(nn2))
            nn30 = np.array(nn3_list.append(nn3))

            CAs = torch.tensor(CAs)
            Ts = torch.tensor(Ts)

            hw = torch.cat((sigma1*x1 , sigma2*x2 ), dim=2)

            eck = 1
            ec_hat = torch.bmm(torch.bmm(X.view(-1, 1, 2), Q),X)/eck + U/eck + torch.bmm(dVdX, dX) +  0*torch.bmm(dBFdX.view(-1, 1, 2), dX)/eck
            trace = torch.bmm(hw, torch.bmm(h, hw.view(-1, 2, 1)))
            ec_hat += 0.5*trace
            ec = torch.zeros(X_size, 1).to(device)
            ec = ec.view(-1, 1, 1)

            loss = torch.log(loss1(ec_hat, ec))
            # loss = loss1(ec_hat, ec)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss_recorder.update(loss)

        pi = 0
        for param_group in optimizer.param_groups:
            lr0[0][pi] = param_group['lr']
            pi += 1
        lr_list.append(lr0)
        # scheduler.step()

        train_loss_list.append(train_loss_recorder.avg.cpu().detach().numpy())

        modelF.eval()  # prep model for evaluation
        for step, (valdata) in enumerate(dataloader_val):

            valdata = valdata.to(torch.float32).to(device)

            X = valdata[:, 0, 0:2]
            X.requires_grad = True
            X_size = len(X)
            dX = valdata[:, 0, 2:4]
            U = valdata[:, 0, 4:5]

            X1 = X[:, 0:1]
            X2 = X[:, 1:2]

            V_ = modelF(X1,  X2/12)
            V_hat = V_.view(-1, 1, 2)
            V_hat = torch.bmm(V_hat, V_.view(-1, 2, 1))

            grad_VX = torch.autograd.grad(outputs=V_hat, inputs=X, grad_outputs=torch.ones_like(V_hat), create_graph=True)
            dVdX = grad_VX[0].view(-1, 1, 2)
            h = torch.zeros(X_size, 2, 2).to(device)
            for k in range(X_size):
                for i in range(2):
                    for j in range(2):
                        h[k, i, j] = torch.autograd.grad(dVdX[k][0][i], X, retain_graph=True, allow_unused=True)[0][k][j]
            
            Q = np.array([[9.35, 0.41], [0.41, 0.02]])
            Q = torch.tensor(Q).view(-1, 2, 2)
            Q = Q.repeat(X_size, 1, 1)
            Q = Q.to(torch.float32).to(device)

            # X = X.view(-1, 2, 1)
            dX = dX.view(-1, 2, 1)
            U = U.view(-1, 1, 1)

            x1 = X1.view(-1, 1, 1)*1
            x2 = X2.view(-1, 1, 1)*1
            X = torch.cat([x1, x2], 1).view(-1, 2, 1)
            
            #-- Define barrier function---#
            BF_FD = (x1+0.22)*(x1+0.22)+(x2-4.6)*(x2-4.6)/10000
            lambda0 = 5000
            alpha0 = 0.1

            BF = torch.zeros(len(BF_FD), 1, 1)
            dBFdX = torch.zeros(len(BF_FD), 2, 1).to(device)
            BF[i][0][0] = 0
            dBFdX[i] = torch.tensor([[0], [0]])
                    
            hw = torch.cat((sigma1*x1 , sigma2*x2 ), dim=2)
            
            eck = 1
            ec_hat = torch.bmm(torch.bmm(X.view(-1, 1, 2), Q), X)/eck + U/eck + torch.bmm(dVdX, dX) + torch.bmm(dBFdX.view(-1, 1, 2), dX)/eck
            trace = torch.bmm(hw, torch.bmm(h, hw.view(-1, 2, 1)))
            ec_hat += 0.5*trace
            ec = torch.zeros(X_size, 1)
            ec = ec.view(-1, 1, 1).to(device)

            val_loss = torch.log(loss1(ec_hat, ec))
            val_loss_recorder.update(val_loss)

        if min_valid_loss > val_loss_recorder.avg:
            print(f"Epoch {t+1}")
            print(
                f"avg_train_loss: {train_loss_recorder.avg:.7f}\n avg_val_loss: {val_loss_recorder.avg:.7f} ")

            print(
                f'Validation Loss Decreased({min_valid_loss:.7f}--->{val_loss_recorder.avg:.7f}) \t Saving The Model')
            print(f"net[0]={lr0[0][0]},net[2]={lr0[0][1]},net[4]={lr0[0][2]}")
            min_valid_loss = val_loss_recorder.avg
            loss_dict.append(min_valid_loss.cpu().detach().numpy())
            best_NN_Model = copy.deepcopy(modelF)
            print("-------------------------------")

    print("Train Done!")
    return best_NN_Model, loss_dict, val_loss_recorder, min_valid_loss, train_loss_list


if __name__ == '__main__':

    invR1 = 1
    invR2 = 1
    uncertainty=1
    u1b = 2  # 2.2  # 1  # 15
    u2b = 0.167  # 4.67  # 0.0167  # 25
    sigma1, sigma2 = 0.015, 0.25
    x1_0 = 0.42 #-0.3 #0.2 #-0.2 #0.42  # safe
    x2_0 = -8 #8 #-5 #5 #-8

    save_num = 0
    learning_rate0 = 3*1e-4
    learning_rate = learning_rate0
    loss_list = []

    FlagCLF = 0
    FlagCLF0 = 0
    kdVdt = 0
    ksumV = 0
    knonzero = 0
    allnum = 0
    flag_CLBF00 = 1

    Flag_train = 1  # Flag_train = 1 denotes training

    if Flag_train == 1:
        t_final = 5
    else:
        t_final = 5

    modelF = NetF().to(device)
    modelBF = NetBF().to(device)
    modelBF.load_state_dict(torch.load("saved_modelBF00_test35.pth"))

    if Flag_train == 1:
        # Initialize a initial modelF CLF
        # while FlagCLF == 0:
        modelF = Initialize_weight(modelF).to(device)
        modelF.load_state_dict(torch.load("saved_model_initial30.pth"))
        # modelF.load_state_dict(torch.load("saved_model_pre.pth"))
        traindata = systemGenerateData(
            modelF, x1_0, x2_0, Flag_train, t_final, FlagCLF0)
        FlagCLF = traindata[9]
        print(FlagCLF)

        modelF_pre = copy.deepcopy(modelF)
        traindata_pre = traindata
        sumV_pre = traindata[1]
        sumV_pre00 = traindata[1]
        sumV_pre0 = traindata[1]
    else:
        modelF = Initialize_weight(modelF).to(device)
        modelF.load_state_dict(torch.load("saved_model_pre.pth"))


    flag_pre = 0
    if Flag_train == 1:

        for i in range(90):
            if learning_rate < learning_rate0*10e-187:  # *10e-7
                break
            train_output = train_online(traindata[2], learning_rate, modelF)

            modelF = copy.deepcopy(train_output[0])
            traindata = systemGenerateData(
                modelF, x1_0, x2_0, Flag_train, t_final, FlagCLF0)
            torch.save(modelF_pre.state_dict(), 'saved_model_test.pth',
                       _use_new_zipfile_serialization=False)

            FlagCLF = traindata[9]
            if FlagCLF == 0:
                RisCLF = False
            else:
                RisCLF = True

            if RisCLF == False:
                modelF = copy.deepcopy(modelF_pre)
                traindata = traindata_pre
                # random.uniform(0.0001, 0.999)
                learning_rate = learning_rate*0.5
                continue
            else:

                sumV = traindata[1]
                detaSum = sumV-sumV_pre
                if detaSum < 0:
                    checkCLBF0 = checkCLBF(modelF, modelBF)
                    flag_CLBF = checkCLBF0[0]
                    final_C0 = checkCLBF0[1]
                    final_C_pre = checkCLBF0[1]
                    if flag_CLBF00 == 1:
                        if flag_CLBF == 1:
                            modelF0_initial = copy.deepcopy(modelF)
                            torch.save(modelF0_initial.state_dict(), 'saved_model_initial.pth',
                                       _use_new_zipfile_serialization=False)
                            sumV_pre0 = sumV
                            flag_CLBF00 = 0

                    save_num = save_num+1
                    print(f'====================Iteration:{save_num}==================')
                    train_loss_list = train_output[4]
                    loss_list.extend(train_loss_list)
                    lossss = loss_list
                    modelF_pre = copy.deepcopy(modelF)
                    traindata_pre = traindata
                    sumV_pre = sumV
                    # * 0.5 #math.pow(0.5, (save_num))
                    learning_rate = learning_rate0

                    torch.save(modelF_pre.state_dict(), 'saved_model_pre.pth',
                               _use_new_zipfile_serialization=False)
                    continue
                else:
                    modelF = copy.deepcopy(modelF_pre)
                    traindata = traindata_pre
                    learning_rate = learning_rate*0.5
                    continue

        traindata = systemGenerateData(
            modelF_pre, x1_0, x2_0, Flag_train, t_final, FlagCLF0)
        print(
            f"sumV_initial0={sumV_pre00[0][0]};sumV_initial00={sumV_pre0[0][0]};sumV_best={traindata[1][0][0]};learning_rate={learning_rate};save_num={save_num}")
    else:
        traindata = systemGenerateData(
            modelF, x1_0, x2_0, Flag_train, t_final, FlagCLF0)
        FlagCLF = traindata[9]

        print(f"test:i={FlagCLF};dVdt > 0")

    x1_list = traindata[3]
    x2_list = traindata[4]
    u1_list = traindata[5]
    u2_list = traindata[6]

    hat_V_list = traindata[7]
    t_list = traindata[8]
    LV_list = traindata[10]
    dw1_list = traindata[11]
    dw2_list = traindata[12]

    np.save(r"loss_list.npy", loss_list)
    np.save(r"t_list.npy", t_list)
    np.save(r"x1_list.npy", x1_list)
    np.save(r"x2_list.npy", x2_list)
    np.save(r"u1_list.npy", u1_list)
    np.save(r"u2_list.npy", u2_list)
    np.save(r"hat_V_list.npy", hat_V_list)
    np.save(r"dVdt_list.npy", LV_list)
    np.save(r"dw1_list.npy", dw1_list)
    np.save(r"dw2_list.npy", dw2_list)
    # define unsafe region1

    def f(x, y):
        return (x+0.22)*(x+0.22)+(y-4.6)*(y-4.6)/10000  # -0.08
    x = np.linspace(-0.26, -0.16, 800)
    y = np.linspace(2, 7, 800)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ZC = [abs(Z) < 0.0002]
    X1 = X[tuple(ZC)]
    Y1 = Y[tuple(ZC)]

    # define unsafe region2
    x3, y3 = np.mgrid[-0.26:-0.16:500j, 2:7:500j]
    z3 = (x3+0.22)*(x3+0.22)+(y3-4.6)*(y3-4.6)/10000-0.0004

    # define state set1
    x1, y1 = np.mgrid[-0.57:0.57:500j, -15:15:500j]
    z1 = 9.35*x1*x1+0.82*x1*y1+0.02*y1*y1-0.02

    # define state set2
    x2, y2 = np.mgrid[-0.57:0.57:500j, -15:15:500j]
    z2 = 9.35*x2*x2+0.82*x2*y2+0.02*y2*y2-0.2

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t_list, x1_list, "r")
    plt.plot(t_list, x2_list, "b")
    plt.legend(labels=['x1', 'x2'])
    plt.grid()
    plt.ylabel('x1,x2')
    plt.xlabel('Time')

    plt.subplot(212)
    #plt.plot(X1, Y1, "r")
    #plt.contour(x3, y3, z3, levels=[-2, 0], colors='r', linestyles=['--'])
    plt.contour(x1, y1, z1, levels=[-2, 0], colors='k', linestyles=['--'])
    plt.contour(x2, y2, z2, levels=[-2, 0], colors='b', linestyles=['-'])
    plt.plot(x1_list, x2_list, "k")
    plt.grid()
    plt.xlim(-0.5, 0.5)
    plt.ylim(-15, 15)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('state')

    plt.figure(2)
    plt.subplot(311)
    plt.plot(t_list, u1_list, "r")
    plt.plot(t_list, u2_list, "b-.")
    plt.legend(labels=['u1', 'u2'])
    plt.grid()
    plt.ylabel('u1,u2')
    plt.xlabel('Time')
    plt.subplot(312)
    plt.plot(t_list, hat_V_list)
    plt.legend(labels=['hat_V'])
    plt.grid()
    plt.ylabel('hat_V')
    plt.xlabel('Time')
    plt.subplot(313)
    plt.plot(t_list, LV_list)
    plt.legend(labels=['dVdt'])
    plt.grid()
    plt.ylabel('dVdt')
    plt.xlabel('Time')
    plt.savefig('V')

    plt.figure(4)
    plt.subplot(511)
    plt.plot(t_list, dw1_list, "r")
    plt.grid()
    plt.ylabel('dw1')
    plt.xlabel('Time')
    plt.subplot(512)
    plt.plot(t_list, dw2_list, "b")
    plt.grid()
    plt.ylabel('dw2')
    plt.xlabel('Time')
    plt.savefig('dw')
        
    plt.figure(6)
    plt.plot(loss_list)
    plt.grid()
    plt.ylabel('TrainLoss')
    plt.xlabel('Epochs')
    plt.savefig('train')
    plt.show()