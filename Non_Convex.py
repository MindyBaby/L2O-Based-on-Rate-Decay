#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")


# In[2]:


if torch.cuda.is_available():
    USE_CUDA = True  

print('USE_CUDA = {}'.format(USE_CUDA))


# In[3]:


def f(x):
    if USE_CUDA:
        x = x.cuda()
    loss_1 = ((x[0] - 1)**2).sum()
    loss_2 = 100 * ((x[1] - x[0]**2)**2).sum()
    loss = loss_1 + loss_2
    return loss


# In[4]:


'''def f(x):
    if USE_CUDA:
        x = x.cuda()
    loss = (x - 1)**2 + 100 * x**4  # 一维 Rosenbrock 函数
    return loss.sum()'''


# ## 2.构造LSTM_BlackBox优化器----2016

# In[5]:


class LSTM_BlackBox_Optimizee_Model(nn.Module):   
    def __init__(self,input_size, output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        use_bias = True
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size, bias=use_bias) #1-> output_size
    
    def LogAndSign_Preprocess_Gradient(self, gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        lr = 0.00001# 0.000005太小了   50    0.00001
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale * lr #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上 
        return update, next_state
        
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
      
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(input_gradients , prev_state, i)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = torch.add(x, update)
        return x , next_state


# In[6]:


DIM = 2
batchsize = 32
Layers = 2
Hidden_nums = 20
Input_DIM = DIM
Output_DIM = DIM
output_scale_value=1


# In[7]:


LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee_Model( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(LSTM_BlackBox_Optimizee)

if USE_CUDA:
    LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee.cuda()


# ## 2.构造Lr_LSTM_BlackBox优化器

# In[8]:


class LSTM_BlackBox_Optimizee_Model_lr(nn.Module):   
    def __init__(self,input_size, output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        use_bias = True
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size, bias=use_bias) #1-> output_size
    
    def LogAndSign_Preprocess_Gradient(self, gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def learn_rate(self, initial_lr, gamma, step, min_lr=1e-6):
        lr = initial_lr * gamma ** step
        return lr
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        lr = self.learn_rate(initial_lr = 0.00005, gamma = 0.96, step = i)#0.00005
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale * lr #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上 
        return update, next_state
        
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
      
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(input_gradients , prev_state, i)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = torch.add(x, update)
        return x , next_state


# In[9]:


DIM = 2
batchsize = 32
Layers = 2
Hidden_nums = 20
Input_DIM = DIM
Output_DIM = DIM
output_scale_value=1


# In[10]:


LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_Model_lr( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(LSTM_BlackBox_Optimizee_lr)

if USE_CUDA:
    LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_lr.cuda()


# ## 7.优化问题目标函数的学习过程

# In[11]:


class Learner( object ):
    def __init__(self,    
                 f ,  
                 optimizee,  
                 train_steps,  
                 eval_flag = False,
                 retain_graph_flag=False,
                 reset_theta = False ,
                 reset_function_from_IID_distirbution = True,
                 **options)-> None:
        
        self.f = f
        self.optimizee = optimizee
        self.train_steps = train_steps
        self.eval_flag = eval_flag
        self.retain_graph_flag = retain_graph_flag
        self.reset_theta = reset_theta
        self.reset_function_from_IID_distirbution = reset_function_from_IID_distirbution  
        self.state = None

        self.global_loss_graph = 0 #这个是为LSTM优化器求所有loss相加产生计算图准备的
        self.losses = []   # 保存每个训练周期的loss值
        self.x = torch.zeros(batchsize, DIM, requires_grad=True)

        self.sqr_grads = 0
                          
    def Reset_Or_Reuse(self , x  , state, num_roll):      
        if num_roll == 0:  
            self.x = x
            state = None
            
        if USE_CUDA:
            x = x.cuda()
            x.retain_grad()                      
        return  x , state    
        
    def __call__(self, num_roll=0) :  #全局训练
        f  = self.f 
        x , state =  self.Reset_Or_Reuse(self.x ,  self.state , num_roll )
        sqr_grads = self.sqr_grads
        self.global_loss_graph = 0   #每个unroll的开始需要重新置零
        optimizee = self.optimizee
        print('state is None = {}'.format(state == None))
        
        if optimizee == LSTM_BlackBox_Optimizee:    
            for i in range(self.train_steps):    
                loss = f(x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
          
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)

                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
            return self.losses ,self.global_loss_graph 
         
        if optimizee == LSTM_BlackBox_Optimizee_lr:   
            for i in range(self.train_steps):    
                loss = f(x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
          
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)

                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
            return self.losses ,self.global_loss_graph
        
        
        if optimizee == 'SGD':
            x.detach_()
            x.requires_grad = True
            optimizee = torch.optim.SGD([x], lr=0.001)

            for i in range(self.train_steps):
                optimizee.zero_grad()

                # 使用随机抽取的数据点计算损失
                loss = f( x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss

                loss.backward(retain_graph=self.retain_graph_flag)
                optimizee.step()
                self.losses.append(loss.detach_())

            return self.losses, self.global_loss_graph
            
        if optimizee == 'RMS': 
            x.detach_()
            x.requires_grad = True
            optimizee= torch.optim.RMSprop([x], lr=0.01, alpha=0.9 )
            
            for i in range(self.train_steps):
                
                optimizee.zero_grad()
                loss = f(x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss
                
                loss.backward(retain_graph=self.retain_graph_flag)
                optimizee.step()
                self.losses.append(loss.detach_())
 
            return self.losses, self.global_loss_graph 
        
        if optimizee == 'Adam':
            x.detach_()
            x.requires_grad = True
            optimizee= torch.optim.Adam([x],lr=0.01 )#0.005
            
            for i in range(self.train_steps):
                
                optimizee.zero_grad()
                loss = f(x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss
                
                loss.backward(retain_graph=self.retain_graph_flag)
                optimizee.step()
                self.losses.append(loss.detach_())
                
            return self.losses, self.global_loss_graph
        
        if optimizee == 'AdaGrad':
            x.detach_()
            x.requires_grad = True
            optimizee = torch.optim.Adagrad([x], lr=0.05)#0.05
            
            for i in range(self.train_steps):
                
                optimizee.zero_grad()
                loss = f(x)
                #log_loss = torch.log(loss)
                self.global_loss_graph += loss
                
                loss.backward(retain_graph=self.retain_graph_flag)
                optimizee.step()
                self.losses.append(loss.detach_())
                
            return self.losses, self.global_loss_graph


# In[12]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam' #因为这里Adam使用Pytorch
SGD = 'SGD'
RMS = 'RMS'
AdaGrad = 'AdaGrad'

#for _ in range(1): 
for loop_count in range(1):  # 在这里设置循环次数
   
    SGD_Learner = Learner(f , SGD, STEPS, eval_flag=True,reset_theta=True,)
    RMS_Learner = Learner(f , RMS, STEPS, eval_flag=True,reset_theta=True,)
    Adam_Learner = Learner(f ,  Adam, STEPS, eval_flag=True,reset_theta=True,)
    Adagrad_Learner = Learner(f,  AdaGrad, STEPS, eval_flag=True, reset_theta=True,)
    LSTM_BlackBox_learner = Learner(f , LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_BlackBox_lr_learner = Learner(f , LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    

    sgd_losses, sgd_sum_loss = SGD_Learner()
    rms_losses, rms_sum_loss = RMS_Learner()
    adam_losses, adam_sum_loss = Adam_Learner()
    adagrad_losses, adagrad_sum_loss = Adagrad_Learner()
    lstm_blackbox_losses, lstm_blackbox_sum_loss = LSTM_BlackBox_learner()
    lstm_blackbox_lr_losses, lstm_blackbox_lr_sum_loss = LSTM_BlackBox_lr_learner()

    
    sgd_losses_tensor = torch.tensor(sgd_losses)
    rms_losses_tensor = torch.tensor(rms_losses)
    adam_losses_tensor = torch.tensor(adam_losses)
    adagrad_losses_tensor = torch.tensor(adagrad_losses)
    lstm_blackbox_losses_tensor = torch.tensor(lstm_blackbox_losses)
    lstm_blackbox_lr_losses_tensor = torch.tensor(lstm_blackbox_lr_losses)
    

    p1, = plt.plot(x, sgd_losses_tensor.numpy(), label='SGD')
    p2, = plt.plot(x, rms_losses_tensor.numpy(), label='RMS')
    p3, = plt.plot(x, adam_losses_tensor.numpy(), label='Adam')
    p4, = plt.plot(x, adagrad_losses_tensor.numpy(), label='AdaGrad')
    p5, = plt.plot(x, lstm_blackbox_losses_tensor.numpy(), label='LSTM_BlackBox')
    p6, = plt.plot(x, lstm_blackbox_lr_losses_tensor.numpy(), label='LSTM_BlackBox_Lr')

    
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4, p5, p6])
    plt.title('Losses')
    plt.show()
    print("sum_loss:sgd={},rms={},adam={},adagrad={},lstm_black={},lstm_black_lr={}".format(sgd_sum_loss,rms_sum_loss,adam_sum_loss,
                                                                                            adagrad_sum_loss,
                                                                                            lstm_blackbox_sum_loss,lstm_blackbox_lr_sum_loss  ))


# ## 9.自动学习的LSTM优化器Learning to learn

# In[13]:


from timeit import default_timer as timer
def Learning_to_learn_global_training(optimizee, global_taining_steps, Optimizee_Train_Steps, UnRoll_STEPS, 
                                      Evaluate_period ,optimizer_lr=0.1):
    global_loss_list = []
    Total_Num_Unroll = Optimizee_Train_Steps // UnRoll_STEPS
    adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)

    Optimizer_Learner = Learner(f, optimizee, UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,)

    best_sum_loss = 999999999
    best_final_loss = 999999999
    best_flag = False
    start = timer()
    for i in range(Global_Train_Steps): 
        print('global training steps(全局步数): {}'.format(i))
        
        total_time = timer()
        for num in range(Total_Num_Unroll):
            
            start = timer()
            _,global_loss = Optimizer_Learner(num)   

            adam_global_optimizer.zero_grad()
            global_loss.backward() 
       
            adam_global_optimizer.step()
            global_loss_list.append(global_loss.detach_())
            
            time = timer() - start
            print(f'Epoch [{(num +1)* UnRoll_STEPS}/{Optimizee_Train_Steps}], Time: {time:.2f}, Global_Loss: {global_loss:.4f}')

        if (i + 1) % Evaluate_period == 0:
            
            best_sum_loss, best_final_loss, best_flag  = evaluate(best_sum_loss, best_final_loss, best_flag, optimizer_lr)
        
        end_time = total_time/ 3600
        print('总时间：{:.2f}h'.format(end_time))
    return global_loss_list, best_flag


# ## 2.构造LSTM_BlackBox优化器----2016

# In[14]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f , LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('Non_Convex/LSTM_BlackBox_best_loss.txt')
    except IOError:
        print ('can not find LSTM_BlackBox_best_loss.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(LSTM_BlackBox_Optimizee.state_dict(),'Non_Convex/LSTM_BlackBox_best_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'Non_Convex/LSTM_BlackBox_best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[15]:


Global_Train_Steps = 100 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 2.构造Lr_LSTM_BlackBox优化器

# In[16]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f ,LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('Non_Convex/LSTM_BlackBox_best_loss_lr.txt')
    except IOError:
        print ('can not find LSTM_BlackBox_best_loss_lr.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(LSTM_BlackBox_Optimizee_lr.state_dict(),'Non_Convex/LSTM_BlackBox_best_optimizer_lr.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'Non_Convex/LSTM_BlackBox_best_loss_lr.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[17]:


Global_Train_Steps = 100 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee_lr,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# In[18]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam' #因为这里Adam使用Pytorch
SGD = 'SGD'
RMS = 'RMS'
AdaGrad = 'AdaGrad'

#for _ in range(1): 
for loop_count in range(1):  # 在这里设置循环次数
   
    SGD_Learner = Learner(f , SGD, STEPS, eval_flag=True,reset_theta=True,)
    RMS_Learner = Learner(f , RMS, STEPS, eval_flag=True,reset_theta=True,)
    Adam_Learner = Learner(f ,  Adam, STEPS, eval_flag=True,reset_theta=True,)
    Adagrad_Learner = Learner(f,  AdaGrad, STEPS, eval_flag=True, reset_theta=True,)
    LSTM_BlackBox_learner = Learner(f , LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_BlackBox_lr_learner = Learner(f , LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    

    sgd_losses, sgd_sum_loss = SGD_Learner()
    rms_losses, rms_sum_loss = RMS_Learner()
    adam_losses, adam_sum_loss = Adam_Learner()
    adagrad_losses, adagrad_sum_loss = Adagrad_Learner()
    lstm_blackbox_losses, lstm_blackbox_sum_loss = LSTM_BlackBox_learner()
    lstm_blackbox_lr_losses, lstm_blackbox_lr_sum_loss = LSTM_BlackBox_lr_learner()

    
    sgd_losses_tensor = torch.tensor(sgd_losses)
    rms_losses_tensor = torch.tensor(rms_losses)
    adam_losses_tensor = torch.tensor(adam_losses)
    adagrad_losses_tensor = torch.tensor(adagrad_losses)
    lstm_blackbox_losses_tensor = torch.tensor(lstm_blackbox_losses)
    lstm_blackbox_lr_losses_tensor = torch.tensor(lstm_blackbox_lr_losses)
    

    p1, = plt.plot(x, sgd_losses_tensor.numpy(), label='SGD')
    p2, = plt.plot(x, rms_losses_tensor.numpy(), label='RMS')
    p3, = plt.plot(x, adam_losses_tensor.numpy(), label='Adam')
    p4, = plt.plot(x, adagrad_losses_tensor.numpy(), label='AdaGrad')
    p5, = plt.plot(x, lstm_blackbox_losses_tensor.numpy(), label='LSTM_BlackBox')
    p6, = plt.plot(x, lstm_blackbox_lr_losses_tensor.numpy(), label='LSTM_BlackBox_Lr')

    
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4, p5, p6])
    plt.title('Losses')
    plt.show()
    print("sum_loss:sgd={},rms={},adam={},adagrad={},lstm_black={},lstm_black_lr={}".format(sgd_sum_loss,rms_sum_loss,adam_sum_loss,
                                                                                            adagrad_sum_loss,
                                                                                            lstm_blackbox_sum_loss,lstm_blackbox_lr_sum_loss  ))


# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import torch

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 定义线条样式列表
linestyles = ['-', '--', '-.', ':', '-', '--']

# 定义标记样式列表
markers = ['*', 'o', 's', '^', 'D', 'x']

# Monte Carlo实验函数
def monte_carlo_experiment(num_experiments, num_steps, optimizee):
    all_losses = []
    
    for _ in range(num_experiments):
    
        if optimizee == 'SGD':
            learner = Learner(f,optimizee, num_steps, eval_flag=True, reset_theta=True)
         
        elif optimizee == 'RMS':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
            
        elif optimizee == 'Adam':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
        
        elif optimizee == 'AdaGrad':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
        
        elif optimizee == LSTM_BlackBox_Optimizee:
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == LSTM_BlackBox_Optimizee_lr:
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        losses, _ = learner()
        losses_tensor = torch.tensor(losses)
        all_losses.append(losses_tensor)        
    
    # 计算所有实验的平均损失、标准差和95%置信区间
    all_losses_array = np.array(all_losses)
    avg_losses = np.mean(all_losses_array, axis=0)
    std_losses = np.std(all_losses_array, axis=0)
    sem_losses = std_losses / np.sqrt(num_experiments)  # 标准误差
    conf_interval = 1.96 * sem_losses  # 95%置信区间
    return avg_losses, conf_interval

# 设置参数
num_experiments = 1  # 蒙特卡洛实验次数
num_steps = 100

# 进行蒙特卡洛实验
avg_losses_sgd, conf_interval_sgd = monte_carlo_experiment(num_experiments, num_steps, 'SGD')
avg_losses_rms, conf_interval_rms = monte_carlo_experiment(num_experiments, num_steps, 'RMS')
avg_losses_adam, conf_interval_adam = monte_carlo_experiment(num_experiments, num_steps, 'Adam')
avg_losses_adagrad, conf_interval_adagrad = monte_carlo_experiment(num_experiments, num_steps, 'AdaGrad')
avg_losses_lstm_blackbox, conf_interval_lstm_blackbox = monte_carlo_experiment(num_experiments, num_steps, LSTM_BlackBox_Optimizee)
avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr = monte_carlo_experiment(num_experiments, num_steps, LSTM_BlackBox_Optimizee_lr)

# 绘制结果
x = np.arange(num_steps)

plt.figure(figsize=(10, 6))
for i, (avg_losses, conf_interval, label) in enumerate([
    (avg_losses_sgd, conf_interval_sgd, 'MBGD'),
    (avg_losses_adagrad, conf_interval_adagrad, 'AdaGrad'),
    (avg_losses_rms, conf_interval_rms, 'RMSprop'),
    (avg_losses_adam, conf_interval_adam, 'Adam'),
    (avg_losses_lstm_blackbox, conf_interval_lstm_blackbox, 'LSTM-DM'),
    (avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr, 'LSTM-LR')
]):
    plt.plot(x, avg_losses, label=label, linestyle=linestyles[i % len(linestyles)], 
             marker=markers[i % len(markers)], markersize=6, color=colors[i % len(colors)], 
             markeredgewidth=0, markevery=10)  # 去掉标记边框并减少标记密度
    plt.fill_between(x, avg_losses - conf_interval, avg_losses + conf_interval, color=colors[i % len(colors)], alpha=0.2)

plt.yscale('log')
plt.title('Rosenbrock Function Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 添加图例并将其放在右上角
plt.legend(loc='upper right')
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import torch

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 定义线条样式列表
linestyles = ['-', '--', '-.', ':', '-', '--']

# 定义标记样式列表
markers = ['*', 'o', 's', '^', 'D', 'x']

# Monte Carlo实验函数
def monte_carlo_experiment(num_experiments, num_steps, optimizee):
    all_losses = []
    
    for _ in range(num_experiments):
    
        if optimizee == 'SGD':
            learner = Learner(f,optimizee, num_steps, eval_flag=True, reset_theta=True)
         
        elif optimizee == 'RMS':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
            
        elif optimizee == 'Adam':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
        
        elif optimizee == 'AdaGrad':
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True)
        
        elif optimizee == LSTM_BlackBox_Optimizee:
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == LSTM_BlackBox_Optimizee_lr:
            learner = Learner(f, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        losses, _ = learner()
        losses_tensor = torch.tensor(losses)
        all_losses.append(losses_tensor)        
    
    # 计算所有实验的平均损失、标准差和95%置信区间
    all_losses_array = np.array(all_losses)
    avg_losses = np.mean(all_losses_array, axis=0)
    std_losses = np.std(all_losses_array, axis=0)
    sem_losses = std_losses / np.sqrt(num_experiments)  # 标准误差
    conf_interval = 1.96 * sem_losses  # 95%置信区间
    return avg_losses, conf_interval

# 设置参数
num_experiments = 1  # 蒙特卡洛实验次数
num_steps = 100

# 进行蒙特卡洛实验
avg_losses_sgd, conf_interval_sgd = monte_carlo_experiment(num_experiments, num_steps, 'SGD')
avg_losses_rms, conf_interval_rms = monte_carlo_experiment(num_experiments, num_steps, 'RMS')
avg_losses_adam, conf_interval_adam = monte_carlo_experiment(num_experiments, num_steps, 'Adam')
avg_losses_adagrad, conf_interval_adagrad = monte_carlo_experiment(num_experiments, num_steps, 'AdaGrad')
avg_losses_lstm_blackbox, conf_interval_lstm_blackbox = monte_carlo_experiment(num_experiments, num_steps, LSTM_BlackBox_Optimizee)
avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr = monte_carlo_experiment(num_experiments, num_steps, LSTM_BlackBox_Optimizee_lr)

# 绘制结果
x = np.arange(num_steps)

plt.figure(figsize=(10, 6))
for i, (avg_losses, conf_interval, label) in enumerate([
    (avg_losses_sgd, conf_interval_sgd, 'MBGD'),
    (avg_losses_adagrad, conf_interval_adagrad, 'AdaGrad'),
    (avg_losses_rms, conf_interval_rms, 'RMSprop'),
    (avg_losses_adam, conf_interval_adam, 'Adam'),
    (avg_losses_lstm_blackbox, conf_interval_lstm_blackbox, 'LSTM-DM'),
    (avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr, 'LSTM-LR')
]):
    plt.plot(x, avg_losses, label=label, linestyle=linestyles[i % len(linestyles)], 
             marker=markers[i % len(markers)], markersize=6, color=colors[i % len(colors)], 
             markeredgewidth=0, markevery=10)  # 去掉标记边框并减少标记密度
    plt.fill_between(x, avg_losses - conf_interval, avg_losses + conf_interval, color=colors[i % len(colors)], alpha=0.2)

plt.yscale('log')
plt.title('Rosenbrock Function Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)  # 添加网格线
# 添加图例并将其放在右上角
plt.legend(loc='upper right')
plt.show()


# In[ ]:




