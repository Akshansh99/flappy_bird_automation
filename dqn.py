import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot

from game.flappy_bird import GameState
from prettytable import PrettyTable



class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.noa = 2
        self.gam = 0.99
        self.f_ep = 0.0001
        self.i_ep = 0.1
        self.iter = 2000000
        self.rep_mem = 10000
        self.min_size = 32

        self.layer1 = nn.Conv2d(4, 32, 8, 4)
        self.activation1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.activation2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.activation3 = nn.ReLU(inplace=True)        
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.activation4 = nn.ReLU(inplace=True)        
        self.layer3 = nn.Conv2d(128, 64, 3, 1)
        self.activation5 = nn.ReLU(inplace=True)        
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.activation6 = nn.ReLU(inplace=True)
        self.fullyconnected7 = nn.Linear(3136, 512)
        self.activation4 = nn.ReLU(inplace=True)
        self.fullyconnected8 = nn.Linear(512, self.noa)

    def forward(self, x):
        result_layer = self.layer1(x)
        result_layer = self.activation1(result_layer)
        result_layer = self.layer2(result_layer)
        result_layer = self.activation2(result_layer)
        result_layer = self.layer3(result_layer)
        result_layer = self.activation3(result_layer)        
        result_layer = self.layer4(result_layer)
        result_layer = self.activation3(result_layer)        
        result_layer = self.layer5(result_layer)
        result_layer = self.activation3(result_layer)
        result_layer = result_layer.view(result_layer.size()[0], -1)
        result_layer = self.fullyconnected4(result_layer)
        result_layer = self.activation4(result_layer)
        result_layer = self.fullyconnected5(result_layer)

        return result_layer


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available(): 
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]

    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    # pyplot.show(image_data)
    return image_data


def train(model, start):
    torch.nn.init.uniform(model.weight, -0.01, 0.01)
    model.bias.data.fill_(0.01)

    # f = open("optimizers_performance/RMSprop.txt", "a")
    # define Adam optimizer
    optimizer = optim.Adagrad(model.parameters(), lr=1e-6)

    criterion = nn.MSELoss()

    game_state = GameState()

    replay_memory = []

    action = torch.zeros([model.noa], dtype=torch.float32)
    action[0] = 1
    image_data,score, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)


    epsilon = model.i_ep
    iteration = 0

    epsilon_decrements = np.linspace(model.i_ep, model.f_ep, model.iter)

    # max_score = -1000000
    while iteration < model.iter:
        output = model(state)[0]

        #input action
        action = torch.zeros([model.noa], dtype=torch.float32)

        #epsilon greedy policy in action
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.noa, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]



        action[action_index] = 1
        #creating tensor of raw image data and for feeding to the model
        image_data_1, score,reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        #replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > model.rep_mem:
            replay_memory.pop(0)


        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.min_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()


        output_1_batch = model(state_1_batch)


        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gam * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))


        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()

        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)


        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        # max_score = max(max_score,score)

        # if iteration % 100 == 0:
        #     torch.save(model, "pretrained_model/adagrad/current_model_" + str(iteration) + ".pth")

        # if iteration % 1000 == 0:
        #     f.write("The score at iteration: "+str(iteration)+" is "+str(score)+'\n')

        # print("max score",max_score)
        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",np.max(output.cpu().detach().numpy()))
        
    # f.close()

def test(model,iter):
    f = open('rmsScore'+str(iter)+'.txt','a')

    game_state = GameState()

    action = torch.zeros([model.noa], dtype=torch.float32)
    action[0] = 1
    image_data,score, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    i=0
    started = False
    curr_score = 0

    while i<5:
        curr_score=0
        while True:
            output = model(state)[0]

            action = torch.zeros([model.noa], dtype=torch.float32)
            if torch.cuda.is_available():  
                action = action.cuda()

            # get action
            action_index = torch.argmax(output)
            if torch.cuda.is_available(): 
                action_index = action_index.cuda()
            action[action_index] = 1

            image_data_1, score,reward, terminal = game_state.frame_step(action)
            image_data_1 = resize_and_bgr2gray(image_data_1)
            image_data_1 = image_to_tensor(image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

            state = state_1
            print(f'model no. {iter} is running')
            print(str(i)+"th iteration"+str(score)+"\n")

            if score>curr_score:
                f.write(str(i)+"th iteration"+str(score)+"\n")


            if terminal or curr_score>=3500:
                f.write("================================================\n")
                f.write("The Final score is: "+str(curr_score)+"\n")
                f.write("================================================\n")
                i+=1
                break

            curr_score = score

    
    f.close()


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        curr_iteration = 2000000

        # while curr_iteration <=1300000:
        mod_name ='pretrained_model/current_model_'
        iter = mod_name+str(curr_iteration)+'.pth'
        model = torch.load(
            iter,
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  
            model = model.cuda()
        
        test(model,curr_iteration)
            # curr_iteration+=25000

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
