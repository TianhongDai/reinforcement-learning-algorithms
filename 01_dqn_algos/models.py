import torch
import torch.nn as nn
import torch.nn.functional as F

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)

        return x

# in the initial, just the nature CNN
class net(nn.Module):
    def __init__(self, num_actions, use_dueling=False):
        super(net, self).__init__()
        # if use the dueling network
        self.use_dueling = use_dueling
        # define the network
        self.cnn_layer = deepmind()
        # if not use dueling
        if not self.use_dueling:
            self.fc1 = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(32 * 7 * 7, 256)
            self.state_value_fc = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
            self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        if not self.use_dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            # get the action value
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            # get the state value
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            # action value mean
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            # Q = V + A
            action_value_out = state_value + action_value_center
        return action_value_out
