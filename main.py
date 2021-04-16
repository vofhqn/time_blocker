import torch
from torch import nn 
from torch.nn import functional as F
from torch.nn import init
import numpy as np


from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer=SummaryWriter(comment=f"testing_n1_n2")

class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        ).to(device)
        
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    def forward(self, x):
        return self.fc(x)

class LearnerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LearnerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        ).to(device)
        
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    def forward(self, x):
        return self.fc(x)



input_dim, output_dim = 10, 5
n_data = 10000
batch_size = 32

num_epoch = 1000000
teacher = TeacherModel(input_dim, output_dim)
learner = LearnerModel(input_dim, output_dim)
optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)

data = np.random.normal(0, 1, (n_data, input_dim))
weight = np.array(range(n_data), dtype=float)
weight = weight / weight.sum()

for steps in range(num_epoch):
    batch_id = np.random.choice(n_data, batch_size, p = weight/weight.sum())
    batch_data = torch.FloatTensor(data[batch_id, :]).to(device)
    batch_data = batch_data + 0.001 * torch.randn_like(batch_data).to(device)
    teacher_pred = teacher(batch_data)
    learner_pred = learner(batch_data)

    loss = F.mse_loss(teacher_pred, learner_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("loss", loss, steps)

    # if steps % 100 == 0:
    #     tensor_data = torch.FloatTensor(data).to(device)
    #     t_pred = teacher(tensor_data).detach().cpu().numpy()
    #     l_pred = learner(tensor_data).detach().cpu().numpy()
    #     diff = t_pred - l_pred 
    #     diff = (diff ** 2).sum(axis=1)
    #     squared_norm_sum = diff.sum()
    #     squared_norm_vec = diff / squared_norm_sum
    #     norm_sum = np.sqrt(diff).sum()
    #     norm_vec = np.sqrt(diff) / norm_sum
    #     squared_dist_to_prob = np.sum( ((squared_norm_vec - weight) ** 2 ).sum() )
    #     norm_dist_to_prob = np.sum( ((norm_vec - weight) ** 2 ).sum() )
    #     writer.add_scalar("squared", squared_dist_to_prob, steps)
    #     writer.add_scalar("norm", norm_dist_to_prob, steps)
    #     print("weight", weight)
    #     print("squared", squared_norm_vec)
    #     print("norm", norm_vec)        
    if steps % 5000 == 0:
        tensor_data = torch.FloatTensor(data).to(device)
        t_pred = teacher(tensor_data).detach().cpu().numpy()
        l_pred = learner(tensor_data).detach().cpu().numpy()
        diff = t_pred - l_pred 
        diff = (diff ** 2).sum(axis=1)

        diff = diff.reshape(-1)

        pows = [1.0]

        # for p in pows:
        #     for i in range(n_data):
        #         writer.add_scalar(f"diff with pow {p}, steps {steps}", np.power(diff[i], p), i)
        
        for p in pows:
            for i in range(n_data):
                writer.add_scalar(f"1 over pow(diff,{p}) {steps}", -np.log( diff[i] ), i)


        # for i in range(n_data):
        #     writer.add_scalar(f"diff with log", np.log(diff[i]), i)

# for i in range(n_data):
#     writer.add_scalar(f"diff with 1/log", 1./np.log(diff[i]), i)
# for i in range(n_data):
#     writer.add_scalar("diff square norm", diff[i], i)
#     writer.add_scalar("diff norm", np.sqrt(diff[i]), i)

# squared_norm_sum = diff.sum()
# squared_norm_vec = diff / squared_norm_sum
# norm_sum = np.sqrt(diff).sum()
# norm_vec = np.sqrt(diff) / norm_sum
# squared_dist_to_prob = np.sum( ((squared_norm_vec - weight) ** 2 ).sum() )
# norm_dist_to_prob = np.sum( ((norm_vec - weight) ** 2 ).sum() )
# writer.add_scalar("squared", squared_dist_to_prob, steps)
# writer.add_scalar("norm", norm_dist_to_prob, steps)
# print("weight", weight)
# print("squared", squared_norm_vec)
# print("norm", norm_vec)        
