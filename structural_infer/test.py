import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.ones(1, 2, 3, 3).to(device)
print(a)


a = a + torch.zeros(a.size()).data.normal_(0, 0.1).to(device)
print(a.size(), a)