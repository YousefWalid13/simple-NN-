import torch
import torch.nn as nn
import torch.optim as optim

# =======================
# DATA
# =======================
images = torch.tensor([
    [[[0., 1., 0.],
      [1., 0., 1.],
      [0., 1., 0.]]],
    
    [[[1., 1., 1.],
      [1., 0., 1.],
      [1., 1., 1.]]],

    [[[0., 0., 1.],
      [0., 1., 0.],
      [1., 0., 0.]]],

    [[[1., 0., 0.],
      [1., 1., 0.],
      [1., 0., 1.]]]
]).float()

labels = torch.tensor([[0.], [1.], [0.], [1.]]).float()


# =======================
# MODEL
# =======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x


# =======================
# TRAIN
# =======================
torch.manual_seed(1)
model = SimpleCNN()
opt = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for _ in range(1000):
    opt.zero_grad()
    out = model(images)
    loss = loss_fn(out, labels)
    loss.backward()
    opt.step()

# =======================
# TEST
# =======================
test_img = torch.tensor([[[[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]]]]).float()

print("Output:", model(test_img).item())