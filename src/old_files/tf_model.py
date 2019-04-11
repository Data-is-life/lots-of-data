import torch
import torch.utils.data
from torch import device
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MaxAbsScaler as RAS
from col_info import all_cols
from dummies_bins_test_train_cv import initial_df
from dummies_bins_test_train_cv import bin_df_get_y
from dummies_bins_test_train_cv import partial_df
from dummies_bins_test_train_cv import xy_custom
device = device("cuda:0")


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(len_X_h, 160)
        self.fc2 = nn.Linear(160, 320)
        self.fcd = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(320, 80)
        self.fc4 = nn.Linear(80, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fcd(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


df = initial_df('../data/use_for_predictions.csv')

df_s = partial_df(df, 1, 0, 10, 'n')
df_s, y = bin_df_get_y(df_s)
clm = all_cols[0]
print(clm)
X_train, X_test, y_train, y_test, X = xy_custom(df_s, y, 0.90, clm)
ras = RAS().fit(X_train)
X_train = ras.transform(X_train)
X_test = ras.transform(X_test)
ly = (len(y_train))
y_train.resize(ly, 1)
y_train = torch.as_tensor(y_train, dtype=torch.float)
y_train = y_train.cuda().to(device)
X_train = torch.FloatTensor(X_train).cuda().to(device)
len_X_h = len(X_train[0])
len_X = len(X_train)
sae = SAE().cuda().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adagrad(sae.parameters(), lr=1e-3, weight_decay=0.115)
epochs = 200
train_loss = 0
s = 0.
for epoch in range(epochs):
    for num in range(len_X):
        input = Variable(X_train)
        target = y_train
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
#            mean_corrector = len_X/float(torch.sum(target.data > 0) + 1e-10)
            # optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data
            s += 1.
            optimizer.step()
    print(f'epoch: {epoch+1} loss: {train_loss/s}')

s = 0.
while s <= 20:
    test_loss = 0
    y_test = y_test[int(ly*s/21):ly]
    ly = (len(y_test))
    y_test.resize(ly, 1)
    y_test = torch.as_tensor(y_test, dtype=torch.float)
    y_test = y_test.cuda().to(device)
    X_test = torch.FloatTensor(X_test).cuda().to(device)
    len_X = len(X_test)
    for num in range(len_X):
        input = Variable(X_test)
        target = Variable(y_test)
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
#            mean_corrector = len_X/float(torch.sum(target.data > 0.5) + 1e-10)
            test_loss += loss.data
            s += 1.
            print(f'test loss: {test_loss/s}')

print(torch.sum(target.data > 0))

print(loss.backward())
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in sae.state_dict():
    print(param_tensor, "\t", sae.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(sae.state_dict(), '../data/tm-bce')
