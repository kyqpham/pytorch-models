from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn

n_samples = 1000
X, y = make_circles(n_samples, noise = 0.03, random_state = 42)

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

circles = pd.DataFrame({"X1": X[:, 0],
                        "X2" : X[:, 1],
                        "label": y
})

# confirming 500 labels in X0 and 500 labels in X1
print(circles.head(10))
print(circles.label.value_counts())

plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y,
            cmap = plt.cm.RdYlBu);

plt.show()

# confirming two inputs for one output
# X has two dimnesions, y has one
print(X.shape)
print(y.shape)

# turning sklearn data into tensors
x = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], y[:5])

# creating 80-20 data split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# creating the model
class CircleModelV0 (nn.Module):
    def __init__(self):
        super().__init__()
        # creating layers that hold parameters
        # these layers contain parameters

        # passing in out features to in features between layers
        # those are neurons/hidden units
        self.layer_1 = nn.Linear(in_features = 2, out_features = 5)
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1)

    # input passes through the first layer
    # feeds into the second layer and squeezed downt o one number per dot
    # this number is the classification
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
    
model_0 = CircleModelV0()

# making predictions with the model
# note that the predictions are 2D
# but the labels are 1D
untrained_preds = model_0(X_test)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# creating loss function + optimization
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# finding the logits
y_logits = model_0(X_test)
print(y_logits)

# logits through sigmoid activation, turned to labels.
# >= 0.5, y = 1 (class 1)
# < 0.5, y = 0 (class 0)
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)

# get rid of extra dimension
y_preds.squeeze()
print(y_preds)
print(y_test)

# starting the training + testing loop
torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_0.train()
    
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_function(y_logits, y_train)
    acc = accuracy_function(y_true = y_train, y_pred = y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_function(test_logits, y_test)
        test_acc = accuracy_function(y_true = y_test, y_pred = test_pred)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# model improvement
# adding alterations to the model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_1 = CircleModelV1()
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr = 0.1)

# training for longer
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # calculating loss + accuracy
    loss = loss_function(y_logits, y_train)
    acc = accuracy_function(y_true = y_train, y_pred = y_pred)

    optimizer.zero_grad()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_function(test_logits, y_test)
        test_accuracy = accuracy_function(y_true = y_test, y_pred = test_pred)

    if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# ddetermining if model can learn at all
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim = 1)
y_regression = weight * X_regression + bias

# checking the data
print(len(X_regression))
print(X_regression[:5], y_regression[:5])

train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

model_2 = nn.Sequential(
    nn.Linear(in_features = 1, out_features = 10, bias = True),
    nn.Linear(in_features = 10, out_features = 10, bias = True),
    nn.Linear(in_features = 10, out_features = 1, bias = True)
)

regression_loss_function = nn.L1Loss()
regression_optimizer = torch.optim.SGD(model_2.parameters(), lr = 0.1)

# training regression model
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    
    loss = regression_loss_function(y_pred, y_train_regression)
    regression_optimizer.zero_grad()
    loss.backward()
    regression_optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = regression_loss_function(test_pred, y_test_regression)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)

        # adding the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2()

model_3_loss_function = nn.BCEWithLogitsLoss()
model_3_optimizer = torch.optim.SGD(model_3.parameters(), lr = 0.1)

torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = model_3_loss_function(y_logits, y_train)
    acc = accuracy_function(y_true = y_train, y_pred = y_pred)

    model_3_optimizer.zero_grad()
    loss.backward()
    model_3_optimizer.step()

    model_3.eval()

    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))

        test_loss = model_3_loss_function(test_logits, y_test)
        test_acc = accuracy_function(y_true = y_test, y_pred = test_pred)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
