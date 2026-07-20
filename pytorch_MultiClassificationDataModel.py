import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy

num_classes = 4
num_features = 2
random_seed = 42

# creating multi-class data
X_blob, y_blob = make_blobs(n_samples = 1000, 
                            n_features = num_features, 
                            centers = num_classes,
                            cluster_std = 1.5,
                            random_state = random_seed)

# turning data into tensors
# recall that x holds the coordinates, and therefore a float
X_blob = torch.from_numpy(X_blob).type(torch.float)

# y are the categories, and therefore, a tensor
# it becomes one big tensor
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(y_blob)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size = 0.2,
                                                                        random_state = random_seed)

plt.figure(figsize = (10, 7))

# x coordinate: all rows, first column
# y coordinate: all rows, second column
# color: category/lable by y
plt.scatter(X_blob[:, 0], X_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
plt.show()

# building the model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

model_4 = BlobModel(input_features = num_features, 
                    output_features = num_classes, 
                    hidden_units = 8)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr = 0.1)

def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

print(model_4(X_blob_train))

y_logits = model_4(X_blob_test)
y_pred_probs = torch.softmax(y_logits, dim = 1)

print(y_logits[:5])
print(y_pred_probs[:5])
print(torch.sum(y_pred_probs[0]))

print(torch.argmax(y_pred_probs[0]))

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

    loss = loss_function(y_logits, y_blob_train)
    acc = accuracy_function(y_true = y_blob_train, y_pred = y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)

        test_loss = loss_function(test_logits, y_blob_test)
        test_acc = accuracy_function(y_true = y_blob_test, y_pred = test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    

torchmetrics_accuracy = Accuracy(task = 'multiclass', num_classes = 4)
torchmetrics_accuracy(test_pred, y_blob_test)