import torch
from torch import nn
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

# creating a 1D tensor of values that are evenly spaced
# the values of the tensor are specified
# then, adds another dimension
X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias

# spliting training data up + assigning to new variables

train_split = int(0.8 * len(X))

# everything before train_split is assigned as training data
X_train, y_train = X[:train_split], y[:train_split]

# everything after train_split is assigned as teting data
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    
    plt.figure(figsize = (10, 7))

    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Traing data")
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

    plt.legend(prop = {"size": 14});
plot_predictions();

# creating a linear regression model
class LinearRegressionModel(nn.Module):
    # creating a blueprint that inerits from the nn.Module
    def __init__(self):
        super().__init__()

        # initializes the model parameters
        # creating random weights
        # requires_grad = True denotes that values are updated with gradient descent
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float), 
                                    requires_grad = True)
        
        # creating random biases
        # requires_grad = True denotes that values are updated with gradient descent
        self.bias = nn.Parameter (torch.randn(1, dtype = torch.float), 
                                  requires_grad = True)
    
    # defining the computations in the model
    # x is the input data
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns the linear regression formula
        return self.weights * x + self.bias

# setting seed for random generation
torch.manual_seed(42)

# creating an instance of the class
# the parameters, weight and bias, are randomly generated
model_0 = LinearRegressionModel()

# making predictions with inference mode
# the inference mode helps make predictions
with torch.inference_mode():
    # assigning predicted outputs by running test data through model
    y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions = y_preds)

lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)

# setting number of epochs
epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # trainig
    model_0.train()

    # passing input data through hte forward function
    y_pred = model_0(X_train)

    # calculating the loss
    # what is calculated vs the training data 
    loss = lossFunction(y_pred, y_train)

    # zero grad for optimize
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # updating the optimizer
    optimizer.step()

    # when it is time to test:

    # evaluating the model
    model_0.eval()
    with torch.inference_mode():
        # passing test data
        test_pred = model_0(X_test)

        # determining loss rom test data
        test_loss = lossFunction(test_pred, y_test.type(torch.float))

        # printing what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test: {test_loss} ")

# plotting loss curves
plt.plot(epoch_count, train_loss_values, label = "Train Loss")
plt.plot(epoch_count, test_loss_values, label = "Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

print("The model learned the following values for weights and bias: ")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are: ")
print(f"weights: {weight}, bias: {bias}")

