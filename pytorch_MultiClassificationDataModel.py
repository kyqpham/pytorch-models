import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(y_blob)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size = 0.2,
                                                                        random_state = random_seed)

plt.figure(figsize = (10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
plt.show()

