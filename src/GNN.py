import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from spektral.layers import GCNConv

# X: node features
# y: labels
# A: adjacency matrix

def build_adj_matrix(X, k=10):
    """
    Build adjacency matrix using KNN
    X: node features [N, F]
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    N = X.shape[0]
    A = np.zeros((N, N))

    for i in range(N):
        for j in indices[i]:
            A[i][j] = 1
            A[j][i] = 1  # undirected graph

    return A

# Model
class GCN(tf.keras.Model):
    def __init__(self, n_classes):
        super().__init__()

        self.conv1 = GCNConv(64, activation='relu')
        self.conv2 = GCNConv(64, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs  # node features + adjacency matrix

        x = self.conv1([x, a])
        x = self.conv2([x, a])

        x = self.flatten(x)
        return self.dense(x)

model = GCN(n_classes=4)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x=[X, A],
    y=y,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
