import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

vessel_df = vessel_df.sort_values(['mmsi', 'timestamp'])

sequences = []
labels = []

for mmsi, group in vessel_df.groupby('mmsi'):
    group = group.sort_values('timestamp')
    
    seq = group[['length', 'width', 'speed']].values
    label = group['ship_type'].iloc[0]
    
    sequences.append(seq)
    labels.append(label)

max_len = 20
X = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post')

le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model
model = Sequential()

model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2).

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
