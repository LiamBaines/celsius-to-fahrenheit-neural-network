import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

xs = np.array([[-2.0], [-1.0], [0.0], [1.0], [2]], dtype=float)
ys = np.array([[28.4], [30.2], [32.0], [33.8], [35.6]], dtype=float)

model = Sequential([
  Dense(1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([5.0], dtype=float)))
