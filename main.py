import pandas as pd

# Load the radar signal dataset from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
df = pd.read_csv(url, header=None)

# Show dataset shape and first few rows
print("Dataset shape:", df.shape)
print(df.head())
from sklearn.preprocessing import LabelEncoder

# Separate input features and labels
X = df.iloc[:, :-1].values  # all columns except the last
y_raw = df.iloc[:, -1].values  # the last column (g or b)

# Convert labels to 1 and 0
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Show what we have
print("X shape (features):", X.shape)
print("y shape (labels):", y.shape)
print("First 5 y labels:", y[:5])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
import tensorflow as tf

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=1, validation_split=0.2)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
