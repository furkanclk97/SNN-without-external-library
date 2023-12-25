import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

batchs=[64] #16, 32, 48, 128
epochs=300  #200, 400, 800
hidden_neuron_size=[7]  #3, 5, 9
learning_rate=0.005     #0.02, 0.01

# to read the training and test CSV files
train_df = pd.read_csv('Dataset_train.csv')
test_df = pd.read_csv('Dataset_test.csv')
#-------------------------Preprocess-------------------------------------
# to handle missing values for both datasets
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Split data into features and labels for both datasets
X_train = train_df.iloc[:, :-1]  # All columns except the last
Y_train = train_df.iloc[:, -1]   # Only the last column

X_test = test_df.iloc[:, :-1]  
Y_test = test_df.iloc[:, -1]   

# Normalize the data to prevent huge range dominance
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
for hidden_size in hidden_neuron_size:
    # Neural network architecture
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=X_train_scaled.shape[1], activation='relu'))  # Adjust the number of neurons in the hidden layer
    model.add(Dropout(0.3))  # Dropout 30% of the nodes
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate ), metrics=['accuracy'])
    for batch_size in batchs:
        # Train the model
        history = model.fit(X_train_scaled, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled, Y_test))

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_scaled, Y_test)
        

        
       # Plotting training history
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()

