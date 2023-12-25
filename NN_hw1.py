import matplotlib.pyplot as plt # In order to show confusion matrix
import warnings #In order to ignore warnings
import numpy as np # In order to do mathematical calculations
import pandas as pd # In order to read csv files for test and train sets

warnings.filterwarnings("ignore") 

# Read the Train dataset CSV file
df = pd.read_csv('Dataset_train.csv')

#------------------------------------ Preprocess the data------------------------------------------
# Handle missing values if necessary
df.fillna(df.mean(), inplace=True)

# Normalize the data because there are huge range differences for each input. To prevent these huge differences to be dominant
df_normalized = (df - df.min()) / (df.max() - df.min())

# Split data into features and labels
X = df_normalized.iloc[:, :-1].values  # All columns except the last column. INPUTs
Y = df_normalized.iloc[:, -1].values   # Only the last column. OUTPUT

# Reshape Y to be a 2D array
Y = Y.reshape(-1, 1)

# Sigmoid activation function and its derivative for binary selection
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
def compute_confusion_matrix(true, pred):
    TP = sum((true == 1) & (pred == 1))
    TN = sum((true == 0) & (pred == 0))
    FP = sum((true == 0) & (pred == 1))
    FN = sum((true == 1) & (pred == 0))
    return [[TN, FP], [FN, TP]]

input_size = 11  # 11 input 1 label
hidden_neuron_size = [3,5,7]  # neuron number for hidden layer
output_size = 1  # 1 Because of binary classification
for hidden_size in hidden_neuron_size:
    # To initialize weights and biases

    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size)) # Gives a random number to each weights
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size)) # Gives a random number to each weights
    bias_hidden = np.random.uniform(size=(1, hidden_size)) # Also gives random number to every bias in hidden layer
    bias_output = np.random.uniform(size=(1, output_size)) # Also gives random number to every bias in output layer

    # Learning rate and parameters for Adam optimizer
    learning_rate = 0.01
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    m0, v0 = 0, 0
    m1, v1 = 0, 0

    def create_batches(X, Y, batchs):
        for i in range(0, len(X), batchs):
            yield (X[i:i + batchs], Y[i:i + batchs])

    # Batch size
    batchs = [16,32,64,128]  # Batch sizes are arranged
    epoch_range=300 #Epoch numbers to
    # Training the network
    for batch_size in batchs:
        for epoch in range(epoch_range):  # For every batches is gonna processed in epoch
            for X_batch, Y_batch in create_batches(X, Y, batch_size):
                print(f"Epoch number: {epoch} in {epoch_range}, batch size:{batch_size}, neuron number:{hidden_size}")
                # Forward pass 
                input_layer = X_batch
                hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden) + bias_hidden)
                output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

                # Backpropagation to teach neural network 
                error = Y_batch - output_layer # The error will decide the weight indirectly
                d_output = error * sigmoid_derivative(output_layer)
                error_hidden_layer = d_output.dot(weights_hidden_output.T)
                d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer)

                # Calculate the gradients to descent iteratively (averaged over the batch)
                grad_weights_hidden_output = hidden_layer.T.dot(d_output) / batch_size
                grad_weights_input_hidden = input_layer.T.dot(d_hidden_layer) / batch_size

                # Adam optimizer updates
                # Update from hidden to output weights
                m0 = beta1 * m0 + (1 - beta1) * grad_weights_hidden_output
                v0 = beta2 * v0 + (1 - beta2) * (grad_weights_hidden_output ** 2)
                m0_corr = m0 / (1 - beta1 ** (epoch + 1))
                v0_corr = v0 / (1 - beta2 ** (epoch + 1))
                weights_hidden_output += learning_rate * m0_corr / (np.sqrt(v0_corr) + epsilon)

                # Update from input to hidden weights
                m1 = beta1 * m1 + (1 - beta1) * grad_weights_input_hidden
                v1 = beta2 * v1 + (1 - beta2) * (grad_weights_input_hidden ** 2)
                m1_corr = m1 / (1 - beta1 ** (epoch + 1))
                v1_corr = v1 / (1 - beta2 ** (epoch + 1))
                weights_input_hidden += learning_rate * m1_corr / (np.sqrt(v1_corr) + epsilon)

                # Update biases
                bias_hidden += learning_rate * np.sum(d_hidden_layer, axis=0) / X.shape[0]
                bias_output += learning_rate * np.sum(d_output, axis=0) / X.shape[0]
            
        input_size = X.shape[1]
        #NOW it learns something
        #it's time to test :)

        # Read the test CSV file to test our neural netwrok
        test_df = pd.read_csv('Dataset_test.csv')

        # Preprocess the test data
        # to handle missing values if necessary
        test_df.fillna(test_df.mean(), inplace=True)

        # Normalize the test data
        # There is a important point use the min and max from the training data for normalization
        test_df_normalized = (test_df - df.min()) / (df.max() - df.min())

        # Split test data into features and labels
        X_test = test_df_normalized.iloc[:, :-1].values  # All columns except the last
        Y_test = test_df_normalized.iloc[:, -1].values   # Only the last column

        # Reshape Y_test to be a 2D array
        Y_test = Y_test.reshape(-1, 1)

        # Use the trained neural network to make predictions on the test data
        # Forward will pass on test data. Dot production is necessary for activation function
        hidden_layer_test = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
        output_layer_test = sigmoid(np.dot(hidden_layer_test, weights_hidden_output) + bias_output)

        # Evaluate the predictions
        # Convert probabilities to binary predictions
        # more that 50% means there is a disease
        predictions = (output_layer_test > 0.5).astype(int)

        # Compare test outpus with their real labels to find accuracy
        accuracy = np.mean(predictions == Y_test)
        confusion_matrix = compute_confusion_matrix(Y_test, predictions)

        # Plotting the confusion matrix 
        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
        plt.title(f'CM without libs Batch size: {batch_size}, Neuron number: {hidden_size} Accuracy: {accuracy * 100:.2f}%')
        fig.colorbar(cax)

        # Add labels to the plot
        classes = ['Negative', 'Positive']
        ax.set_xticklabels([''] + classes)
        ax.set_yticklabels([''] + classes)

        # Add numbers to the squares
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                ax.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center')

        plt.xlabel('Predicted')
        plt.ylabel('True')
        
print(f"Epoch range: {epoch_range}")
print(f"Learning rate: {learning_rate}")
plt.show()