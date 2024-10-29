import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Train a simple fully connected nn in pytorch
class FFNN(nn.Module):

    def __init__(
        self,
        input_features: int,
        #hidden_layers: int = 3,
        hidden_connections: int = 5,
        dropout: int = 0.2
        ):
        super(FFNN, self).__init__()
        self.hidden1 = nn.Linear(input_features, hidden_connections)
        self.hidden2 = nn.Linear(hidden_connections, hidden_connections)
        self.hidden3 = nn.Linear(hidden_connections, hidden_connections)
        self.hidden4 = nn.Linear(hidden_connections, hidden_connections)
        self.output = nn.Linear(hidden_connections, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Feed input tensor x through the layers
        # Hidden layers use relu, output uses sigmoid activation
        # to constrain the outputs between 0 and 1 which is what
        # we need here
        x = torch.relu(self.hidden1(x))
        x = self.dropout(x)

        x = torch.relu(self.hidden2(x))
        x = self.dropout(x)
        
        x = torch.relu(self.hidden3(x))
        x = self.dropout(x)

        x = torch.relu(self.hidden4(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.output(x))

        return x

def train_model(
    X_test_tensor, num_epochs: int, model, train_loader, test_loader, optimiser, criterion, patience, manual_seed: int = None
    ) -> tuple[np.ndarray, any]:

    if manual_seed:
        # Set the random seed for reproducibility
        torch.manual_seed(manual_seed)
    else:
        torch.manual_seed(torch.initial_seed())
    
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(num_epochs):
        # Loop through all batches and train model
        for inputs, targets in train_loader:
            optimiser.zero_grad()
            
            # Feed inputs through model
            outputs = model(inputs)
            
            # Calculate loss and update weights
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
        
        # Check for early stopping
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)

        #print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        
        early_stopping(test_loss)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # Print out model performance periodically
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')

    # Make predictions
    model.eval()

    # Turn off gradient tracking for inference (Prevents needless gradient computation, saving memory)
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        
        # Convert tensor back to numpy array
        y_pred = y_pred_tensor.numpy().flatten()
    
    return y_pred

def run_training_loop(X_train, y_train, X_test, y_test) -> None:

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.astype(float).values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(float).values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.astype(float).values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.astype(float).values, dtype=torch.float32).view(-1, 1)

    # Create data loader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Perform a small grid search and save the best hyperparameters
    param_grid = {
        'initial_learning_rates': (1e-5, 1e-4, 1e-3),
        'hidden_connections': (5, 6, 7),
        'dropout': (0.2, 0.15, 0.1, 0.05)
    }

    param_combinations = []

    for learning_rate in param_grid['initial_learning_rates']:
        for connections in param_grid['hidden_connections']:
            for dropout in param_grid['dropout']:
                    param_combinations.append((connections, dropout, learning_rate))

    # Keep num epochs small whilst searching for optimal hyperparams
    num_epochs = 10

    best_params = ()
    best_mse = 0

    for params in param_combinations:
        
        connections, dropout, learning_rate = params
        
        # Initialise model, loss function and optimiser with an initial learning rate
        model = FFNN(
            input_features=X_train_tensor.shape[1],
            hidden_connections = connections,
            dropout = dropout
            )

        # Should more harshly penalise poor peak predictions
        criterion = nn.MSELoss()
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
        
        y_pred = train_model(
            num_epochs=num_epochs,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimiser=optimiser,
            criterion=criterion,
            patience=10
        )

        # Convert predictions to a pandas Series with the same index as y_test
        y_pred = pd.Series(y_pred, index=y_test.index)

        # Calculate out-of-sample mse
        mse = np.mean((y_pred - y_test) ** 2)

        print(f'Test set mse: {mse:.4f}')

        if best_mse == 0 or mse < best_mse:
            best_params = params
            best_mse = mse
            print(f'New best parameters. Connections: {connections}. Dropout: {dropout}. Learning rate: {learning_rate}')

    print(best_params)

    # Train the model for more epochs using the best parameters
    best_params = (20, 0.05, 1e-04)

    connections, dropout, learning_rate = best_params

    # Initialise model, loss function and optimiser with an initial learning rate
    model = FFNN(
        input_features=X_train_tensor.shape[1],
        hidden_connections = connections,
        dropout = dropout
        )
    # Should more harshly penalise poor peak predictions
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    y_pred = train_model(
        num_epochs=100,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimiser=optimiser,
        criterion=criterion,
        patience=10
        )
    
    # Convert predictions to a pandas Series with the same index as y_test
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Calculate residuals
    residuals = y_pred - y_test
    
    # TODO: Save weights