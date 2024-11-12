import matplotlib.pyplot as plt
from setting import *

# Plot loss curve
def plot_picture_mae(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title(f'{group_id}_{window_size_oil}_mae')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{group_id}_{window_size_oil}_mae_result.png')


def plot_picture_pre(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{group_id}_{window_size_oil}_True vs Predicted Values')
    plt.legend()
    plt.savefig(f'{group_id}_{window_size_oil}_pre_result.png')