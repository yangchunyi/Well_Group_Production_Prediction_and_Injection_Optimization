import tensorflow
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from model import *
from utils import *


# Run model
def run_model(xdata1, xdata2, ydata, shape1, shape2, Epochs, Batch_Size, Learning_rate):
    # The training set is divided into 20% data for verification
    train_size = int(len(xdata1) * 0.8)
    train_xdata1 = xdata1[:train_size]
    train_xdata2 = xdata2[:train_size]
    train_ydata = ydata[:train_size]

    val_xdata1 = xdata1[train_size:]
    val_xdata2 = xdata2[train_size:]
    val_ydata = ydata[train_size:]

    model = myModel(shape1, shape2)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=Learning_rate)  # 0.01
    model.compile(loss='mae', optimizer=optimizer, run_eagerly=True, metrics='mse')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=10, mode='min')
    checkpoint_callback = ModelCheckpoint(
        filepath=str(group_id) + '_' + str(window_size_oil) + '_best_pre_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1)

    # Training and validation
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, mode='min', verbose=2,
                                   restore_best_weights=True)
    history = model.fit([train_xdata1, train_xdata2], train_ydata, epochs=Epochs,
                        batch_size=Batch_Size,
                        validation_data=([val_xdata1, val_xdata2], val_ydata),
                        callbacks=[reduce_lr, checkpoint_callback, early_stopping])
    # Plot loss curve
    plot_picture_mae(history)
    metric_mae = min(history.history['loss'])
    metric_val_mae = history.history['val_loss']
    metric_mse = min(history.history['mse'])
    plot_picture_mae(history)

    return metric_mae, metric_val_mae, metric_mse, model