import time
import tensorflow as tf

def validation_callback(model, val_dataset, epochs=10, batches_interval=50, verbose=True):
    history = {
        'train loss': [], 'train acc': [],
        'val loss': [], 'val acc': [],
        'times': []
    }

    def batch_end_callback(batch, logs):
        batch = batch + 1
        if batch % batches_interval == 0:
            print(f'  > Batch {batch} \t\t - train_loss: {logs["loss"]:.4f} - train_acc: {logs["accuracy"]:.4f}')

    def epoch_begin_callback(epoch, _):
        history['times'].append(time.time())
        if verbose:
            epoch = epoch+1
            print(f'Starting Epoch {epoch}/{epochs}')

    def epoch_end_callback(epoch, logs):
        train_loss, train_acc = logs.values()
        val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
        elapsed = time.time() - history['times'][-1]
        history['train loss'].append(train_loss)
        history['train acc'].append(train_acc)
        history['val loss'].append(val_loss)
        history['val acc'].append(val_acc)
        history['times'][-1] = elapsed
        if verbose:
            epoch = epoch + 1
            print(f'Ending Epoch {epoch}/{epochs}', end=' \t ')
            print(f'- train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}', end=' ')
            print(f'- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
            print(f'Elapsed Time {elapsed:.2f}s\n')

    if verbose:
        return tf.keras.callbacks.LambdaCallback(
            on_batch_end=batch_end_callback,
            on_epoch_begin=epoch_begin_callback,
            on_epoch_end=epoch_end_callback
        ), history
    else:
        return tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=epoch_begin_callback,
            on_epoch_end=epoch_end_callback
        ), history
