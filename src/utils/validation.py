import time
import tensorflow as tf

def validation_callback(model, val_dataset, epochs=10, initial_epoch=0, batches_interval=50, verbose=True):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'times': []
    }

    def batch_end_callback(batch, train):
        batch = batch + 1
        if batch % batches_interval == 0:
            print(f'  > Batch {batch} - loss {train["loss"]:.4f} - acc {train["accuracy"]:.4f}')

    def epoch_begin_callback(epoch, _):
        history['times'].append(time)
        if verbose:
            print(f'Starting Epoch {epoch:{len(str(epochs))}}/{initial_epoch + epochs}')

    def epoch_end_callback(epoch, train):
        train_loss, train_acc = train
        val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
        elapsed = time.time() - history['times'][-1]
        history['train loss'].append(train_loss)
        history['train acc'].append(train_acc)
        history['val loss'].append(val_loss)
        history['val acc'].append(val_acc)
        history['times'][-1] = elapsed
        if verbose:
            epoch = epoch + 1
            print(f'Ending Epoch {epoch:{len(str(epochs))}}/{initial_epoch+epochs}: {elapsed}s', end='')
            print(f' - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}', end='')
            print(f' - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
            print()

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
