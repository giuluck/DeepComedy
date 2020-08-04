import os
import tensorflow as tf

def filename(directory, file_signature, epoch):
    return f'{directory}{file_signature}{epoch}.h5'

def restore_checkpoint(model, directory, file_signature, verbose=True):
    if os.path.isdir(directory):
        checkpoints = sorted([
            (int(c.replace(file_signature, '').replace('.h5', '')), c)
            for c in os.listdir(directory) if c.endswith('.h5') and file_signature in c
        ], reverse=True)

        if len(checkpoints) == 0:
            if verbose:
                print(f'could not find any model inside the directory {directory}\n')
            return 0
        initial_epoch, checkpoint = checkpoints[0]
        model.load_weights(directory + checkpoint)
        if verbose:
            print(f'model {checkpoint} restored\n')
        return initial_epoch
    else:
        if verbose:
            print(f'directory {directory} created\n')
        os.mkdir(directory)
        return 0

def checkpoint_callback(model, directory, file_signature, epochs_interval=5, verbose=True):
    def callback(epoch, _):
        epoch = epoch + 1
        if epoch % epochs_interval == 0:
            file = filename(directory, file_signature, epoch)
            model.save_weights(file)
            if verbose:
                print(f'> saving checkpoint for epoch {epoch} at {file}\n')
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)
