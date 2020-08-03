import os
import tensorflow as tf

def filename(directory, file_signature, epoch):
    return directory + file_signature + epoch + '.h5'

def restore_checkpoint(model, directory, file_signature, verbose=True):
    if os.path.isdir(directory):
        checkpoints = [c for c in os.listdir(directory) if c.endswith('.h5') and file_signature in c]
        checkpoints.sort(reverse=False)
        if len(checkpoints) == 0:
            if verbose:
                print(f'could not find any model inside the directory {directory}')
            return -1
        checkpoint = checkpoints[0]
        model.load_weights(checkpoint)
        checkpoint = checkpoint.replace(directory, '')
        if verbose:
            print(f'model {checkpoint} restored')
        return int(checkpoint.replace(file_signature, '').replace('.h5', ''))
    else:
        if verbose:
            print(f'directory {directory} created')
        os.mkdir(directory)
        return 0

def checkpoint_callback(model, directory, file_signature, epochs_interval=5, verbose=True):
    def callback(epoch, _):
        if epoch % epochs_interval == 0:
            file = filename(directory, file_signature, epoch)
            model.save_weights(file)
            if verbose:
                print(f'> saving checkpoint for epoch {epoch} at {file}')
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)