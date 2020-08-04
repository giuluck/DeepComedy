import numpy as np
from metrics.metrics import evaluate

def store(result_file, temperature, samples, verbose=True):
    metrics = [
        'putative tercets', 'well-formed tercets', 'structuredness',
        'hendecasyllabicness', 'rhymeness', 'plagiarism'
    ]

    result_file.write(f'> Temperature Factor: {temperature}\n')
    if verbose:
        print(f'> Temperature Factor: {temperature}')

    evaluations = np.zeros(len(metrics))
    for sample in samples:
        evaluations += np.array([v for v in evaluate(sample).values()])

    for m, e in zip(metrics, evaluations / len(samples)):
        result_file.write(f'  - {m} --> {e:.2f}\n')
        if verbose:
            print(f'  - {m} --> {e:.2f}')

    result_file.write('\n')
    if verbose:
        print()
