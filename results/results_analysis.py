import os

thresholds = {
    'putative tercets': 0,
    'well-formed tercets': 0,
    'structuredness': 0.97,
    'hendecasyllabicness': 0.90,
    'rhymeness': 0.86,
    'plagiarism': 0.9,
    'repetitivity': 0.9
}

def check_temperature_configuration(temperature_configuration):
    lines = temperature_configuration.split('\n')
    putative_tercets = float(lines[1][25:])
    well_formed_tercets = float(lines[2][28:])
    structuredness = float(lines[3][23:])
    hendecasyllabicness = float(lines[4][28:])
    rhymeness = float(lines[5][18:])
    plagiarism = float(lines[6][19:])
    repetitivity = float(lines[7][21:])
    return putative_tercets >= thresholds['putative tercets'] \
        and well_formed_tercets >= thresholds['well-formed tercets'] \
        and structuredness >= thresholds['structuredness'] \
        and hendecasyllabicness >= thresholds['hendecasyllabicness'] \
        and rhymeness >= thresholds['rhymeness'] \
        and plagiarism >= thresholds['plagiarism'] \
        and repetitivity >= thresholds['repetitivity']

def process_results(filepath):
    at_least_one = False
    with open(filepath) as f:
        for temperature_configuration in [tc for tc in f.read().split('\n\n') if len(tc) > 0]:
            if check_temperature_configuration(temperature_configuration):
                if not at_least_one:
                    print(filepath[2:-4].upper())
                print(temperature_configuration)
                at_least_one = True
        if at_least_one:
            print()

if __name__ == '__main__':
    for fp in [path + '/' + f for path, _, files in os.walk('./') for f in files if f.endswith('.txt')]:
        process_results(fp)
