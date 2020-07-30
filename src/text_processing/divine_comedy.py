import re

DIVINE_COMEDY = open('../res/divine_comedy.txt', 'r').read()
DIVIDING_SYMBOL = '='
MARKERS = {
    'tercet': f'{DIVIDING_SYMBOL}tercet{DIVIDING_SYMBOL}',
    'canto start': f'{DIVIDING_SYMBOL}startofcanto{DIVIDING_SYMBOL}',
    'canto end': f'{DIVIDING_SYMBOL}endofcanto{DIVIDING_SYMBOL}',
    'cantica start': f'{DIVIDING_SYMBOL}startofcantica{DIVIDING_SYMBOL}',
    'cantica end': f'{DIVIDING_SYMBOL}endofcantica{DIVIDING_SYMBOL}'
}

def mark(text=DIVINE_COMEDY):
    text = '\n' + text

    # replace canto name with 'end+\n+start' marker
    text = re.sub(
        f'\n- Canto.*\n\n',
        f'{MARKERS["canto end"]}\n{MARKERS["canto start"]}\n',
        text
    )

    # replace previous cantica name + canto end with 'end+\n+start' marker
    for name in ['INFERNO', 'PURGATORIO', 'PARADISO']:
        text = re.sub(
            f'\n{name}\n{MARKERS["canto end"]}',
            f'{MARKERS["canto end"]}\n{MARKERS["cantica end"]}\n{MARKERS["cantica start"]}',
            text
        )

    # bring canto end + cantica end from the beginning to the end
    swap = f'{MARKERS["canto end"]}\n{MARKERS["cantica end"]}'
    text = text[len(swap)+1:] + '\n' + swap

    # replace double space with tercets
    text = re.sub('\n\n', f'\n{MARKERS["tercet"]}\n', text)

    return text

def unmark(text):
    for marker in MARKERS.values():
        text = text.replace(marker, '')
    return text
