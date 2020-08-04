DIVINE_COMEDY = 'divine comedy not loaded yet:\ncall utils.resources.load_divine_comedy(path) to load'

def load_divine_comedy(path='../res/divine_comedy.txt'):
    global DIVINE_COMEDY
    DIVINE_COMEDY = open(path, 'r').read()
