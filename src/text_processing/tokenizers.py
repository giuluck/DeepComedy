import tensorflow_datasets as tfds
from text_processing.markers import DIVIDING_SYMBOL, MARKERS


def get_punctuation(text):
    return list(set([char for char in text if char != DIVIDING_SYMBOL and not char.isalnum()]))


def get_words(text):
    return tfds.features.text.Tokenizer().tokenize(text)


def char_tokenizer(text):
    # the vocabulary is made of:
    # - all the alphanumeric characters of the text
    # - all the punctuation symbols of the text
    # - all the markers
    alphanumeric = list(set([c for c in ''.join(get_words(text))]))
    vocabulary = alphanumeric + get_punctuation(text) + list(MARKERS.values())
    # finally, it gets sorted so that each token index will always be assigned to the same character
    vocabulary.sort()
    return tfds.features.text.TokenTextEncoder(
        vocab_list=vocabulary,
        tokenizer=tfds.features.text.Tokenizer(reserved_tokens=vocabulary),
        strip_vocab=False,
        oov_token='',
        decode_token_separator=''
    )


def word_tokenizer(text):
    # the tokenizer used splits the text in words, leaving special tokens to punctuation symbols and markers
    reserved_tokens = list(MARKERS.values()) + get_punctuation(text)
    tokenizer = tfds.features.text.Tokenizer(alphanum_only=False, reserved_tokens=reserved_tokens)
    # the vocabulary is then made of each different word in the text plus the reserved tokens
    # finally, it gets sorted so that each token index will always be assigned to the same character
    vocabulary = list(set(get_words(text))) + reserved_tokens
    vocabulary.sort()
    return tfds.features.text.TokenTextEncoder(
        vocab_list=vocabulary,
        tokenizer=tokenizer,
        strip_vocab=False,
        decode_token_separator=''
    )


def subword_tokenizer(text, target_vocab_size=2048, max_subword_length=3):
    # the vocabulary is then made of each different word in the text plus the reserved tokens
    # finally, it gets sorted so that each token index will always be assigned to the same character
    vocabulary = list(set(get_words(text))) + get_punctuation(text)
    vocabulary.sort()
    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
      corpus_generator=vocabulary,
      target_vocab_size=target_vocab_size,
      max_subword_length=max_subword_length,
      reserved_tokens=list(MARKERS.values())
    )
