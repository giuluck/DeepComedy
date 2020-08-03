import tensorflow_datasets as tfds
from text_processing.divine_comedy import DIVINE_COMEDY, DIVIDING_SYMBOL, MARKERS


def get_punctuation(text=DIVINE_COMEDY):
    return list(set([char for char in text if char != DIVIDING_SYMBOL and not char.isalnum()]))


def get_words(text=DIVINE_COMEDY):
    return tfds.features.text.Tokenizer().tokenize(text)


def char_tokenizer(text=DIVINE_COMEDY):
    vocabulary = list(set([c for c in ''.join(get_words(text))])) + get_punctuation(text) + list(MARKERS.values())
    return tfds.features.text.TokenTextEncoder(
        vocab_list=vocabulary,
        tokenizer=tfds.features.text.Tokenizer(reserved_tokens=vocabulary),
        strip_vocab=False,
        oov_token='',
        decode_token_separator=''
    )


def word_tokenizer(text=DIVINE_COMEDY):
    reserved_tokens = list(MARKERS.values()) + get_punctuation(text)
    tokenizer = tfds.features.text.Tokenizer(alphanum_only=False, reserved_tokens=reserved_tokens)
    return tfds.features.text.TokenTextEncoder(
        vocab_list=list(set(tokenizer.tokenize(text))) + reserved_tokens,
        tokenizer=tokenizer,
        strip_vocab=False,
        decode_token_separator=''
    )


def subword_tokenizer(text=DIVINE_COMEDY, target_vocab_size=2048, max_subword_length=3):
    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
      corpus_generator=get_words(text) + get_punctuation(text),
      target_vocab_size=target_vocab_size,
      max_subword_length=max_subword_length,
      reserved_tokens=list(MARKERS.values())
    )
