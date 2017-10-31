import gzip
import pickle
from itertools import chain

import regex

ALPHANUM_RE = regex.compile(r'[\p{L}]+|[\p{N}]+')


class NerCrf:
    def __init__(self, model_path):
        self.model = _load_model(model_path)

    def predict(self, text):
        text = ' '.join(ALPHANUM_RE.findall(text))
        text = [[word] for word in text.split()]

        features = [sent2features(text)]

        prediction = self.model.predict(features)
        prediction = list(chain.from_iterable(prediction))

        assert len(text) == len(prediction)

        result = [(text[i][0], prediction[i]) for i in range(len(text))]

        return result


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        # 'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word_length': len(word),
    }
    if i > 1:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })

        word2 = sent[i - 2][0]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
        })

    elif i == 1:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def _load_model(path):
    with gzip.open(path, mode='rb') as fp:
        return pickle.load(fp)
