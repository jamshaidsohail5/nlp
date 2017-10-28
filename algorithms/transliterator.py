import pickle

import sklearn_crfsuite

from mappings.abc import ABC

CRF_PARAMS = {
    'algorithm': 'lbfgs',
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 500,
    'all_possible_transitions': True,
    'verbose': False,
}

MODEL_PATH = 'models/transliterator/'


class Transliterator():
    def __init__(self, language_code):
        language_code = language_code.lower()
        model_path = MODEL_PATH + language_code + '.pkl'

        self.model = _load_model(model_path)
        self.language_code = language_code

    def transliterate(self, text):
        if not text:
            return None

        words = text.lower().split()
        features = [_word2features(word) for word in words]
        prediction = self.model.predict(features)

        # return original text formatting and non-alphabet chars
        for w, word in enumerate(text.split()):
            word_is_upper = word.isupper()

            for c, char in enumerate(word):
                if char.lower() not in ABC[self.language_code]:
                    prediction[w][c] = char
                    continue

                if word_is_upper:
                    prediction[w][c] = prediction[w][c].upper()
                elif char.istitle():
                    prediction[w][c] = prediction[w][c].title()

        result = []

        for word in prediction:
            result.append(''.join(word))

        return ' '.join(result)


def append_to_dataset(source, output, filepath):
    source = '|'.join(list(source))
    assert source.count('|') == output.count('|')

    text = '^'.join([source, output]) + '\n'

    with open(filepath, mode='a', encoding='utf-8') as fp:
        fp.writelines(text)


def load_dataset(dataset_path):
    dataset = []

    with open(dataset_path, mode='rt', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip().split('^')
            source = line[0].split('|')
            output = line[1].split('|')
            temp = []

            assert len(source) == len(output)

            for i in range(len(source)):
                temp.append((source[i], output[i]))

            dataset.append(temp)

    return dataset


def save_model(model, path):
    with open(path, mode='wb') as fp:
        pickle.dump(model, fp)


def train_model(dataset, crf_params=CRF_PARAMS):
    x_train = [_word2features(word) for word in dataset]
    y_train = [_word2label(word) for word in dataset]

    model = sklearn_crfsuite.CRF(**crf_params)

    model.fit(x_train, y_train)

    return model


def _char2features(word, char_position):
    char = word[char_position][0]

    features = {
        'bias': 1.0,
        'char': char,
        '1_prev_char': _get_chars(word, char_position, offset=-1),
        '2_prev_char': _get_chars(word, char_position, offset=-2),
        '3_prev_char': _get_chars(word, char_position, offset=-3),
        '1_next_char': _get_chars(word, char_position, offset=1),
        '2_next_char': _get_chars(word, char_position, offset=2),
        '3_next_char': _get_chars(word, char_position, offset=3),
    }

    features = {key: features[key] for key in features if features[key]}

    return features


def _get_chars(word, current_position, offset):
    assert offset != 0

    if offset < 0:
        position = max(current_position + offset, 0)
        temp = word[position:current_position]
    else:
        position = min(current_position + offset + 1, len(word))
        current_position = min(current_position + 1, len(word))
        temp = word[current_position:position]

    result = []

    for char in temp:
        result.append(char[0])

    result = ''.join(result)

    if len(result) == abs(offset):
        return result
    else:
        return None


def _load_model(path):
    with open(path, mode='rb') as fp:
        return pickle.load(fp)


def _word2features(word):
    return [_char2features(word, i) for i in range(len(word))]


def _word2label(word):
    return [output for source, output in word]
