import numpy as np


class StringSplitter:
    def __init__(self, words_vocab):
        self.max_word_length = max(len(word) for word in words_vocab)
        self.word_cost = dict()

        for rank, word in enumerate(words_vocab):
            self.word_cost[word] = np.log(
                (rank + 1) * np.log(len(words_vocab)),
            )

    def split(self, text):
        assert isinstance(text, str), 'Only string can be splitted!'

        text = text.lower()

        # Build the cost array.
        cost_array = [0]
        for num_chars in range(1, len(text) + 1):

            cost, match_length = _best_match(
                num_chars,
                cost_array=cost_array,
                max_word_length=self.max_word_length,
                text=text,
                word_cost=self.word_cost,
            )

            cost_array.append(cost)

        # Backtrack to recover the minimal-cost string.
        result = []
        text_length = len(text)
        while text_length > 0:

            cost, match_length = _best_match(
                text_length,
                cost_array=cost_array,
                max_word_length=self.max_word_length,
                text=text,
                word_cost=self.word_cost,
            )

            assert cost == cost_array[text_length]
            result.append(text[text_length - match_length:text_length])
            text_length -= match_length

        return ' '.join(reversed(result))


def _best_match(num_chars, cost_array, max_word_length, text, word_cost):
    """
    Find the best match for the num_chars first characters.

    Assuming cost has been built for the i-1 first characters.
    Returns a pair (match_cost, match_length).
    """
    candidates = reversed(
        cost_array[max(0, num_chars - max_word_length):num_chars],
    )

    result = []
    for iter_num, candidate in enumerate(candidates):
        chars_to_check = text[num_chars - iter_num - 1:num_chars]
        temp = (
            candidate + word_cost.get(chars_to_check, np.inf),
            iter_num + 1,
        )
        result.append(temp)

    return min(result)
