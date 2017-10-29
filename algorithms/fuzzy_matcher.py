import numpy as np
from py_stringmatching import Levenshtein
from scipy.optimize import linear_sum_assignment


class CustomJaccard:

    def __init__(
        self,
        sim_func=Levenshtein().get_sim_score,
        threshold=.75,
    ):

        self.sim_func = sim_func
        self.threshold = threshold

    def get_sim_score(
            self,
            left,
            right,
            word_order_matters=True,
            substring=False,
    ):
        assert isinstance(left, str), 'only strings can be compared'
        assert isinstance(right, str), 'only strings can be compared'
        
        left = left.split()
        right = right.split()

        left_len = len(left)
        right_len = len(right)

        sim_score = np.zeros((left_len, right_len))

        chars = np.zeros((left_len, right_len))

        for i, left_string in enumerate(left):
            for j, right_string in enumerate(right):
                sim_score[i][j] = -self.sim_func(left_string, right_string)
                chars[i][j] = len(left_string) + len(right_string)

        sim_score[sim_score > -self.threshold] = 0

        sim_score = sim_score * chars

        row_ind, col_ind = linear_sum_assignment(sim_score)

        result = -sim_score[row_ind, col_ind]

        optimum = result.sum()

        if substring:
            sum_chars_matched = chars[row_ind, col_ind].sum()
            result = optimum / sum_chars_matched
        else:
            left_sum_chars = sum(len(l) for l in left)
            right_sum_chars = sum(len(r) for r in right)
            result = optimum / (left_sum_chars + right_sum_chars)

        if word_order_matters:
            if row_ind[0]:
                row_ind -= row_ind[0]

            if col_ind[0]:
                col_ind -= col_ind[0]

            correct_position_cnt = np.sum(row_ind == col_ind)

            position_weight = correct_position_cnt / len(row_ind)
            result = result * position_weight

        return result
