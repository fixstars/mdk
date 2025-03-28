import pytest

from src.preprocess.summerize import aggregate_filtered_qa, find_confusion_matrix


class TestSummerize:
    data_aggregate_filtered_qa = [
        (
            [
                {
                    "qa": [{"question": "これはエラーですか？", "answer": "これはエラーです"}],
                    "num_error": 1,
                }
            ],
            (
                {"wrong": [{"question": "これはエラーですか？", "answer": "これはエラーです"}]},
                {"wrong": 1},
            ),
        ),
        (
            [{"qa": [{"question": "これは正解ですか？", "answer": "はい正解です"}], "num_error": 0}],
            (
                {"clean": [{"question": "これは正解ですか？", "answer": "はい正解です"}]},
                {"clean": 1},
            ),
        ),
        (
            [{"qa": [{"question": "これは無効ですか？", "answer": "(無効)"}], "num_error": None}],
            (
                {"invalid": [{"question": "これは無効ですか？", "answer": "(無効)"}]},
                {"invalid": 1},
            ),
        ),
        (
            [
                {
                    "qa": [{"question": "これはエラーですか？", "answer": "これはエラーです"}],
                    "num_error": 1,
                },
                {
                    "qa": [{"question": "これはエラーですか？", "answer": "はい。これもエラーです"}],
                    "num_error": 1,
                },
                {
                    "qa": [{"question": "これはエラーですか？", "answer": "３つ目のエラーです"}],
                    "num_error": 1,
                },
                {"qa": [{"question": "これは正解ですか？", "answer": "はい正解です"}], "num_error": 0},
                {
                    "qa": [{"question": "これは正解ですか？", "answer": "はい、これも正解です"}],
                    "num_error": 0,
                },
                {
                    "qa": [{"question": "これは無効ですか？", "answer": "(無効)"}],
                    "num_error": None,
                },
            ],
            (
                {
                    "wrong": [
                        {"question": "これはエラーですか？", "answer": "これはエラーです"},
                        {"question": "これはエラーですか？", "answer": "はい。これもエラーです"},
                        {"question": "これはエラーですか？", "answer": "３つ目のエラーです"},
                    ],
                    "clean": [
                        {"question": "これは正解ですか？", "answer": "はい正解です"},
                        {"question": "これは正解ですか？", "answer": "はい、これも正解です"},
                    ],
                    "invalid": [{"question": "これは無効ですか？", "answer": "(無効)"}],
                },
                {"wrong": 3, "clean": 2, "invalid": 1},
            ),
        ),
    ]

    @pytest.mark.parametrize(("eval_result", "expected"), data_aggregate_filtered_qa)
    def test_aggregate_filtered_qa(self, eval_result, expected):
        qa_list, counter = aggregate_filtered_qa(eval_result)
        assert (qa_list, counter) == expected

    data_find_confusion_matrix = [
        ((0, 0), 1, 0, 0, 0, 0, 0, 0, 0, 0),
        ((0, 1), 0, 1, 0, 0, 0, 0, 0, 0, 0),
        ((0, None), 0, 0, 1, 0, 0, 0, 0, 0, 0),
        ((1, 0), 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ((1, 1), 0, 0, 0, 0, 1, 0, 0, 0, 0),
        ((1, None), 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ((None, 0), 0, 0, 0, 0, 0, 0, 1, 0, 0),
        ((None, 1), 0, 0, 0, 0, 0, 0, 0, 1, 0),
        ((None, None), 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ]

    @pytest.mark.parametrize("data", data_find_confusion_matrix)
    def test_find_confusion_matrix(self, data):
        ((pne, ne), cc, cw, ci, wc, ww, wi, uc, uw, ui) = data
        eval_result = [{"previous_num_error": pne, "num_error": ne}]
        counter = find_confusion_matrix(eval_result)
        assert counter == {
            "clean": {"clean": cc, "wrong": cw, "invalid": ci},
            "wrong": {"clean": wc, "wrong": ww, "invalid": wi},
            "unknown": {"clean": uc, "wrong": uw, "invalid": ui},
        }

    def test_find_confusion_matrix_with_minus_num_error(self):
        eval_result = [{"previous_num_error": 0, "num_error": -1}]
        with pytest.raises(AssertionError):
            find_confusion_matrix(eval_result)
