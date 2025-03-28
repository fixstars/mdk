import pytest

from src.preprocess.json_finder import (
    any_contains,
    calc_num_error,
    calc_score,
    find_dpo,
    find_json,
    find_parenthesis,
    find_qa,
    raw_score,
    simplify_key,
    simplify_value,
    split_to_dict,
)


class TestFindJson:
    def test_find_parenthesis_simple(self):
        text = "a{bcd}e{f}"
        assert find_parenthesis(text) == ["bcd", "f"]

    def test_find_parenthesis_backquote(self):
        text = 'a\n{\n"bc,d}e{f}'
        assert find_parenthesis(text) == ['\n"bc,d', "f"]

    def test_split_to_dict(self):
        text = "s:t, u:v"
        assert split_to_dict(text) == dict(s="t", u="v")

    def test_simplify_key(self):
        assert simplify_key(" a b c ") == "abc"
        assert simplify_key("") == ""

    def test_simplify_value(self):
        assert simplify_value('"abc"') == "abc"
        assert simplify_value("'abc'") == "abc"
        assert simplify_value('"abc') == '"abc'
        assert simplify_value(" \" ' a b c ' \" ") == "a b c"
        assert simplify_value("") == ""

    def test_find_json_simple(self):
        text = "a{s:t,u:v}bc{x:y}d"
        assert find_json(text) == [dict(s="t", u="v"), dict(x="y")]

    def test_find_json_null(self):
        text = "a{null:null}bc{x:y}d"
        assert find_json(text) == [dict(null="null"), dict(x="y")]

    def test_find_json_0(self):
        text = "\n Sure, I'd be happy to help create some question and answer pairs based on the context you've provided. Here they are:\n\n```json\n{\n  \"質問\": \"OpenCLのプラットフォームで公開されるデバイスとは何を指しますか？\",\n  \"回答\": \"OpenCLのプラットフォームで公開されるデバイスは、OpenCLプラットフォームによって公開される1つ以上のデバイスを指します。\"\n}\n{\n  \"質問\": \"Kernel ObjectsとはOpenCLでどのように扱われますか？\",\n  \"回答\": \"Kernel ObjectsはOpenCL関数とその関連する引数値で、OpenCLデバイスで実行されます。\"\n}\n{\n  \"質問\": \"OpenCLでMemory Objectsとは何を指しますか？\",\n  \"回答\": \"Memory ObjectsはホストとOpenCLデバイスとで可視の変数で、インスタンスのKernelはこれらのオブジェクトを操作しながら実行されます。\"\n}\n{\n  \"質問\": \"OpenCLでコマンドクイーンとはどのように扱われますか？\",\n  \"回答\": \"OpenCLでのコマンドクイーンは、ホストがデバイスと通信するためのAPI関数を使用して作成および管理されます。コマンドクイーンには3種類のコマンドがあり、Kernel-enqueueコマンド、Memoryコマンド、およびSynchronizationコマンドです。\"\n}\n{\n  \"質問\": \"OpenCLでKernel-enqueueコマンドとはどのようなものですか？\",\n  \"回答\": \"Kernel-enqueueコマンドは、デバイスでKernelを実行するためのコマンドです。\"\n}\n{\n  \"質問\": \"OpenCLでMemoryコマンドとはどのようなものですか？\",\n  \"回答\": \"Memoryコマンドは、ホストとデバイスメモリ、メモリオブジェクト、またはメモリオブジェクトをホストアドレス空間からマップおよびアンマップすることを含むホストとデバイスメモリの間でデータを転送するコマンドです。\"\n}\n{\n  \"質問\": \"OpenCLでSynchronizationコマンドとはどのようなものですか？\",\n  \"回答\": \"Synchronizationコマンドは、コマンド間の順序制約を定義する明示的同期ポイントです。\"\n}\n{\n  \"質問\": \"OpenCLでの子Kernelはどのように実行されますか？\",\n  \"回答\": \"Kernelがデバイスで実行されているとき、デバイス上でKernelが別のデバイス上の子Kernelをエンクューにできます。\"\n}\n{\n  \"質問\": \"OpenCLでコマンドの6つの実行状態は何ですか？\",\n  \"回答\": \"OpenCLでのコマンドの6つの実行状態はQueued、Submitted、Ready、Running、Ended、Completeです。\"\n}\n```\n\nI hope these examples are helpful! If you have any other requests, feel free to ask."  # fmt: skip
        assert find_json(text) == [
            {
                "質問": "OpenCLのプラットフォームで公開されるデバイスとは何を指しますか？",
                "回答": "OpenCLのプラットフォームで公開されるデバイスは、OpenCLプラットフォームによって公開される1つ以上のデバイスを指します。",
            },
            {
                "質問": "Kernel ObjectsとはOpenCLでどのように扱われますか？",
                "回答": "Kernel ObjectsはOpenCL関数とその関連する引数値で、OpenCLデバイスで実行されます。",
            },
            {
                "質問": "OpenCLでMemory Objectsとは何を指しますか？",
                "回答": "Memory ObjectsはホストとOpenCLデバイスとで可視の変数で、インスタンスのKernelはこれらのオブジェクトを操作しながら実行されます。",
            },
            {
                "質問": "OpenCLでコマンドクイーンとはどのように扱われますか？",
                "回答": "OpenCLでのコマンドクイーンは、ホストがデバイスと通信するためのAPI関数を使用して作成および管理されます。コマンドクイーンには3種類のコマンドがあり、Kernel-enqueueコマンド、Memoryコマンド、およびSynchronizationコマンドです。",
            },
            {
                "質問": "OpenCLでKernel-enqueueコマンドとはどのようなものですか？",
                "回答": "Kernel-enqueueコマンドは、デバイスでKernelを実行するためのコマンドです。",
            },
            {
                "質問": "OpenCLでMemoryコマンドとはどのようなものですか？",
                "回答": "Memoryコマンドは、ホストとデバイスメモリ、メモリオブジェクト、またはメモリオブジェクトをホストアドレス空間からマップおよびアンマップすることを含むホストとデバイスメモリの間でデータを転送するコマンドです。",
            },
            {
                "質問": "OpenCLでSynchronizationコマンドとはどのようなものですか？",
                "回答": "Synchronizationコマンドは、コマンド間の順序制約を定義する明示的同期ポイントです。",
            },
            {
                "質問": "OpenCLでの子Kernelはどのように実行されますか？",
                "回答": "Kernelがデバイスで実行されているとき、デバイス上でKernelが別のデバイス上の子Kernelをエンクューにできます。",
            },
            {
                "質問": "OpenCLでコマンドの6つの実行状態は何ですか？",
                "回答": "OpenCLでのコマンドの6つの実行状態はQueued、Submitted、Ready、Running、Ended、Completeです。",
            },
        ]

    def test_find_json_1(self):
        # correctly split text containing ":"
        text = "{\n    \"質問\": \"OpenCLでサブグループがある場合、ワーク・アイテムは同時に実行されますか？\",\n    \"回答\": \"いいえ、サブグループ内のワーク・アイテムは同時に、しかし必ずしも並列に実行されるとは保証されません。サブグループ内のすべてのワーク・アイテムに適用される高レベルの同期コンストラクト（例: バリアなどのサブグループ関数）が定義されています。サブグループはOpenCLのバージョン2.1以前で利用できません。\"\n  }"  # fmt: skip
        assert find_json(text) == [
            {
                "質問": "OpenCLでサブグループがある場合、ワーク・アイテムは同時に実行されますか？",
                "回答": "いいえ、サブグループ内のワーク・アイテムは同時に、しかし必ずしも並列に実行されるとは保証されません。サブグループ内のすべてのワーク・アイテムに適用される高レベルの同期コンストラクト（例: バリアなどのサブグループ関数）が定義されています。サブグループはOpenCLのバージョン2.1以前で利用できません。",
            }
        ]

    def test_find_qa(self):
        text = "{\n    \"質問\": \"OpenCLでサブグループがある場合、ワーク・アイテムは同時に実行されますか？\",\n    \"回答\": \"いいえ、サブグループ内のワーク・アイテムは同時に、しかし必ずしも並列に実行されるとは保証されません。サブグループ内のすべてのワーク・アイテムに適用される高レベルの同期コンストラクト（例: バリアなどのサブグループ関数）が定義されています。サブグループはOpenCLのバージョン2.1以前で利用できません。\"\n  }"  # fmt: skip
        assert find_qa(text) == [
            {
                "question": "OpenCLでサブグループがある場合、ワーク・アイテムは同時に実行されますか？",
                "answer": "いいえ、サブグループ内のワーク・アイテムは同時に、しかし必ずしも並列に実行されるとは保証されません。サブグループ内のすべてのワーク・アイテムに適用される高レベルの同期コンストラクト（例: バリアなどのサブグループ関数）が定義されています。サブグループはOpenCLのバージョン2.1以前で利用できません。",
            }
        ]

    data_any_contains = [
        ("不正解", ["正", "correct", "right"], True),  # contains 正
        ("正解", ["正", "correct", "right"], True),  # contains 正
        ("Correct", ["正", "correct", "right"], False),  # case sensitive
        ("right", ["正", "correct", "right"], True),  # contains right
        ("問題", ["問", "q"], True),  # contains 問
        ("question", ["問", "q"], True),  # contains q
        ("Question", ["問", "q"], False),  # case sensitive
    ]

    @pytest.mark.parametrize(("s", "substr", "expected"), data_any_contains)
    def test_any_contains(self, s: str, substr: list[str], expected: bool):
        assert any_contains(s, substr) == expected

    data_num_error = [
        (  # missing the threshold
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            None,
            None,
        ),
        (  # the average is over the threshold
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            -0.15,
            1,
        ),
        (  # under the threshold
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            0.13,
            0,
        ),
        (  # wrong second attr_x
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_3]",
            ],
            -0.10,
            None,
        ),
        (  # missing attr_2
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]"
            ],
            -0.10,
            None,
        ),
    ]

    @pytest.mark.parametrize(
        ("output", "max_ok_error_threshold", "expected_num_error"), data_num_error
    )
    def test_calc_num_error(self, output, max_ok_error_threshold, expected_num_error):
        coefficient_yaml_path = "config/dataset_evaluator/coefficient.yaml"
        assert (
            calc_num_error(output, max_ok_error_threshold, coefficient_yaml_path)
            == expected_num_error
        )

    data_raw_score = [
        (  # 3 times with same score
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            [
                0.1294927980164986,
                0.1294927980164986,
                0.1294927980164986,
            ],
        ),
        (  # 2 times with different score
            [
                "helpfulness: 1 correctness: 1 coherence: 1 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 0 correctness: 0 coherence: 0 complexity: 0 verbosity: 0 [/ATTR_1]",
                "quality: 1 toxicity: 1 humor: 1 creativity: 1 [/ATTR_2]",
                "quality: 0 toxicity: 0 humor: 0 creativity: 0 [/ATTR_2]",
            ],
            [0.9710308313075455, 1.2656680851786704],
        ),
        (  # wrong second attr_x
            [
                "helpfulness: 0 correctness: 0 coherence: 0 complexity: 0 verbosity: 0 [/ATTR_1]",
                "quality: 0 toxicity: 0 humor: 0 creativity: 0 [/ATTR_1]",
            ],
            [],
        ),
        (  # missing second attr_2
            [
                "helpfulness: 0 correctness: 0 coherence: 0 complexity: 0 verbosity: 0 [/ATTR_1]",
                "helpfulness: 0 correctness: 0 coherence: 0 complexity: 0 verbosity: 0 [/ATTR_1]",
                "quality: 0 toxicity: 0 humor: 0 creativity: 0 [/ATTR_2]",
            ],
            [],
        ),
    ]

    @pytest.mark.parametrize(("output", "expected_scores"), data_raw_score)
    def test_raw_score(self, output, expected_scores):
        coefficient_yaml_path = "config/dataset_evaluator/coefficient.yaml"
        assert raw_score(output, coefficient_yaml_path) == expected_scores

    data_calc_score = [
        (  # 3 times with same score
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
                "quality: 3 toxicity: 1 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            0.1294927980164986,
        ),
        (  # different score
            [
                "helpfulness: 1 correctness: 1 coherence: 3 complexity: 1 verbosity: 2 [/ATTR_1]",
                "quality: 3 toxicity: 3 humor: 0 creativity: 1 [/ATTR_2]",
            ],
            0.5930321767764754,
        ),
        (  # wrong attr_x
            [
                "helpfulness: 1 correctness: 1 coherence: 3 complexity: 1 verbosity: 2 [/ATTR_1]",
                "quality: 3 toxicity: 3 humor: 0 creativity: 1 [/ATTR_3]",
            ],
            None,
        ),
        (  # missing attr_2
            [
                "helpfulness: 2 correctness: 2 coherence: 3 complexity: 1 verbosity: 1 [/ATTR_1]"
            ],
            None,
        ),
    ]

    @pytest.mark.parametrize(("output", "expected"), data_calc_score)
    def test_calc_score(self, output, expected):
        coefficient_yaml_path = "config/dataset_evaluator/coefficient.yaml"
        assert calc_score(output, coefficient_yaml_path) == expected

    data_find_dpo = [
        (  # with min_length ok
            '{\n    "prompt": "hello",\n    "正": "hi nice to meet you",\n    "誤": "leave me alone"\n  }',
            3,
            [
                {
                    "chosen": "hi nice to meet you",
                    "rejected": "leave me alone",
                }
            ],
        ),
        (  # with min_length ng
            '{\n    "prompt": "hello",\n    "正": "hi nice to meet you",\n    "誤": "leave me alone"\n  }',
            20,
            [],
        ),
        (  # without min_length ok
            '{\n    "prompt": "日本の首都はどこですか？",\n    "正": "日本の首都は、東京です。東京は、政治、経済、文化の中心として、世界中から注目を集めています。",\n    "誤": "トヨタ自動車の本社がある愛知県の名古屋市です。"\n  }',
            None,
            [
                {
                    "chosen": "日本の首都は、東京です。東京は、政治、経済、文化の中心として、世界中から注目を集めています。",
                    "rejected": "トヨタ自動車の本社がある愛知県の名古屋市です。",
                }
            ],
        ),
        (  # without min_length ng
            '{\n    "prompt": "hello",\n    "正": "hi nice to meet you",\n    "誤": "leave me alone"\n  }',
            None,
            [],
        ),
        (  # can't find keyword
            '{\n    "prompt": "hello",\n    "正": "hi nice to meet you",\n    "他": "leave me alone"\n  }',
            3,
            [],
        ),
    ]

    @pytest.mark.parametrize(("reply", "min_length", "expected"), data_find_dpo)
    def test_find_dpo(self, reply, min_length, expected):
        if min_length:
            assert find_dpo(reply, min_length) == expected
        else:
            assert find_dpo(reply) == expected
