import unittest

from src.datasets.prepare_semantic_cache_workload import convert, summarize

ROWS = [dict(query=f"q{i}", origin_id=f"id{i // 3}") for i in range(12)]   # 4 intents x 3 paraphrases


class ConvertTests(unittest.TestCase):
    def test_maps_columns_and_validates(self):
        workload = convert(ROWS, seed=0, shuffle=False)
        self.assertEqual(workload[0], dict(request_id="s0-r0", intent_id="id0", query="q0",
                                           topic_id=None, phase_id=0, step=0))
        self.assertEqual(summarize(workload), dict(requests=12, distinct_intents=4))

    def test_shuffle_is_seeded_and_changes_order(self):
        a, b = convert(ROWS, seed=1), convert(ROWS, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual([r["query"] for r in a], [r["query"] for r in convert(ROWS, shuffle=False)])
        self.assertNotEqual([r["query"] for r in a], [r["query"] for r in convert(ROWS, seed=2)])

    def test_shuffle_keeps_every_row_and_label(self):
        shuffled = convert(ROWS, seed=5)
        self.assertEqual({(r["query"], r["intent_id"]) for r in shuffled},
                         {(r["query"], r["origin_id"]) for r in ROWS})
        self.assertEqual([r["step"] for r in shuffled], list(range(12)))

    def test_does_not_modify_the_input(self):
        before = [dict(r) for r in ROWS]
        convert(ROWS, seed=3)
        self.assertEqual(ROWS, before)

    def test_rejects_wrong_columns_and_empty_queries(self):
        with self.assertRaises(ValueError):
            convert([dict(prompt="x", origin_id="a")])
        with self.assertRaises(ValueError):
            convert([dict(query="  ", origin_id="a")])
        with self.assertRaises(ValueError):
            convert([dict(query="x", origin_id=None)])


if __name__ == "__main__":
    unittest.main()