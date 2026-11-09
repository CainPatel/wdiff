"""Phase 3 metric functions on hand-made strings and token lists."""

from wdiff.behavioral import aggregate, has_repeat, is_refusal, last_boxed, score, to_number


def test_repeat_detection():
    assert not has_repeat(list(range(100)))
    loop = list(range(20)) * 3
    assert has_repeat(loop)
    assert not has_repeat(list(range(20)) * 2 + list(range(100, 120)))


def test_boxed_and_numbers():
    assert last_boxed(r"so \boxed{3} wait \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed("no box") is None
    assert last_boxed(r"\boxed{unclosed") is None
    assert to_number("$1,250.50") == 1250.5
    assert to_number(r"\text{18 dollars}") == 18.0
    assert to_number("none") is None


def test_refusal():
    assert is_refusal("I’m sorry, but I can’t help with that.")
    assert not is_refusal("Sure! Here is a recipe.")
    assert not is_refusal(None)


def test_score_and_aggregate():
    tok_len = lambda s: len(s.split())
    good = {"text": "<think>\nadd\n</think>\n\nThe answer is \\boxed{42}.", "tokens": [1, 2],
            "hit_cap": False}  # fmt: skip
    capped = {"text": "<think>\nhmm hmm", "tokens": [7] * 30, "hit_cap": True}
    a, b = score(good, "42", tok_len), score(capped, "42", tok_len)
    assert a["correct"] and a["answered"] and not a["loop"] and a["short"]
    assert b["loop"] and b["repeat"] and b["no_response"] and not b["answered"]
    m = aggregate([a, b])
    assert m["loop_rate"] == 0.5 and m["accuracy"] == 0.5 and m["mean_tokens"] == 16
    assert "accuracy" not in aggregate([score(good, None, tok_len)])


def test_response_without_thinking():
    tok_len = lambda s: len(s.split())
    gen = {"text": "I'm sorry, I can't help with that.", "tokens": [1], "hit_cap": False,
           "thinking": False}  # fmt: skip
    s = score(gen, None, tok_len)
    assert s["refusal"] and not s["no_response"]
