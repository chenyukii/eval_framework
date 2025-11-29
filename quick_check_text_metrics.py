from utils.text_metrics import bleu_score, rouge_l, meteor_like, cider_like


def test_perfect_match():
    cand = "a small boat on the river"
    refs = ["a small boat on the river"]

    print("=== 完全匹配场景 ===")
    print("BLEU-4:", bleu_score(cand, refs, max_n=4))
    print("ROUGE-L:", rouge_l(cand, refs))
    print("METEOR:", meteor_like(cand, refs))
    print("CIDEr:", cider_like(cand, refs))


def test_partial_match():
    cand = "a boat on the river"
    refs = ["a small red boat is sailing on the river"]

    print("\n=== 部分匹配场景 ===")
    print("BLEU-4:", bleu_score(cand, refs, max_n=4))
    print("ROUGE-L:", rouge_l(cand, refs))
    print("METEOR:", meteor_like(cand, refs))
    print("CIDEr:", cider_like(cand, refs))


if __name__ == "__main__":
    test_perfect_match()
    test_partial_match()
