# quick_check_evaluator.py
from evaluator import EvaluationCore


def test_vqa2():
    """
    测试 1：VQA2（数量问答）
        - 3 条样本，故意搞 1 条对，2 条错
    """
    core = EvaluationCore()
    task = "VQA2"

    pairs = [
        {"gt": 3, "pred": 3, "source": "img1"},  # 对
        {"gt": 5, "pred": 4, "source": "img2"},  # 差 1
        {"gt": 0, "pred": 2, "source": "img3"},  # 差 2
    ]

    summary = core.evaluate_parsed_samples(task, pairs)
    print("VQA2 指标:", summary)
    # 这里不做严格断言，只要 accuracy / abs_error 大致合理即可
    assert "accuracy" in summary
    assert "abs_error" in summary
    print("VQA2 评估测试通过 ✅")


def test_horizontal_detection():
    """
    测试 2：水平区域检测

    使用和之前 IoU/AP 测试类似的结构：
        gt / pred: (count, [box_list])
        box: (x1, y1, x2, y2)
    """
    core = EvaluationCore()
    task = "水平区域检测"

    # img1: 1 个目标，预测完全正确
    pairs = [
        {
            "gt": (1, [(0.0, 0.0, 2.0, 2.0)]),
            "pred": (1, [(0.0, 0.0, 2.0, 2.0)]),
            "source": "img1",
        },
        # img2: 2 个目标，预测也完全正确
        {
            "gt": (2, [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)]),
            "pred": (2, [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)]),
            "source": "img2",
        },
    ]

    summary = core.evaluate_parsed_samples(task, pairs)
    print("水平区域检测指标:", summary)
    assert "ap_iou_0_5" in summary
    print("水平区域检测 评估测试通过 ✅")


def test_retrieval():
    """
    测试 3：图片检索

    gt / pred: 每条样本是“图片序号列表”
    """
    core = EvaluationCore()
    task = "图片检索"

    pairs = [
        # 样本 1：完全命中 {1,2}
        {"gt": [1, 2], "pred": [2, 1], "source": "q1"},
        # 样本 2：命中 3，多了 5
        {"gt": [3], "pred": [3, 5], "source": "q2"},
        # 样本 3：命中 3，漏掉 1
        {"gt": [1, 3], "pred": [3], "source": "q3"},
        # 样本 4：一个也没命中
        {"gt": [2, 4], "pred": [], "source": "q4"},
    ]

    summary = core.evaluate_parsed_samples(task, pairs, calc_aux_metric=True)
    print("图片检索指标:", summary)

    assert "retrieval_accuracy" in summary
    assert "retrieval_recall" in summary
    assert "retrieval_f1" in summary
    print("图片检索 评估测试通过 ✅")


def test_caption():
    """
    测试 4：简洁图片描述（文本描述）

    使用我们之前在 quick_check_caption_metrics 里构造过的例子。
    """
    core = EvaluationCore()
    task = "简洁图片描述"

    gt_list = [
        "a small harbor with several boats docked near the coastline",
        "a dense urban area with high rise buildings and wide roads",
        "farmland fields divided into rectangular plots with a river nearby",
    ]

    # 完全匹配
    pairs_perfect = [
        {"gt": gt, "pred": gt, "source": f"img{i}"} for i, gt in enumerate(gt_list)
    ]

    summary_perfect = core.evaluate_parsed_samples(task, pairs_perfect)
    print("简洁描述 - 完美匹配:", summary_perfect)

    assert "cider" in summary_perfect
    assert "rouge_l" in summary_perfect
    assert "bleu_4" in summary_perfect
    assert "meteor" in summary_perfect

    print("caption 评估测试通过 ✅")


if __name__ == "__main__":
    test_vqa2()
    test_horizontal_detection()
    test_retrieval()
    test_caption()
