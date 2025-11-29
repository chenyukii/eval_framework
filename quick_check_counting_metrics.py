from metrics.classification import build_classification_metrics


def test_vqa2_metrics():
    """
    测试场景：VQA2（数量问答）

    设计 3 条样本：
        1) gt=3, pred=3 -> 正确，误差 0
        2) gt=5, pred=4 -> 错误，误差 1
        3) gt=0, pred=2 -> 错误，误差 2

    期望：
        - accuracy = 1 / 3 ≈ 0.33
        - abs_error = (0 + 1 + 2) / 3 = 1.0
    """
    metrics = build_classification_metrics("VQA2")

    gts = [3, 5, 0]
    preds = [3, 4, 2]

    for gt, pred in zip(gts, preds):
        metrics.update(gt=gt, pred=pred, task="VQA2", source="vqa2_dummy")

    summary = metrics.compute_all()
    print("VQA2 指标:", summary)

    # 因为 compute_all 里保留两位小数，所以 accuracy 会是 0.33 左右
    assert abs(summary["accuracy"] - 0.33) < 0.02
    # 绝对误差应该是 1.0
    assert abs(summary["abs_error"] - 1.0) < 1e-6

    print("VQA2 指标测试通过 ✅")


def test_counting_metrics():
    """
    测试场景：计数任务

    设计 3 条样本：
        1) gt=10, pred=8  -> 误差 2
        2) gt=3,  pred=3  -> 误差 0
        3) gt=7,  pred=10 -> 误差 3

    期望：
        - accuracy = 1 / 3 ≈ 0.33
        - mae = (2 + 0 + 3) / 3 = 5/3 ≈ 1.67
    """
    metrics = build_classification_metrics("计数")

    gts = [10, 3, 7]
    preds = [8, 3, 10]

    for gt, pred in zip(gts, preds):
        metrics.update(gt=gt, pred=pred, task="计数", source="count_dummy")

    summary = metrics.compute_all()
    print("计数任务指标:", summary)

    assert abs(summary["accuracy"] - 0.33) < 0.02
    assert abs(summary["mae"] - 1.67) < 0.02

    print("计数任务指标测试通过 ✅")


if __name__ == "__main__":
    test_vqa2_metrics()
    test_counting_metrics()
