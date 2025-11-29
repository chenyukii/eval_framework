from metrics.classification import (
    AccuracyMetric,
    MacroF1Metric,
    MacroRecallMetric,
    build_classification_metrics,
)
from metrics.base import MetricCollection


def test_multilabel_image_classification():
    """
    测试场景 1：图片分类，多标签 + 部分预测错误

    类别集合假设是：{"car", "truck", "building"}

    样本设计（gt / pred 都用列表，模拟 DataParser 的输出）：
        样本1：gt=["car","truck"], pred=["car","truck"]      -> 完全正确
        样本2：gt=["car"],          pred=["car","truck"]      -> 多预测一个 truck
        样本3：gt=["truck"],        pred=["truck"]            -> 正确
        样本4：gt=["car","building"], pred=["building"]       -> 漏掉 car

    这样：
        - 严格匹配的准确率 = 2 / 4 = 0.5
        - Macro-F1 ≈ 0.87
        - Macro-Recall ≈ 0.89
    """

    metrics = MetricCollection([
        AccuracyMetric("accuracy"),
        MacroF1Metric("macro_f1", is_aux=True),
        MacroRecallMetric("macro_recall", is_aux=True),
    ])

    gt_samples = [
        ["car", "truck"],
        ["car"],
        ["truck"],
        ["car", "building"],
    ]
    pred_samples = [
        ["car", "truck"],
        ["car", "truck"],
        ["truck"],
        ["building"],
    ]

    for gt, pred in zip(gt_samples, pred_samples):
        metrics.update(gt=gt, pred=pred, task="图片分类", source="dummy_image")

    summary = metrics.compute_all()
    print("多标签图片分类指标:", summary)

    # 简单断言检查（允许有一点浮点误差）
    assert abs(summary["accuracy"] - 0.50) < 1e-6
    assert abs(summary["macro_f1"] - 0.87) < 0.02
    assert abs(summary["macro_recall"] - 0.89) < 0.02

    print("多标签图片分类指标测试通过 ✅")


def test_singlelabel_region_classification():
    """
    测试场景 2：区域分类（单标签）

    假设只有两类：car, truck

        样本1：gt="car",   pred="car"   -> 正确
        样本2：gt="truck", pred="car"   -> 预测错
        样本3：gt="car",   pred="car"   -> 正确

    这样：
        - accuracy = 2 / 3 ≈ 0.67
        - 对于 car：
            TP = 2, FP = 1, FN = 0
        - 对于 truck：
            TP = 0, FP = 0, FN = 1

        -> Macro-Recall = (1 + 0) / 2 = 0.5
        -> Macro-F1 = (0.8 + 0) / 2 = 0.4
    """

    metrics = build_classification_metrics("水平区域分类")

    gt_samples = ["car", "truck", "car"]
    pred_samples = ["car", "car", "car"]

    for gt, pred in zip(gt_samples, pred_samples):
        metrics.update(gt=gt, pred=pred, task="水平区域分类", source="dummy_region")

    summary = metrics.compute_all()
    print("单标签区域分类指标:", summary)

    # 这里主要是看数值大致合理，不一定非要断言
    # 你可以按需要打开下面的断言
    assert abs(summary["accuracy"] - 0.67) < 0.02
    assert abs(summary.get("macro_recall", 0.0) - 0.50) < 0.02
    assert abs(summary.get("macro_f1", 0.0) - 0.40) < 0.02

    print("单标签区域分类指标测试通过 ✅")


if __name__ == "__main__":
    test_multilabel_image_classification()
    test_singlelabel_region_classification()
