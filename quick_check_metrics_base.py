from metrics.base import BaseMetric, MetricCollection


class DummyAccuracyMetric(BaseMetric):
    """
    一个非常简单的准确率实现，只用于测试接口是否工作。
    gt 和 pred 完全相等就算预测正确。
    """

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, *, gt, pred, task=None, source=None, **kwargs) -> None:
        self.total += 1
        if gt == pred:
            self.correct += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def run_dummy_test():
    # 1) 准备一个指标集合，目前只有一个 DummyAccuracy
    acc_metric = DummyAccuracyMetric(name="accuracy")
    metrics = MetricCollection([acc_metric])

    # 2) 模拟三条样本：两条正确，一条错误
    # 注意这里只是用简单的 int 做 gt / pred，后面真正使用时会是解析后的结构
    samples = [
        {"gt": 1, "pred": 1},
        {"gt": 2, "pred": 2},
        {"gt": 3, "pred": 0},
    ]

    for s in samples:
        metrics.update(gt=s["gt"], pred=s["pred"], task="VQA2", source="dummy")

    # 3) 计算结果
    summary = metrics.compute_all()
    print("测试结果:", summary)
    # 期望：accuracy = 2/3 ≈ 0.67，保留两位小数后为 0.67


if __name__ == "__main__":
    run_dummy_test()
