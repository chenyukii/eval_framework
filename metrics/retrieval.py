# metrics/retrieval.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import BaseMetric, MetricCollection


class RetrievalAccuracy(BaseMetric):
    """
    图片检索 - 样本级准确率（Accuracy）

    定义：
        准确率 = 检索正确的样本数 / 总样本数

    “检索正确的样本”：
        该样本下，预测的图片序号集合 == gt 中的图片序号集合（完全一致）。
    """

    def __init__(self, name: str = "retrieval_accuracy", is_aux: bool = False) -> None:
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.correct_samples = 0
        self.total_samples = 0

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # 这里的 gt / pred 期望是“当前这一条检索样本”的图片序号列表，例如 [1,2]
        if gt is None or pred is None:
            return

        gt_set = set(gt)
        pred_set = set(pred)

        # 都是空集合就不计入统计
        if not gt_set and not pred_set:
            return

        self.total_samples += 1
        if gt_set == pred_set:
            self.correct_samples += 1

    def compute(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples


class RetrievalRecall(BaseMetric):
    """
    图片检索 - 目标级召回率（Recall）

    定义：
        召回率 = 检索正确的目标样本数 / 所有真实目标样本数

    实现：
        - 每个 gt 中的每一张图像视为一个“真实目标”；
        - 若该图像出现在预测列表中，则视为“检索正确的目标样本”（TP）。
    """

    def __init__(self, name: str = "retrieval_recall", is_aux: bool = True) -> None:
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.correct_targets = 0  # TP
        self.total_targets = 0    # TP + FN

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if gt is None or pred is None:
            return

        gt_set = set(gt)
        pred_set = set(pred)

        inter = gt_set & pred_set
        self.correct_targets += len(inter)
        self.total_targets += len(gt_set)

    def compute(self) -> float:
        if self.total_targets == 0:
            return 0.0
        return self.correct_targets / self.total_targets


class RetrievalF1(BaseMetric):
    """
    图片检索 - 微平均 F1 分数

    在所有样本上累计：
        TP: 预测中命中真实目标的图片数
        FP: 预测中不在 gt 里的图片数
        FN: gt 中有但预测没命中的图片数
    """

    def __init__(self, name: str = "retrieval_f1", is_aux: bool = True) -> None:
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if gt is None or pred is None:
            return

        gt_set = set(gt)
        pred_set = set(pred)

        inter = gt_set & pred_set

        self.tp += len(inter)
        self.fp += len(pred_set - gt_set)
        self.fn += len(gt_set - pred_set)

    def compute(self) -> float:
        if self.tp + self.fp == 0:
            precision = 0.0
        else:
            precision = self.tp / (self.tp + self.fp)

        if self.tp + self.fn == 0:
            recall = 0.0
        else:
            recall = self.tp / (self.tp + self.fn)

        if precision + recall == 0.0:
            return 0.0

        return 2 * precision * recall / (precision + recall)


def build_retrieval_metrics(calc_aux_metric: bool = True) -> MetricCollection:
    """
    构建图片检索指标集合

    Args:
        calc_aux_metric: 是否计算 Recall / F1
    """
    metrics: List[BaseMetric] = [
        RetrievalAccuracy(name="retrieval_accuracy", is_aux=False)
    ]
    if calc_aux_metric:
        metrics.append(RetrievalRecall(name="retrieval_recall", is_aux=True))
        metrics.append(RetrievalF1(name="retrieval_f1", is_aux=True))
    return MetricCollection(metrics)


def calculate_retrieval_metrics(
    gt_list: Sequence[Sequence[int]],
    pred_list: Sequence[Sequence[int]],
    calc_aux_metric: bool = True,
) -> Dict[str, float]:
    """
    便捷函数：直接输入一批样本的 gt / pred，返回检索指标

    注意：这里和 MetricCollection.update 的接口对齐：
        - 按“单条样本”为单位调用 metrics.update(gt=..., pred=...)
    """
    if len(gt_list) != len(pred_list):
        raise ValueError("gt_list 和 pred_list 长度不一致")

    metric_collection = build_retrieval_metrics(calc_aux_metric=calc_aux_metric)

    for gt, pred in zip(gt_list, pred_list):
        # 关键修正：使用关键字参数 gt=..., pred=...
        metric_collection.update(gt=gt, pred=pred)

    # 统一用 compute_all，拿到保留两位小数的结果（0.xx 这种比例）
    return metric_collection.compute_all()
