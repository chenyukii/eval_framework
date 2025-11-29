# metrics/classification.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from .base import BaseMetric, MetricCollection


def _to_label_set(value: Any) -> Set[str]:
    """
    将各种形式的标签表示统一转换为标签集合 set[str]。

    约定：
        - None 或空字符串 -> 空集合
        - list/tuple/set -> 元素转为字符串后去空格，再放进集合
        - 普通字符串：
            - 若包含分号 ';'，按分号切分多标签
            - 否则认为是单个标签
        - 其他类型 -> str(value)
    """
    if value is None:
        return set()

    # 已经是列表 / 元组 / 集合：逐个转字符串
    if isinstance(value, (list, tuple, set)):
        result = set()
        for v in value:
            s = str(v).strip()
            if s:
                result.add(s)
        return result

    # 其他情况，先转成字符串
    s = str(value).strip()
    if not s:
        return set()

    # 多标签字符串："car;truck"
    if ";" in s:
        return {p.strip() for p in s.split(";") if p.strip()}

    # 单标签
    return {s}


class AccuracyMetric(BaseMetric):
    """
    通用准确率指标。

    适用范围：
        - 图片分类（多标签时要求“完全匹配”才算正确）
        - VQA1（Yes/No）
        - 区域分类（水平 / 旋转）
        - 其他可以用“标签集合是否相等”来判断对错的场景
    """

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # 防御式：如果有 None，直接跳过，不计入 total
        if gt is None or pred is None:
            return

        gt_set = _to_label_set(gt)
        pred_set = _to_label_set(pred)

        # 如果两边都空，也可以选择跳过，这里我们认为“没有有效标签”就不计入
        if not gt_set and not pred_set:
            return

        self.total += 1
        if gt_set == pred_set:
            self.correct += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class MacroRecallMetric(BaseMetric):
    """
    宏平均召回率（Macro-Recall）。

    对每个类别单独计算召回率：
        recall_c = TP_c / (TP_c + FN_c)

    再对所有类别取平均：
        MacroRecall = mean_c(recall_c)

    注意：
        - 这里按照“每个类别是否在 gt 中出现”来定义 TP/FN
        - 不关心 FP（预测多了不影响 recall，只影响 precision/F1）
        - 对于从未在 gt 中出现的类别（TP+FN=0），我们直接跳过，不纳入平均
    """

    def reset(self) -> None:
        from collections import defaultdict

        self.tp = defaultdict(int)
        self.fn = defaultdict(int)

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

        gt_set = _to_label_set(gt)
        pred_set = _to_label_set(pred)

        # 只对“真实为正”的样本统计 TP / FN
        for label in gt_set:
            if label in pred_set:
                self.tp[label] += 1
            else:
                self.fn[label] += 1

    def compute(self) -> float:
        labels = set(self.tp.keys()) | set(self.fn.keys())
        if not labels:
            return 0.0

        recalls = []
        for label in labels:
            tp = self.tp[label]
            fn = self.fn[label]
            if tp + fn == 0:
                continue
            recalls.append(tp / (tp + fn))

        if not recalls:
            return 0.0
        return sum(recalls) / len(recalls)


class MacroF1Metric(BaseMetric):
    """
    宏平均 F1（Macro-F1）。

    对每个类别 c 计算：
        precision_c = TP_c / (TP_c + FP_c)
        recall_c    = TP_c / (TP_c + FN_c)
        F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

    然后对所有类别取平均：
        MacroF1 = mean_c(F1_c)

    适用于：
        - 图片分类多标签场景（每个类别看成一个二分类）
        - 区域分类（单标签也没问题）
    """

    def reset(self) -> None:
        from collections import defaultdict

        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

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

        gt_set = _to_label_set(gt)
        pred_set = _to_label_set(pred)

        labels = gt_set | pred_set
        for label in labels:
            in_gt = label in gt_set
            in_pred = label in pred_set

            if in_gt and in_pred:
                self.tp[label] += 1
            elif (not in_gt) and in_pred:
                self.fp[label] += 1
            elif in_gt and (not in_pred):
                self.fn[label] += 1
            # in_gt=False & in_pred=False 的情况不会出现在 labels 里

    def compute(self) -> float:
        labels = set(self.tp.keys()) | set(self.fp.keys()) | set(self.fn.keys())
        if not labels:
            return 0.0

        f1_list = []
        for label in labels:
            tp = self.tp[label]
            fp = self.fp[label]
            fn = self.fn[label]

            if tp == 0 and fp == 0 and fn == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_list.append(f1)

        if not f1_list:
            return 0.0
        return sum(f1_list) / len(f1_list)

class AbsoluteErrorMetric(BaseMetric):
    """
    绝对误差类指标。

    用于：
        - VQA2（数量问答）：作为辅助指标（名称可以配置为 "abs_error"）
        - 计数任务：作为 MAE（名称可以配置为 "mae"）

    计算逻辑：
        - 对每条样本，计算 abs(pred - gt)
        - compute() 返回所有样本绝对误差的平均值（即 MAE）
    """

    def reset(self) -> None:
        # 记录每条样本的绝对误差，后续既可以算平均，也可以在需要时导出明细
        self.abs_errors = []

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # 如果任一为 None，说明解析失败或无效样本，直接跳过
        if gt is None or pred is None:
            return

        try:
            g = float(gt)
            p = float(pred)
        except (TypeError, ValueError):
            # 万一传进来的是奇怪类型（比如字符串里混了别的），直接跳过
            return

        self.abs_errors.append(abs(p - g))

    def compute(self) -> float:
        if not self.abs_errors:
            return 0.0
        return sum(self.abs_errors) / len(self.abs_errors)


def build_classification_metrics(task: str) -> MetricCollection:
    """
    根据任务类型构建分类 / 数量相关的指标集合。

    任务与指标对应关系（根据需求文档）：
        - 图片分类：
            - 核心：Accuracy
            - 辅助：Macro-F1, Macro-Recall
        - 水平/旋转区域分类：
            - 核心：Accuracy
            - 辅助：Macro-F1, Macro-Recall（后续可扩展混淆矩阵）
        - VQA1：
            - 核心：Accuracy
        - VQA2（数量问答）：
            - 核心：Accuracy
            - 辅助：绝对误差（Absolute Error）
        - 计数：
            - 核心：Accuracy
            - 辅助：平均绝对误差（MAE）

    返回：
        MetricCollection，其中每个指标名称可用于最终报告中的 key。
    """
    metrics: List[BaseMetric] = []

    # 所有这些任务至少有准确率
    metrics.append(AccuracyMetric(name="accuracy", is_aux=False))

    # 图片分类 & 区域分类：需要宏平均指标
    if task in {"图片分类", "水平区域分类", "旋转区域分类"}:
        metrics.append(MacroF1Metric(name="macro_f1", is_aux=True))
        metrics.append(MacroRecallMetric(name="macro_recall", is_aux=True))

    # 数量相关任务：VQA2 & 计数
    if task in {"VQA2", "计数"}:
        # 注意：这里用不同的名称，方便在报告中区分
        if task == "VQA2":
            metrics.append(AbsoluteErrorMetric(name="abs_error", is_aux=True))
        else:  # 计数任务
            metrics.append(AbsoluteErrorMetric(name="mae", is_aux=True))

    return MetricCollection(metrics)

