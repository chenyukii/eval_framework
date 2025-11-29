# metrics/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MetricResult:
    """
    单个指标的汇总结果结构。

    Attributes:
        name: 指标名称，例如 "accuracy"、"AP@0.5"。
        value: 指标数值（未做百分比转换，0~1 之间或其他数值）。
        is_aux: 是否为辅助指标（True 表示辅助指标，False 表示核心指标）。
    """
    name: str
    value: float
    is_aux: bool = False


class BaseMetric(ABC):
    """
    所有指标的基类。

    设计目标：
    1. 不关心任务细节（分类/检测/描述…），只约定统一接口；
    2. 支持逐样本累积（update），最后一次性计算整体指标（compute）；
    3. 支持核心 / 辅助指标的区分，方便通过配置开关控制。

    使用方式（子类示例）：
        class AccuracyMetric(BaseMetric):
            def reset(self):
                self.correct = 0
                self.total = 0

            def update(self, *, gt, pred, task=None, source=None, **kwargs):
                self.total += 1
                if gt == pred:
                    self.correct += 1

            def compute(self) -> float:
                if self.total == 0:
                    return 0.0
                return self.correct / self.total
    """

    def __init__(self, name: str, *, is_aux: bool = False) -> None:
        """
        Args:
            name: 指标名称（用于最终报告中的 key），如 "accuracy"、"macro_f1"。
            is_aux: 是否为辅助指标，配置文件中可以按此开关控制输出。
        """
        self.name = name
        self.is_aux = is_aux
        # 子类通常在 reset 中初始化内部状态
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        清空内部状态，为一次新的评估做准备。

        注意：子类必须在这里初始化所有用于累计的成员变量，
        不要在 __init__ 里直接写累计逻辑，避免状态难以重置。
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        用一条样本更新指标的内部状态。

        Args:
            gt: 该样本的 ground truth（已经经过 DataParser 解析好的结构）。
            pred: 该样本的模型输出（同样是解析后的结构）。
            task: 任务类型字符串，例如 "图片分类"、"水平区域检测"（可选）。
            source: 数据源 / 图像路径，便于在需要时记录样本级明细（可选）。
            **kwargs: 预留扩展字段（例如置信度、类别列表等）。

        典型用法（在上层调用）：
            metric.update(gt=parsed_gt, pred=parsed_pred,
                          task=sample["task"], source=sample["source"])
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """
        根据当前累计的状态，计算最终指标数值。

        Returns:
            指标原始数值（不做百分比和保留小数位处理）。
            例如准确率 0.923，而不是 92.30。
        """
        raise NotImplementedError

    def compute_rounded(self, ndigits: int = 2) -> float:
        """
        计算指标并按需求文档要求保留指定位数小数（默认 2 位）。

        需求文档要求：所有指标保留 2 位小数。:contentReference[oaicite:2]{index=2}
        """
        value = self.compute()
        # 使用字符串格式化可以避免 0.1+0.2 这种浮点误差带来的长尾
        return float(f"{value:.{ndigits}f}")


class MetricCollection:
    """
    多个指标的集合，用于同一任务下同时计算多个指标。

    例如：
        - 图片分类：accuracy（核心）+ macro_f1（辅助）+ macro_recall（辅助）
        - 检测任务：AP@0.5（核心）+ AP@0.75（辅助）

    调用方式：
        metrics = MetricCollection([
            AccuracyMetric("accuracy"),
            MacroF1Metric("macro_f1", is_aux=True),
        ])

        for sample in samples:
            metrics.update(
                gt=sample_gt,
                pred=sample_pred,
                task=sample_task,
                source=sample_source,
            )

        summary = metrics.compute_all()
        # summary 形如：{"accuracy": 0.92, "macro_f1": 0.88}
    """

    def __init__(self, metrics: List[BaseMetric]) -> None:
        self.metrics = metrics

    def reset(self) -> None:
        """重置集合中所有指标的内部状态。"""
        for m in self.metrics:
            m.reset()

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        同时用一条样本更新集合中的所有指标。
        上层只需要调用一次，不用分别对每个指标调用 update。
        """
        for m in self.metrics:
            # 将同一条样本广播给所有指标
            m.update(gt=gt, pred=pred, task=task, source=source, **kwargs)

    def compute_all(self, ndigits: int = 2) -> Dict[str, float]:
        """
        计算集合中所有指标的数值，并按位数要求做四舍五入。

        Args:
            ndigits: 保留的小数位数，默认 2。

        Returns:
            一个字典：{指标名称: 指标值}，例如：
            {"accuracy": 0.92, "macro_f1": 0.88}
        """
        results: Dict[str, float] = {}
        for m in self.metrics:
            results[m.name] = m.compute_rounded(ndigits=ndigits)
        return results

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """
        按名称获取某一个具体指标对象，便于在需要时访问其内部明细。
        例如某些指标会记录每个样本的中间结果，可以通过这个方法拿出来。
        """
        for m in self.metrics:
            if m.name == name:
                return m
        return None
