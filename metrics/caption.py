# metrics/caption.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import BaseMetric, MetricCollection
from ..utils.text_metrics import _ensure_coco_evalcap


class CaptionBLEUMetric(BaseMetric):
    """
    图片/区域描述任务的 BLEU 指标（官方 MS-COCO 实现，BLEU-4）。

    - 使用 pycocoevalcap.bleu.Bleu 计算 corpus-level BLEU
    - 累积所有样本的 (gt, pred)，在 compute() 阶段一次性算分
    """

    def __init__(
        self,
        name: str = "bleu_4",
        max_n: int = 4,
        is_aux: bool = True,
    ) -> None:
        # 延迟导入官方实现，只有真正用到 caption 指标时才依赖 pycocoevalcap
        _ensure_coco_evalcap()
        from pycocoevalcap.bleu.bleu import Bleu  # type: ignore

        self.max_n = max_n
        # COCO 默认就是 4-gram BLEU
        self._scorer = Bleu(n=4)
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        # img_id -> [ref1, ref2, ...]
        self._gts: Dict[str, List[str]] = {}
        # img_id -> [hypo]
        self._res: Dict[str, List[str]] = {}
        self._idx: int = 0

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
        每次调用处理一条样本（一个 gt caption + 一个模型 caption）。
        DataParser 对应任务会把 gt / pred 解析为字符串。
        """
        if gt is None or pred is None:
            return

        gt_str = str(gt).strip()
        pred_str = str(pred).strip()
        if not gt_str and not pred_str:
            return

        img_id = f"{self._idx}"
        self._idx += 1

        # 当前数据集每条样本只有 1 个参考 caption
        self._gts.setdefault(img_id, []).append(gt_str)
        self._res[img_id] = [pred_str]

    def compute(self) -> float:
        if not self._gts or not self._res:
            return 0.0

        score, _ = self._scorer.compute_score(self._gts, self._res)
        # score 是 [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
        if isinstance(score, (list, tuple)):
            idx = min(self.max_n, len(score)) - 1
            return float(score[idx])
        return float(score)


class CaptionROUGELMetric(BaseMetric):
    """
    图片/区域描述任务的 ROUGE-L 指标（官方 MS-COCO 实现）。

    使用 pycocoevalcap.rouge.Rouge 计算 corpus-level ROUGE-L。
    """

    def __init__(self, name: str = "rouge_l", is_aux: bool = False) -> None:
        _ensure_coco_evalcap()
        from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

        self._scorer = Rouge()
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self._gts: Dict[str, List[str]] = {}
        self._res: Dict[str, List[str]] = {}
        self._idx: int = 0

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

        gt_str = str(gt).strip()
        pred_str = str(pred).strip()
        if not gt_str and not pred_str:
            return

        img_id = f"{self._idx}"
        self._idx += 1

        self._gts.setdefault(img_id, []).append(gt_str)
        self._res[img_id] = [pred_str]

    def compute(self) -> float:
        if not self._gts or not self._res:
            return 0.0

        score, _ = self._scorer.compute_score(self._gts, self._res)
        # 官方实现直接返回单个浮点数
        return float(score)


class CaptionMETEORMetric(BaseMetric):
    """
    图片/区域描述任务的 METEOR 指标（官方 MS-COCO 实现）。

    依赖 Java 环境（pycocoevalcap 内部会启动一个 Meteor 的 Java 进程）。
    """

    def __init__(self, name: str = "meteor", is_aux: bool = True) -> None:
        _ensure_coco_evalcap()
        from pycocoevalcap.meteor.meteor import Meteor  # type: ignore

        self._scorer = Meteor()
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self._gts: Dict[str, List[str]] = {}
        self._res: Dict[str, List[str]] = {}
        self._idx: int = 0

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

        gt_str = str(gt).strip()
        pred_str = str(pred).strip()
        if not gt_str and not pred_str:
            return

        img_id = f"{self._idx}"
        self._idx += 1

        self._gts.setdefault(img_id, []).append(gt_str)
        self._res[img_id] = [pred_str]

    def compute(self) -> float:
        if not self._gts or not self._res:
            return 0.0

        score, _ = self._scorer.compute_score(self._gts, self._res)
        return float(score)


class CaptionCIDErMetric(BaseMetric):
    """
    图片/区域描述任务的 CIDEr 指标（官方 MS-COCO 实现）。

    使用 pycocoevalcap.cider.Cider 计算 corpus-level CIDEr。
    """

    def __init__(self, name: str = "cider", is_aux: bool = False) -> None:
        _ensure_coco_evalcap()
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        self._scorer = Cider()
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self._gts: Dict[str, List[str]] = {}
        self._res: Dict[str, List[str]] = {}
        self._idx: int = 0

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

        gt_str = str(gt).strip()
        pred_str = str(pred).strip()
        if not gt_str and not pred_str:
            return

        img_id = f"{self._idx}"
        self._idx += 1

        self._gts.setdefault(img_id, []).append(gt_str)
        self._res[img_id] = [pred_str]

    def compute(self) -> float:
        if not self._gts or not self._res:
            return 0.0

        score, _ = self._scorer.compute_score(self._gts, self._res)
        return float(score)


def build_caption_metrics(task: str) -> MetricCollection:
    """
    根据任务类型构建图片/区域描述类指标集合。

    需求文档中：
        - 图片描述（简洁 / 详细）: 核心：CIDEr、ROUGE-L；辅助：BLEU-4、METEOR
        - 区域描述: 同上
    """
    metrics: List[BaseMetric] = []

    if task in {"简洁图片描述", "详细图片描述", "区域描述"}:
        # 核心指标
        metrics.append(CaptionCIDErMetric(name="cider", is_aux=False))
        metrics.append(CaptionROUGELMetric(name="rouge_l", is_aux=False))
        # 辅助指标
        metrics.append(CaptionBLEUMetric(name="bleu_4", max_n=4, is_aux=True))
        metrics.append(CaptionMETEORMetric(name="meteor", is_aux=True))

    return MetricCollection(metrics)


def calculate_caption_metrics(
    task: str,
    gt_list: Sequence[str],
    pred_list: Sequence[str],
) -> Dict[str, float]:
    """
    便捷函数：针对某个描述类任务，一次性计算全量 CIDEr / ROUGE-L / BLEU-4 / METEOR。

    Returns:
        指标结果字典，数值均为 0.xx 形式（已经在 MetricCollection 里统一保留两位小数）。
    """
    if len(gt_list) != len(pred_list):
        raise ValueError("gt_list 和 pred_list 长度不一致")

    metrics = build_caption_metrics(task)
    for gt, pred in zip(gt_list, pred_list):
        metrics.update(gt=gt, pred=pred, task=task, source=None)

    return metrics.compute_all()
