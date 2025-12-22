# metrics/segmentation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

from .base import BaseMetric, MetricCollection
from .classification import build_classification_metrics

Polygon = Sequence[Tuple[float, float]]


# ================= 基础工具函数 =================
def _to_polygon_list(value: Any) -> List[Polygon]:
    """
    将解析后的多边形/点结果规整为“多边形列表”。
    - 输入：解析后的 value，通常来自 DataParser 的输出，可能是 list/tuple。
    - 规则：元素必须是 list/tuple 且长度 >= 1 才被视作一个 poly；其它情况丢弃。
    - 输出：List[Polygon]，列表中的每个元素代表一个 <poly> 集合。
    """
    if value is None:
        return []
    if isinstance(value, list):
        result: List[Polygon] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                result.append(tuple(item))
        return result
    return []


def _poly_to_point_set(poly: Polygon) -> Set[Tuple[float, float]]:
    """
    把单个 polygon 转为点集合，方便后续集合运算（交/并等）。
    """
    return {tuple(pt) for pt in poly}


def _point_set_iou(a: Polygon, b: Polygon) -> float:
    """
    基于点集合的 IoU（交并比）：
        IoU = |A ∩ B| / |A ∪ B|
    - 若并集为空（极端情况），返回 0.0。
    """
    set_a = _poly_to_point_set(a)
    set_b = _poly_to_point_set(b)
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _greedy_match_polygons(
    gt_polys: List[Polygon],
    pred_polys: List[Polygon],
    iou_thresh: float,
) -> Tuple[int, int, int]:
    """
    逐 polygon 进行贪心匹配，统计 TP/FP/FN：
        1) 遍历每个 pred，多边形与所有“尚未匹配”的 gt 计算 IoU（基于点集合）。
        2) 找到 IoU 最高的 gt，若 IoU >= 阈值，则记为 TP，并标记该 gt 已匹配。
           否则该 pred 记为 FP。
        3) 所有未被匹配的 gt 计为 FN。
    返回: (tp, fp, fn)
    - 逻辑类似检测中的框匹配，但相似度是“点集合 IoU”。
    """
    matched_gt = set()
    tp = fp = 0

    for pred in pred_polys:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_polys):
            if idx in matched_gt:
                continue
            iou = _point_set_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_polys) - len(matched_gt)
    return tp, fp, fn


# ================= 点级指标（原有逻辑，保留） =================
class PointSetPrecision(BaseMetric):
    """
    点集合精度：把所有 <poly> 的点合并为一个大集合，
    precision = |GT∩Pred| / |Pred|。
    """
    def reset(self) -> None:
        self.values: List[float] = []

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_pts = {tuple(pt) for poly in _to_polygon_list(gt) for pt in poly}
        pred_pts = {tuple(pt) for poly in _to_polygon_list(pred) for pt in poly}
        # 双空样本不计入
        if not gt_pts and not pred_pts:
            return
        if not pred_pts:
            self.values.append(0.0)
            return
        inter = len(gt_pts & pred_pts)
        self.values.append(inter / len(pred_pts))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class PointSetRecall(BaseMetric):
    """
    点集合召回：把所有点合并后，
    recall = |GT∩Pred| / |GT|。
    """
    def reset(self) -> None:
        self.values: List[float] = []

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_pts = {tuple(pt) for poly in _to_polygon_list(gt) for pt in poly}
        pred_pts = {tuple(pt) for poly in _to_polygon_list(pred) for pt in poly}
        if not gt_pts and not pred_pts:
            return
        if not gt_pts:
            self.values.append(0.0)
            return
        inter = len(gt_pts & pred_pts)
        self.values.append(inter / len(gt_pts))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class PointSetF1(BaseMetric):
    """
    点集合 F1：把所有点合并后，
    F1 = 2 * |GT∩Pred| / (|GT| + |Pred|)。
    """
    def reset(self) -> None:
        self.values: List[float] = []

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_pts = {tuple(pt) for poly in _to_polygon_list(gt) for pt in poly}
        pred_pts = {tuple(pt) for poly in _to_polygon_list(pred) for pt in poly}
        if not gt_pts and not pred_pts:
            return
        denom = len(gt_pts) + len(pred_pts)
        if denom == 0:
            self.values.append(0.0)
            return
        inter = len(gt_pts & pred_pts)
        self.values.append(2 * inter / denom)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


# ================= 数量辅助指标（原有逻辑，保留） =================
class CountAccuracy(BaseMetric):
    """数量准确率：len(pred_polys) == len(gt_polys) 记为正确。"""
    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        self.total += 1
        if len(gt_polys) == len(pred_polys):
            self.correct += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class CountMAE(BaseMetric):
    """数量 MAE：|len(pred_polys) - len(gt_polys)| 的平均值。"""
    def reset(self) -> None:
        self.errors: List[float] = []

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        self.errors.append(abs(len(pred_polys) - len(gt_polys)))

    def compute(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)


# ================= 新增：集合级匹配指标（逐 poly 匹配） =================
class PolyMatchPrecision(BaseMetric):
    """
    Polygon 集合级精度：
        - 把每个 <poly> 视为一个“集合”，基于点集合 IoU 做贪心匹配。
        - IoU >= 阈值 视为命中（TP），否则为 FP。
        - Precision = TP / (TP + FP)，跨样本累积 TP/FP 后统一计算。
    """
    def __init__(self, name: str = "poly_precision", iou_thresh: float = 0.5, is_aux: bool = False) -> None:
        self.iou_thresh = iou_thresh
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        # 双空样本不计入
        if not gt_polys and not pred_polys:
            return
        tp, fp, _ = _greedy_match_polygons(gt_polys, pred_polys, self.iou_thresh)
        self.tp += tp
        self.fp += fp

    def compute(self) -> float:
        denom = self.tp + self.fp
        if denom == 0:
            return 0.0
        return self.tp / denom


class PolyMatchRecall(BaseMetric):
    """
    Polygon 集合级召回：
        - 同一套匹配逻辑，召回 = TP / (TP + FN)，跨样本累积 TP/FN。
    """
    def __init__(self, name: str = "poly_recall", iou_thresh: float = 0.5, is_aux: bool = False) -> None:
        self.iou_thresh = iou_thresh
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.tp = 0
        self.fn = 0

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        tp, _, fn = _greedy_match_polygons(gt_polys, pred_polys, self.iou_thresh)
        self.tp += tp
        self.fn += fn

    def compute(self) -> float:
        denom = self.tp + self.fn
        if denom == 0:
            return 0.0
        return self.tp / denom


class PolyMatchF1(BaseMetric):
    """
    Polygon 集合级 F1：
        - 基于同一套 TP/FP/FN 计数，F1 = 2TP / (2TP + FP + FN)。
        - 适合衡量“多集合匹配”综合表现。
    """
    def __init__(self, name: str = "poly_f1", iou_thresh: float = 0.5, is_aux: bool = False) -> None:
        self.iou_thresh = iou_thresh
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, *, gt: Any, pred: Any, task: Optional[str] = None,
               source: Optional[str] = None, **kwargs: Any) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        tp, fp, fn = _greedy_match_polygons(gt_polys, pred_polys, self.iou_thresh)
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return 0.0
        return 2 * self.tp / denom


# ================= 指标构建入口 =================
def build_segmentation_metrics(task: str) -> MetricCollection:
    """
    像素级任务指标集合构建：
        - 语义分割 / 实例分割 / 变化检测：
            * 保留原有的“点级” Precision/Recall/F1（把所有点合并为一个集合）
            * 保留原有数量辅助指标 count_acc / count_mae
            * 新增“集合级”匹配指标 poly_precision / poly_recall / poly_f1（逐 poly 贪心匹配，点集 IoU）
        - 像素分类 / 细粒度识别：
            * 复用分类任务的指标
    """
    task = task or ""

    # 分类型像素任务复用分类指标
    if task in {"像素分类", "细粒度识别"}:
        return build_classification_metrics("图片分类")

    metrics: List[BaseMetric] = []

    # 语义分割 / 实例分割 / 变化检测：点级 + 数量 + 集合级匹配
    if task in {"语义分割", "变化检测", "实例分割"}:
        # 点集合指标（原有逻辑）
        metrics.append(PointSetPrecision(name="point_precision", is_aux=True))
        metrics.append(PointSetRecall(name="point_recall", is_aux=True))
        metrics.append(PointSetF1(name="point_f1", is_aux=False))
        metrics.append(CountAccuracy(name="count_acc", is_aux=True))
        metrics.append(CountMAE(name="count_mae", is_aux=True))
        # 集合级匹配指标（新增，可按需调整 IoU 阈值）
        metrics.append(PolyMatchPrecision(name="poly_precision", iou_thresh=0.5, is_aux=True))
        metrics.append(PolyMatchRecall(name="poly_recall", iou_thresh=0.5, is_aux=True))
        metrics.append(PolyMatchF1(name="poly_f1", iou_thresh=0.5, is_aux=False))

    return MetricCollection(metrics)
