# metrics/detection.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMetric, MetricCollection
from ..utils.iou import bbox_iou, quad_iou


BBox = Tuple[float, float, float, float]
Quad = Tuple[float, float, float, float, float, float, float, float]


class DetectionAPMetric(BaseMetric):
    """
    通用检测 AP 指标（单类），支持：

        - 水平区域检测（水平框）
        - 旋转区域检测（旋转框）
        - VQA3 边界框问答（水平框）
        - 视觉定位（水平框）

    设计思路：
        - 每条样本在 update() 阶段只做“收集”：
            - 根据 task 切出 gt / pred 中的 bbox / quad 列表
            - 用 source 作为 image_id，将所有 box 存起来
        - 在 compute() 阶段统一根据 IoU 阈值做匹配，计算 AP。
        - 置信度目前需求没有提供，先全部当作 score=1.0，
          相当于只在一个点上的 P-R 曲线，AP 退化为该点处的面积。
          后续如果模型输出包含 score，可以很容易扩展。
    """

    def __init__(
        self,
        name: str,
        iou_threshold: float,
        box_mode: str = "bbox",  # "bbox" 或 "quad"
        is_aux: bool = False,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.box_mode = box_mode

        if box_mode == "bbox":
            self._iou_fn = bbox_iou
        elif box_mode == "quad":
            self._iou_fn = quad_iou
        else:
            raise ValueError(f"未知 box_mode: {box_mode}")

        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        # image_id -> gt 框列表
        self.gts: Dict[str, List[Any]] = {}
        # image_id -> 预测框列表（当前不区分类别，单类检测）
        self.preds: Dict[str, List[Any]] = {}

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
        根据任务类型，从解析后的 gt / pred 中抽取出 bbox / quad 列表。

        约定：
            - 水平 / 旋转区域检测：
                gt / pred: (count, box_list)
            - VQA3 / 视觉定位：
                gt / pred: box_list
        """
        if source is None:
            # 没有 image_id 无法做匹配，直接跳过
            return

        # 1) 按任务类型取出框列表
        if task in {"水平区域检测", "旋转区域检测"}:
            gt_boxes: List[Any] = []
            if gt is not None:
                if isinstance(gt, tuple) and len(gt) == 2:
                    _, gt_boxes = gt
                elif isinstance(gt, list):
                    gt_boxes = gt

            pred_boxes: List[Any] = []
            if pred is not None:
                if isinstance(pred, tuple) and len(pred) == 2:
                    _, pred_boxes = pred
                elif isinstance(pred, list):
                    pred_boxes = pred

        elif task in {"VQA3", "视觉定位"}:
            gt_boxes = gt if isinstance(gt, list) else (gt or [])
            pred_boxes = pred if isinstance(pred, list) else (pred or [])

        else:
            # 非检测类任务，直接忽略
            return

        if gt_boxes is None:
            gt_boxes = []
        if pred_boxes is None:
            pred_boxes = []

        # 2) 将同一 source 下的所有框累积起来
        self.gts.setdefault(source, []).extend(gt_boxes)
        self.preds.setdefault(source, []).extend(pred_boxes)

    def compute(self) -> float:
        """
        计算在当前 IoU 阈值下的 AP。

        实现步骤：
            1. 展开所有图像的 gt 框，统计正样本总数 npos；
            2. 展开所有预测框（目前 score=1.0 占位）；
            3. 遍历每个预测框，在对应图像里找 IoU 最大且尚未匹配的 gt：
                - 若 IoU >= 阈值：TP
                - 否则：FP
            4. 用累计 TP / FP 计算 P-R 曲线，并按照 VOC / COCO 风格计算 AP。
        """
        # 1) 真实目标总数
        npos = sum(len(boxes) for boxes in self.gts.values())

        # 2) 展开所有预测
        # 元素结构： (image_id, box, score)
        detections: List[Tuple[str, Any, float]] = []
        for image_id, boxes in self.preds.items():
            for b in boxes:
                detections.append((image_id, b, 1.0))  # 暂时统一 score=1.0

        if npos == 0 or not detections:
            return 0.0

        # 如果将来有置信度，可以在这里按 score 排序
        # detections.sort(key=lambda x: x[2], reverse=True)

        # 3) 为每个图像准备 gt 匹配标记
        gt_matched: Dict[str, List[bool]] = {
            img_id: [False] * len(boxes)
            for img_id, boxes in self.gts.items()
        }

        tp: List[int] = []
        fp: List[int] = []

        for image_id, pred_box, score in detections:
            gt_boxes = self.gts.get(image_id, [])
            if image_id not in gt_matched:
                gt_matched[image_id] = [False] * len(gt_boxes)

            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if gt_matched[image_id][idx]:
                    continue
                iou = self._iou_fn(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                gt_matched[image_id][best_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # 4) 累积 TP / FP，计算 P-R 曲线
        tp_cum: List[int] = []
        fp_cum: List[int] = []
        c_tp = 0
        c_fp = 0
        for t, f in zip(tp, fp):
            c_tp += t
            c_fp += f
            tp_cum.append(c_tp)
            fp_cum.append(c_fp)

        recalls: List[float] = []
        precisions: List[float] = []
        for t_c, f_c in zip(tp_cum, fp_cum):
            recall = t_c / npos
            precision = t_c / max(t_c + f_c, 1e-8)
            recalls.append(recall)
            precisions.append(precision)

        return self._compute_ap(recalls, precisions)

    @staticmethod
    def _compute_ap(recalls: List[float], precisions: List[float]) -> float:
        """
        按 VOC / COCO 风格计算 AP：
            1. 对 precision 做“后向包络”（保证随着 recall 增大 precision 非增）
            2. 对 P-R 曲线做积分求面积
        """
        if not recalls:
            return 0.0

        # 在首尾加上 (0,0) 和 (1,0) 方便积分
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]

        # 让 precision 变成单调不增
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        return ap


def build_detection_metrics(task: str) -> MetricCollection:
    """
    根据任务类型构建检测相关的指标集合。

    对应关系（来自需求文档）：:contentReference[oaicite:5]{index=5}
        - 水平 / 旋转区域检测：
            - 核心：AP@IoU=0.5
            - 辅助：AP@IoU=0.75
        - VQA3（边界框问答）：
            - 核心：AP@IoU=0.5
            - 辅助：AP@IoU=0.75
        - 视觉定位：
            - 核心：AP@IoU=0.5
            - 辅助：AP@IoU=0.25
    """
    metrics: List[BaseMetric] = []

    if task == "水平区域检测":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="bbox",
                is_aux=True,
            )
        )

    elif task == "旋转区域检测":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="quad",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="quad",
                is_aux=True,
            )
        )

    elif task == "VQA3":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="bbox",
                is_aux=True,
            )
        )

    elif task == "视觉定位":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_25",
                iou_threshold=0.25,
                box_mode="bbox",
                is_aux=True,
            )
        )

    # 对于非检测任务，返回空集合，上层不该调用这里

    return MetricCollection(metrics)
