from __future__ import annotations

from typing import Sequence
import threading

try:
    # MS-COCO 官方 caption 评估实现
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    _COCO_EVALCAP_AVAILABLE = True
    _COCO_IMPORT_ERROR: Exception | None = None
except Exception as e:  # ImportError 等
    _COCO_EVALCAP_AVAILABLE = False
    _COCO_IMPORT_ERROR = e

# 为了避免多次初始化、以及线程安全问题，这里把 scorer 做成全局单例 + 锁
_bleu_scorer = Bleu(4) if _COCO_EVALCAP_AVAILABLE else None
_meteor_scorer = Meteor() if _COCO_EVALCAP_AVAILABLE else None
_rouge_scorer = Rouge() if _COCO_EVALCAP_AVAILABLE else None
_cider_scorer = Cider() if _COCO_EVALCAP_AVAILABLE else None

_bleu_lock = threading.Lock()
_meteor_lock = threading.Lock()
_rouge_lock = threading.Lock()
_cider_lock = threading.Lock()


def _ensure_coco_evalcap() -> None:
    """
    确认 pycocoevalcap 已安装，否则给出清晰报错。
    """
    if not _COCO_EVALCAP_AVAILABLE:
        msg_lines = [
            "pycocoevalcap 未安装，无法计算官方 BLEU/ROUGE/METEOR/CIDEr 指标。",
            "请在当前 Python 环境中先安装：",
            "    pip install pycocoevalcap",
        ]
        if _COCO_IMPORT_ERROR is not None:
            msg_lines.append(f"原始错误信息：{repr(_COCO_IMPORT_ERROR)}")
        raise RuntimeError("\n".join(msg_lines))


def bleu_score(
    candidate: str,
    references: Sequence[str],
    max_n: int = 4,
) -> float:
    """
    官方 MS-COCO BLEU 实现的单句封装。

    Args:
        candidate: 模型生成的 caption
        references: 参考 caption 列表
        max_n: 使用的 BLEU-n（1~4），默认返回 BLEU-4

    Returns:
        0~1 之间的分数（和 COCO 官方评估保持一致）
    """
    _ensure_coco_evalcap()
    if not isinstance(references, (list, tuple)):
        references = [references]

    img_id = "0"
    gts = {img_id: list(references)}
    res = {img_id: [candidate]}

    with _bleu_lock:
        score, _ = _bleu_scorer.compute_score(gts, res)

    # COCO BLEU 返回的是一个长度为 n 的列表：[BLEU-1, BLEU-2, ..., BLEU-n]
    if isinstance(score, (list, tuple)):
        if max_n < 1 or max_n > len(score):
            raise ValueError(
                f"max_n={max_n} 超出 BLEU 返回长度范围 [1, {len(score)}]"
            )
        return float(score[max_n - 1])

    # 某些实现/版本可能直接返回单值，这里兜底一下
    return float(score)


def rouge_l(candidate: str, references: Sequence[str]) -> float:
    """
    官方 MS-COCO ROUGE-L 实现的单句封装。

    COCO 的 Rouge 实现本质上计算的是 ROUGE-L，
    compute_score 返回 (平均得分, 每句得分列表)，我们取平均得分。
    """
    _ensure_coco_evalcap()
    if not isinstance(references, (list, tuple)):
        references = [references]

    img_id = "0"
    gts = {img_id: list(references)}
    res = {img_id: [candidate]}

    with _rouge_lock:
        score, _ = _rouge_scorer.compute_score(gts, res)

    # 官方实现返回就是单个浮点数
    return float(score)


def meteor_like(candidate: str, references: Sequence[str]) -> float:
    """
    官方 MS-COCO METEOR 实现的单句封装。

    这里叫 meteor_like 是为了和前面简化版区分，但内部实际用的就是
    pycocoevalcap 的 Meteor。
    """
    _ensure_coco_evalcap()
    if not isinstance(references, (list, tuple)):
        references = [references]

    img_id = "0"
    gts = {img_id: list(references)}
    res = {img_id: [candidate]}

    with _meteor_lock:
        score, _ = _meteor_scorer.compute_score(gts, res)

    return float(score)


def cider_like(candidate: str, references: Sequence[str]) -> float:
    """
    官方 MS-COCO CIDEr 实现的单句封装。

    返回值和 COCO 官方评估一致（大致在 0~1 之间，优秀模型通常在 1 左右）。
    """
    _ensure_coco_evalcap()
    if not isinstance(references, (list, tuple)):
        references = [references]

    img_id = "0"
    gts = {img_id: list(references)}
    res = {img_id: [candidate]}

    with _cider_lock:
        score, _ = _cider_scorer.compute_score(gts, res)

    return float(score)
