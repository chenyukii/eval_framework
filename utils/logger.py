"""
统一日志与结构化错误日志工具。

功能：
- configure_logging：初始化标准日志（控制台 + 可选文件），带时间戳/级别/文件行号。
- set_structured_log_path：设置结构化错误日志（JSON Lines）输出路径。
- log_struct：写结构化错误日志，并按级别输出到标准日志。
- get_logger：获取模块级 logger。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from typing import Any, Dict, Optional

# 默认结构化错误日志路径
_STRUCTURED_LOG_PATH = os.path.join(".", "logs", "error_structured.jsonl")
_STRUCT_LOCK = threading.Lock()


def _ensure_parent_dir(path: str) -> None:
    """确保路径的上级目录存在。"""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    初始化基础日志配置。应在程序入口调用一次。

    Args:
        level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 可选，文件路径；提供则输出到文件+控制台
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        _ensure_parent_dir(log_file)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d %(message)s",
        handlers=handlers,
    )


def set_structured_log_path(path: str) -> None:
    """
    设置结构化错误日志（JSONL）的输出路径。
    """
    global _STRUCTURED_LOG_PATH
    _ensure_parent_dir(path)
    _STRUCTURED_LOG_PATH = path


def log_struct(
    stage: str,
    error_code: str,
    message: str,
    *,
    level: str = "ERROR",
    logger: Optional[logging.Logger] = None,
    **fields: Any,
) -> None:
    """
    写一行结构化错误日志（JSONL），并按级别输出到标准日志。

    Args:
        stage: 所在阶段，如 load/validate/parse/match/eval
        error_code: 错误类别编码，便于后期统计
        message: 描述
        level: 日志级别字符串
        logger: 可选，提供则同时调用 logger 输出
        **fields: 其他上下文字段（task/source/sample_id/file 等）
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level.upper(),
        "stage": stage,
        "error_code": error_code,
        "message": message,
    }
    record.update(fields)

    # 写入 JSONL
    try:
        with _STRUCT_LOCK:
            with open(_STRUCTURED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 结构化日志写入失败不应中断主流程
        pass

    # 同步到普通日志
    if logger:
        log_fn = getattr(logger, level.lower(), logger.error)
        log_fn(f"{error_code}: {message} | ctx={fields}")


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(name)
