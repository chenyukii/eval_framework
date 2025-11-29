"""
日志工具模块
负责记录系统运行日志、错误日志、无效样本日志
"""
import os
from loguru import logger


def init_logger(output_dir: str = "./logs"):
    """
    初始化日志配置

    参数:
        output_dir: 日志保存目录
    """
    # 创建日志目录
    os.makedirs(output_dir, exist_ok=True)

    # 移除默认控制台输出
    logger.remove()

    # 添加运行日志（INFO级别，按天滚动）
    logger.add(
        os.path.join(output_dir, "run.log"),
        level="INFO",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8"
    )

    # 添加错误日志（ERROR级别）
    logger.add(
        os.path.join(output_dir, "error.log"),
        level="ERROR",
        rotation="10 MB",
        encoding="utf-8"
    )

    # 添加控制台输出
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO"
    )

    return logger


# 初始化默认日志
logger = init_logger()