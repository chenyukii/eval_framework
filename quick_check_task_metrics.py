from metrics import (
    build_task_metrics,
    CLASSIFICATION_TASKS,
    DETECTION_TASKS,
    RETRIEVAL_TASKS,
    CAPTION_TASKS,
)


def inspect_task(task: str):
    metrics = build_task_metrics(task)
    summary = metrics.compute_all()
    print(f"任务：{task}")
    print("  指标 keys:", list(summary.keys()))


if __name__ == "__main__":
    # 1. 随机挑几个分类 / 数量任务
    for t in ["图片分类", "VQA2", "计数"]:
        inspect_task(t)

    # 2. 检测任务
    for t in ["水平区域检测", "旋转区域检测", "VQA3", "视觉定位"]:
        inspect_task(t)

    # 3. 图片检索
    inspect_task("图片检索")

    # 4. 文本描述
    for t in ["简洁图片描述", "详细图片描述", "区域描述"]:
        inspect_task(t)
