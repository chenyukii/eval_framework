# quick_check_evaluator_streaming.py
from evaluator import EvaluationCore
from data.parser import DataParser

def build_demo_vqa2_pairs():
    """
    构造 3 条 VQA2 的 (anno_sample, model_sample) 原始样本对，
    和之前 quick_check 里的一致。
    """
    anno_samples = [
        {
            "prompt": "How many cars are there?",
            "frames": ["dummy_frame_1"],
            "gt": "3",
            "task": "VQA2",
            "source": "img1",
        },
        {
            "prompt": "How many buildings are visible?",
            "frames": ["dummy_frame_2"],
            "gt": "5",
            "task": "VQA2",
            "source": "img2",
        },
        {
            "prompt": "How many rivers are there?",
            "frames": ["dummy_frame_3"],
            "gt": "0",
            "task": "VQA2",
            "source": "img3",
        },
    ]

    model_samples = [
        {
            "sample_id": "img1",
            "task": "VQA2",
            "model_output": "3",  # 正确
            "source": "img1",
        },
        {
            "sample_id": "img2",
            "task": "VQA2",
            "model_output": "4",  # 差 1
            "source": "img2",
        },
        {
            "sample_id": "img3",
            "task": "VQA2",
            "model_output": "2",  # 差 2
            "source": "img3",
        },
    ]

    return list(zip(anno_samples, model_samples))


def main():
    core = EvaluationCore()
    task = "VQA2"
    all_pairs = build_demo_vqa2_pairs()

    # 方法一：老的“一次性评估”
    summary_batch = core.evaluate_raw_paired_samples(
        task=task,
        paired_samples=all_pairs,
        calc_aux_metric=True,
    )
    print("一次性评估结果:", summary_batch)

    # 方法二：新的“流式评估”（一次喂完）
    metrics_stream = core.build_metrics(task, calc_aux_metric=True)
    core.update_metrics_with_raw_pairs(task, metrics_stream, all_pairs)
    summary_stream = metrics_stream.compute_all()
    print("流式评估结果 (一次喂完):", summary_stream)

    # 方法三：模拟“分两批喂”，验证跨批次累积结果仍然一致
    metrics_stream2 = core.build_metrics(task, calc_aux_metric=True)
    core.update_metrics_with_raw_pairs(task, metrics_stream2, all_pairs[:2])
    core.update_metrics_with_raw_pairs(task, metrics_stream2, all_pairs[2:])
    summary_stream2 = metrics_stream2.compute_all()
    print("流式评估结果 (分批喂):", summary_stream2)

    # 简单断言：三个结果应该完全一样（compute_all 内部已经统一四舍五入）
    assert summary_batch == summary_stream
    assert summary_stream == summary_stream2

    print("流式评估接口测试通过 ✅")


if __name__ == "__main__":
    main()
