# quick_check_evaluator_raw_pairs.py
from evaluator import EvaluationCore


def test_vqa2_raw_pairs():
    """
    用 VQA2 做一个简单测试：

    我们手动构造：
        标注样本 anno_sample:
            - gt 是字符串形式的数字（比如 "3"）
        模型样本 model_sample:
            - model_output 也是字符串数字（比如 "3"）

    DataParser 会把它们解析成 int，
    然后 EvaluationCore 会按 VQA2 任务的指标来算。
    """
    core = EvaluationCore()
    task = "VQA2"

    # 造 3 条标注样本
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

    # 对应的模型输出样本
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
            "model_output": "4",  # 错 1
            "source": "img2",
        },
        {
            "sample_id": "img3",
            "task": "VQA2",
            "model_output": "2",  # 错 2
            "source": "img3",
        },
    ]

    # 手动一一配对（下一步我们会用 DataLoader.pair_samples 做这个事）
    paired_samples = list(zip(anno_samples, model_samples))

    summary = core.evaluate_raw_paired_samples(task, paired_samples)
    print("VQA2 原始样本对指标:", summary)

    # 做一点点 sanity check：有 accuracy / abs_error 这些 key 即可
    assert "accuracy" in summary
    # VQA2 我们之前设计过绝对误差指标名称，如果是 abs_error 或 mae，你可以自己再补一个断言
    print("VQA2 原始样本对评估 测试通过 ✅")


if __name__ == "__main__":
    test_vqa2_raw_pairs()
