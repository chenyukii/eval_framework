from data.parser import DataParser
from data.validator import DataValidator
from data.loader import DataLoader

def test_parser_detection():
    parser = DataParser()
    s1 = "2<box><1><2><3><4></box><box><5><6><7><8></box>"
    count, boxes = parser.parse_gt("水平区域检测", s1)
    print("水平区域检测 无空格:", count, boxes)

    s2 = "2 <box><1><2><3><4></box><box><5><6><7><8></box>"
    count2, boxes2 = parser.parse_gt("水平区域检测", s2)
    print("水平区域检测 有空格:", count2, boxes2)

def test_validator_logs():
    validator = DataValidator(
        error_log_path="./tmp_error_log.txt",
        invalid_sample_log_path="./tmp_invalid_sample_log.txt"
    )
    # 构造一个缺少字段的标注样本，触发 invalid_sample_log
    sample = {
        "prompt": "",
        "frames": "",
        "gt": "",
        "task": "图片分类",
        "source": "test_image.png"
    }
    ok, msg = validator.validate_annotation_sample(sample)
    print("标注样本是否有效:", ok, "错误信息:", msg)

def test_pair_samples():
    loader = DataLoader(
        error_log_path="./tmp_error_log.txt",
        invalid_sample_log_path="./tmp_invalid_sample_log.txt"
    )
    anno_samples = [
        {"source": "a.png", "prompt": "p1", "gt": "Yes", "task": "VQA1", "frames": "xxx"},
        {"source": "b.png", "prompt": "p2", "gt": "No", "task": "VQA1", "frames": "yyy"},
    ]
    model_samples = [
        {"source": "a.png", "prompt": "p1", "sample_id": "1", "task": "VQA1", "model_output": "Yes"},
        {"source": "b.png", "prompt": "p2", "sample_id": "2", "task": "VQA1", "model_output": "No"},
    ]
    paired = loader.pair_samples(anno_samples, model_samples)
    print("配对结果数量:", len(paired))
    print("配对的 sample_id:", [m["sample_id"] for _, m in paired])

if __name__ == "__main__":
    test_parser_detection()
    test_validator_logs()
    test_pair_samples()
