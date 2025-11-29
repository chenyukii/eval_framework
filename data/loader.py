import json
from typing import List, Dict, Tuple, Optional
import os
from .parser import DataParser
from .validator import DataValidator

class DataLoader:
    """数据加载器，负责加载和解析标注文件与模型输出文件"""

    def __init__(self, error_log_path: str = "./error_log.txt",
                 invalid_sample_log_path: str = "./invalid_sample_log.txt"):
        # 原有初始化代码...
        self.parser = DataParser()
        self.validator = DataValidator(error_log_path, invalid_sample_log_path)

    # 新增方法：加载并验证标注文件
    def load_and_validate_annotation_file(self, file_path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """加载并验证标注文件，返回有效和无效样本"""
        samples = self.load_annotation_file(file_path)
        return self.validator.batch_validate_annotations(samples)

    # 新增方法：加载并验证模型输出文件
    def load_and_validate_model_output_file(self, file_path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """加载并验证模型输出文件，返回有效和无效样本"""
        outputs = self.load_model_output_file(file_path)
        return self.validator.batch_validate_model_outputs(outputs)

    # 新增批量加载验证方法
    def batch_load_validate_annotation_dir(self, dir_path: str) -> Dict[
        str, Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
        """批量加载并验证目录下的标注文件"""
        anno_files = self.batch_load_annotation_dir(dir_path)
        result = {}
        for filename, samples in anno_files.items():
            valid, invalid = self.validator.batch_validate_annotations(samples)
            result[filename] = (valid, invalid)
        return result

    def batch_load_validate_model_output_dir(self, dir_path: str) -> Dict[
        str, Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
        """批量加载并验证目录下的模型输出文件"""
        model_files = self.batch_load_model_output_dir(dir_path)
        result = {}
        for filename, outputs in model_files.items():
            valid, invalid = self.validator.batch_validate_model_outputs(outputs)
            result[filename] = (valid, invalid)
        return result


    def load_annotation_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        加载标注文件（TXT-JSON格式）

        Args:
            file_path: 标注文件路径

        Returns:
            样本列表，每个样本是包含prompt、frames、gt、task、source的字典
        """
        samples = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"标注文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        sample = json.loads(line)
                        # 验证必要字段
                        required_fields = ['prompt', 'frames', 'gt', 'task', 'source']
                        for field in required_fields:
                            if field not in sample:
                                raise ValueError(f"缺少必要字段: {field} (行号: {line_num})")
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误 (行号: {line_num}): {str(e)}")
                    except ValueError as e:
                        print(f"数据验证错误 (行号: {line_num}): {str(e)}")
        except IOError as e:
            raise RuntimeError(f"读取标注文件失败: {str(e)}")

        return samples

    def load_model_output_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        加载模型输出文件（TXT或JSON格式）

        Args:
            file_path: 模型输出文件路径

        Returns:
            模型输出列表，每个元素是包含sample_id、task、model_output、source的字典
        """
        outputs = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型输出文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 判断文件格式（简单通过扩展名判断）
                if file_path.endswith('.json'):
                    # JSON格式：数组结构
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            raise ValueError("JSON格式模型输出文件必须是数组结构")
                        outputs = data
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"JSON文件解析错误: {str(e)}")
                else:
                    # TXT格式：每行一个JSON
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            output = json.loads(line)
                            outputs.append(output)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误 (行号: {line_num}): {str(e)}")

            # 验证所有输出的必要字段
            required_fields = ['sample_id', 'task', 'model_output', 'source']
            for idx, output in enumerate(outputs):
                for field in required_fields:
                    if field not in output:
                        raise ValueError(f"模型输出缺少必要字段: {field} (索引: {idx})")

            return outputs

        except IOError as e:
            raise RuntimeError(f"读取模型输出文件失败: {str(e)}")

    def pair_samples(
            self,
            anno_samples: List[Dict[str, str]],
            model_samples: List[Dict[str, str]]
    ) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
        """
        配对标注样本与模型输出样本

        配对规则：
        1. 优先通过 source 字段精确匹配
        2. 若 source 匹配失败，则通过 prompt 字段辅助匹配（如果模型输出中存在 prompt）
        3. 确保每个模型样本只被配对一次
        """
        paired: List[Tuple[Dict[str, str], Dict[str, str]]] = []
        unpaired_anno: List[Dict[str, str]] = []

        # 记录已经被使用过的模型样本索引
        used_model_indices = set()

        # 构建 source -> [模型样本索引列表] 的映射
        model_source_map: Dict[str, List[int]] = {}
        for idx, model_sample in enumerate(model_samples):
            source = model_sample.get("source", "")
            model_source_map.setdefault(source, []).append(idx)

        # 遍历标注样本进行配对
        for anno in anno_samples:
            anno_source = anno.get("source", "")
            matched_idx: Optional[int] = None

            # 1) 尝试通过 source 匹配
            if anno_source in model_source_map:
                indices = model_source_map[anno_source]
                # 取第一个还没用过的索引
                while indices:
                    candidate_idx = indices.pop(0)
                    if candidate_idx not in used_model_indices:
                        matched_idx = candidate_idx
                        break

            # 2) 如果 source 匹配失败，再尝试通过 prompt 匹配
            if matched_idx is None:
                anno_prompt = anno.get("prompt", "")
                if anno_prompt:
                    for idx, model_sample in enumerate(model_samples):
                        if idx in used_model_indices:
                            continue
                        if model_sample.get("prompt", "") == anno_prompt:
                            matched_idx = idx
                            break

            # 3) 根据匹配结果记录
            if matched_idx is not None:
                used_model_indices.add(matched_idx)
                paired.append((anno, model_samples[matched_idx]))
            else:
                unpaired_anno.append(anno)

        # 统计未匹配的模型输出样本
        unpaired_model = [
            model_samples[idx]
            for idx in range(len(model_samples))
            if idx not in used_model_indices
        ]

        print(
            f"样本配对完成 - 成功: {len(paired)}, "
            f"未匹配标注样本: {len(unpaired_anno)}, "
            f"未匹配模型输出: {len(unpaired_model)}"
        )

        if unpaired_anno or unpaired_model:
            print("警告: 存在未匹配样本，可能影响评估结果准确性")

        return paired

    def batch_pair_samples(self, anno_files: Dict[str, List[Dict[str, str]]],
                           model_output_files: Dict[str, List[Dict[str, str]]]) -> Dict[
        str, List[Tuple[Dict[str, str], Dict[str, str]]]]:
        """
        批量配对标注文件与模型输出文件中的样本

        配对规则：基于文件名关联（如"test.txt"对应"test_output.txt"）

        Args:
            anno_files: 批量加载的标注文件字典 {文件名: 样本列表}
            model_output_files: 批量加载的模型输出文件字典 {文件名: 输出列表}

        Returns:
            配对成功的文件-样本字典 {文件名对: 配对样本列表}
        """
        paired_files = {}

        # 遍历标注文件，寻找对应的模型输出文件
        for anno_filename, anno_samples in anno_files.items():
            # 构建可能的模型输出文件名（根据需求文档的命名规则）
            base_name = os.path.splitext(anno_filename)[0]
            possible_output_names = [
                f"{base_name}_output.txt",
                f"{base_name}_output.json",
                f"{base_name}.txt",  # 备选匹配规则
                f"{base_name}.json"
            ]

            # 查找匹配的模型输出文件
            matched_output_name = None
            for output_name in possible_output_names:
                if output_name in model_output_files:
                    matched_output_name = output_name
                    break

            # 如果找到匹配的文件，则进行样本配对
            if matched_output_name:
                model_samples = model_output_files[matched_output_name]
                paired_samples = self.pair_samples(anno_samples, model_samples)
                if paired_samples:
                    paired_files[f"{anno_filename}->{matched_output_name}"] = paired_samples
                    print(f"文件配对成功: {anno_filename} 与 {matched_output_name}, 样本对数: {len(paired_samples)}")
            else:
                print(f"未找到 {anno_filename} 对应的模型输出文件")

        return paired_files

    def batch_load_annotation_dir(self, dir_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        批量加载目录下的所有标注文件

        Args:
            dir_path: 标注文件目录路径

        Returns:
            字典，键为文件名，值为该文件加载的样本列表
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"标注文件目录不存在: {dir_path}")

        annotation_files = {}
        # 遍历目录下所有txt文件（标注文件默认为txt格式）
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    try:
                        # 调用单文件加载方法
                        samples = self.load_annotation_file(file_path)
                        if samples:  # 只保留有样本的文件
                            annotation_files[filename] = samples
                            print(f"成功加载标注文件: {filename}，样本数: {len(samples)}")
                    except Exception as e:
                        print(f"加载标注文件 {filename} 失败: {str(e)}")

        return annotation_files

    def batch_load_model_output_dir(self, dir_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        批量加载目录下的所有模型输出文件

        Args:
            dir_path: 模型输出文件目录路径

        Returns:
            字典，键为文件名，值为该文件加载的模型输出列表
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"模型输出文件目录不存在: {dir_path}")

        model_output_files = {}
        # 遍历目录下所有txt和json文件（模型输出支持这两种格式）
        for filename in os.listdir(dir_path):
            if filename.endswith(('.txt', '.json')):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    try:
                        # 调用单文件加载方法
                        outputs = self.load_model_output_file(file_path)
                        if outputs:  # 只保留有输出的文件
                            model_output_files[filename] = outputs
                            print(f"成功加载模型输出文件: {filename}，输出数: {len(outputs)}")
                    except Exception as e:
                        print(f"加载模型输出文件 {filename} 失败: {str(e)}")

        return model_output_files