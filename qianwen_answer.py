"""
Batch inference script for Qwen; produces eval-ready per-file outputs.
- Input: anno_dir/*.txt, each line JSON with prompt/frames/gt/task/source, optional sample_id.
- Output: out_dir same-named txt; each line JSON with sample_id/task/model_output/source, optional raw_model_output.
- Tasks (must be passed exactly as below):
  Classification/Counting/VQA: 图像分类, 水平区域分类, 旋转区域分类, VQA1, VQA2, 计数
  Detection/Loc: 水平区域检测, 旋转区域检测, VQA3, 视觉定位
  Retrieval: 图片检索
  Caption: 简洁图片描述, 详细图片描述, 区域描述
  Pixel-level: 像素分类, 语义分割, 实例分割, 变化检测
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# --------- 任务集合（仅接受规范名，不做归一化） ---------
CLASSIFICATION_TASKS = {
    "图像分类",
    "水平区域分类",
    "旋转区域分类",
    "VQA1",
    "VQA2",
    "计数",
    "像素分类",
}

DETECTION_TASKS = {
    "水平区域检测",
    "旋转区域检测",
    "VQA3",
    "视觉定位",
}

RETRIEVAL_TASKS = {
    "图片检索",
}

CAPTION_TASKS = {
    "简洁图片描述",
    "详细图片描述",
    "区域描述",
}

PIXEL_POLYGON_TASKS = {
    "语义分割",
    "实例分割",
    "变化检测",
}

PIXEL_TASKS = {"像素分类"} | PIXEL_POLYGON_TASKS

SUPPORTED_TASKS = (
    CLASSIFICATION_TASKS
    | DETECTION_TASKS
    | RETRIEVAL_TASKS
    | CAPTION_TASKS
    | PIXEL_TASKS
)

# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference for Qwen, outputs eval-ready files.")
    p.add_argument("--task", required=True, help="Task name (must be one of SUPPORTED_TASKS).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--anno-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--batch-size", type=int, default=1, help="Reserved; current code uses single-image inference.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--resize", type=int, default=768, help="Max long edge; 0 means no resize.")
    p.add_argument("--sar-enhance", action="store_true", help="Enhance SAR/gray images by contrast.")
    p.add_argument("--n-prompts", type=int, default=1, help="Multi-prompt voting; 1 = single prompt.")
    p.add_argument("--save-raw", action="store_true", help="Keep raw_model_output for debugging.")
    return p.parse_args()

# ---------------------- IO ----------------------
def list_txt_files(folder: str, pattern: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(folder, pattern)))

def load_lines(path: str) -> List[Dict[str, Any]]:
    """
    Read a txt file line by line; each line should be a JSON object. Invalid lines are skipped.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_sample_id(sample: Dict[str, Any], filename: str, idx: int) -> str:
    """
    Prefer sample['sample_id']; otherwise use filename stem plus line index.
    """
    if "sample_id" in sample and sample["sample_id"]:
        return str(sample["sample_id"])
    stem = os.path.splitext(os.path.basename(filename))[0]
    return f"{stem}__{idx}"  # disambiguate multi-line samples per file

# ---------------------- 图像解码 ----------------------
def decode_image(frame_data: Any, resize: int, sar_enhance: bool) -> Optional[Image.Image]:
    """
    Decode base64 image (string or first element of list). Optionally enhance SAR/grayscale and resize.
    """
    if frame_data is None:
        return None
    if isinstance(frame_data, list) and frame_data:
        frame_data = frame_data[0]
    if not isinstance(frame_data, str):
        return None
    b64 = re.sub(r"\s+", "", frame_data)
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
    except Exception:
        return None
    if sar_enhance and img.mode in ["L", "I", "F", "P"]:
        img = ImageEnhance.Contrast(img).enhance(2.0).convert("RGB")
    if resize and (img.size[0] > resize or img.size[1] > resize):
        ratio = min(resize / img.size[0], resize / img.size[1])
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ---------------------- Prompt ----------------------
def build_prompts(task: str, user_prompt: str) -> List[str]:
    """
    Return prompts for the given task. Fallback to user_prompt if task not covered.
    """
    if task in {"VQA2", "计数"}:
        return ["Count the objects in the image. Only output a number."]
    if task == "VQA1":
        return ["Answer the question with only Yes or No."]
    if task in CLASSIFICATION_TASKS:
        return ["List the categories present in the image. Use semicolons to separate. No extra text."]
    if task in RETRIEVAL_TASKS:
        return ["Return the indexes of images that match the query. Use commas or semicolons to separate numbers."]
    if task in CAPTION_TASKS:
        return ["Provide a concise English description of the image in 15-30 words. No list, no numbering."]
    if task in DETECTION_TASKS:
        return ["Output all boxes in format <box><x1><y1><x2><y2></box> with no spaces or text."]
    if task in PIXEL_POLYGON_TASKS:
        return ["Output polygons in format <poly><x1><y1><x2><y2>...</poly> with no spaces or text."]
    return [user_prompt]

def make_messages(prompt_text: str) -> List[Dict[str, Any]]:
    """
    Build chat messages with a system prompt and a user message containing image + text.
    """
    return [
        {"role": "system", "content": "You are a concise vision-language assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]

# ---------------------- 后处理 ----------------------
def postprocess(task: str, raw: str) -> str:
    """
    Normalize raw model output to eval-required format. Return 'invalid' if cannot parse.
    """
    txt = raw.strip()
    if task in {"VQA2", "计数"}:
        m = re.search(r"(-?\d+)", txt)
        return m.group(1) if m else "invalid"
    if task == "VQA1":
        head = txt.lower().strip().split()[0] if txt else ""
        if head in {"yes", "y", "yeah", "true", "affirmative"}:
            return "Yes"
        if head in {"no", "n", "false", "negative"}:
            return "No"
        return "invalid"
    if task in CLASSIFICATION_TASKS:
        parts = [p.strip().lower() for p in txt.replace("\n", " ").split(";")]
        parts = [p for p in parts if p]
        return ";".join(sorted(set(parts))) if parts else "invalid"
    if task in RETRIEVAL_TASKS:
        nums = re.findall(r"\d+", txt)
        return ";".join(nums) if nums else "invalid"
    if task in CAPTION_TASKS:
        one_line = " ".join(txt.split())
        words = one_line.split()
        if len(words) > 50:
            one_line = " ".join(words[:50])
        return one_line if one_line else "invalid"
    if task in DETECTION_TASKS:
        boxes = re.findall(r"<box>.*?</box>", txt.replace(" ", ""))
        return "".join(boxes) if boxes else "invalid"
    if task in PIXEL_POLYGON_TASKS:
        polys = re.findall(r"<poly>.*?</poly>", txt.replace(" ", ""))
        if polys:
            return "".join(polys)
        nums = re.findall(r"-?\d+", txt)
        if len(nums) >= 6 and len(nums) % 2 == 0:
            return "<poly>" + "".join(f"<{nums[i]}><{nums[i+1]}>" for i in range(0, len(nums), 2)) + "</poly>"
        return "invalid"
    return txt or "invalid"

# ---------------------- 推理 ----------------------
def generate_single(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    gen_cfg: Dict[str, Any],
) -> str:
    """
    Run single-image, single-prompt generation and decode text output.
    """
    msgs = make_messages(prompt)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    eos_id = processor.tokenizer.eos_token_id
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["temperature"] > 0,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            num_beams=gen_cfg["num_beams"],
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    return processor.tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

# ---------------------- 主流程 ----------------------
def process_file(
    path: str,
    task: str,
    args: argparse.Namespace,
    model,
    processor,
) -> List[Dict[str, Any]]:
    """
    Process one annotation file: read samples, run inference, postprocess, collect records.
    """
    samples = load_lines(path)
    if not samples:
        return []
    records = []
    for idx, sample in enumerate(samples):
        sid = get_sample_id(sample, path, idx)
        prompt = sample.get("prompt", "")
        frames = sample.get("frames")
        img = decode_image(frames, args.resize, args.sar_enhance)
        if img is None:
            records.append(
                {"sample_id": sid, "task": task, "model_output": "invalid", "source": sample.get("source", "")}
            )
            continue
        prompts = build_prompts(task, prompt)
        outputs = []
        for p in prompts[: max(args.n_prompts, 1)]:
            out_text = generate_single(
                model,
                processor,
                img,
                p,
                gen_cfg={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_beams": args.num_beams,
                },
            )
            outputs.append(out_text)
        final_raw = outputs[0]
        for cand in outputs:
            if postprocess(task, cand) != "invalid":
                final_raw = cand
                break
        cleaned = postprocess(task, final_raw)
        rec = {
            "sample_id": sid,
            "task": task,
            "model_output": cleaned,
            "source": sample.get("source", ""),
        }
        if args.save_raw:
            rec["raw_model_output"] = final_raw
        records.append(rec)
    return records

def write_output(out_dir: str, in_path: str, records: Iterable[Dict[str, Any]]) -> None:
    """
    Write records to out_dir/filename (same name as input), one JSON per line.
    """
    ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(in_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main() -> None:
    args = parse_args()
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {args.task}. Must be one of: {sorted(SUPPORTED_TASKS)}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    files = list_txt_files(args.anno_dir, args.pattern)
    if not files:
        print(f"No files found in {args.anno_dir} matching {args.pattern}")
        return
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(fp)}")
        recs = process_file(fp, args.task, args, model, processor)
        write_output(args.out_dir, fp, recs)

if __name__ == "__main__":
    main()
