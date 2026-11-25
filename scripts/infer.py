"""Run one local STAR Hugging Face checkpoint on an image and a question."""
import argparse

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": args.question}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    answer = ids[:, inputs["input_ids"].shape[-1]:]
    print(processor.batch_decode(answer, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
