"""Convert the released STAR-6K Parquet files to local training inputs.

Copy the four encoded images without re-encoding. Build the exact two-world
fields consumed by the training dataset and its counterfactual-world sampler.
"""
import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


VIEWS = {
    "image_original": "student.png", "image_edited": "edited.png",
    "crop_original": "teacher.png", "crop_edited": "edited-teacher.png",
}


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    shards = [source] if source.is_file() else sorted((source / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No released Parquet shards at {source}")
    output.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                row_id = row["id"]
                if row_id in seen:
                    raise ValueError(f"Repeated pair: {row_id}")
                seen.add(row_id)
                if row["answer_original"] == row["answer_edited"]:
                    raise ValueError(f"Unchanged answer in {row_id}")
                folder = output / "media" / row_id
                folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key, name in VIEWS.items():
                    image_bytes = row[key]["bytes"]
                    if not image_bytes:
                        raise ValueError(f"Missing image bytes: {row_id}/{key}")
                    path = folder / name
                    path.write_bytes(image_bytes)
                    paths[key] = str(path)
                with Image.open(io.BytesIO(row["image_original"]["bytes"])) as im:
                    width, height = im.size
                x0, y0, x1, y1 = row["crop_bbox_pixels"]
                footprint = [x0/width, y0/height, x1/width, y1/height]
                annotation = row["evidence_bbox_normalized"]
                prompt = row["prompt"]
                records.append({
                    "data_source": "zwz_rl_vqa_bbox_teacher",
                    "ability": "visual_question_answering",
                    "prompt": [{"role": "user", "content": prompt}],
                    "images": [{"path": paths["image_original"]}],
                    "bbox_images": [{"path": paths["crop_original"]}],
                    "reward_model": {"style": "none", "ground_truth": row["answer_original"]},
                    "evidence_bbox": annotation,
                    "extra_info": {
                        "answer": row["answer_original"], "question": prompt,
                        "grounded_bbox": annotation,
                        "attention_gap": {
                            "original_path": paths["image_original"],
                            "bbox": footprint, "annotation_bbox": annotation,
                            "teacher_bbox_pixels": row["crop_bbox_pixels"],
                            "geometry_source": row["geometry_source"],
                        },
                    },
                    "world_pair": {
                        "id": row_id,
                        "edited_image": paths["image_edited"],
                        "edited_teacher_image": paths["crop_edited"],
                        "original_answer": row["answer_original"],
                        "edited_answer": row["answer_edited"],
                        "option_letters": list(row["options"]),
                    },
                })
        print(f"Prepared {len(records)} pairs after {shard.name}", flush=True)
    pq.write_table(pa.Table.from_pylist(records), output / "worlds.parquet", compression="zstd")
    summary = {"pairs": len(records), "images": 4*len(records),
               "source_shards": len(shards), "image_encoding": "unchanged",
               "training_file": str(output / "worlds.parquet")}
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="STAR-6K release directory, or one Parquet shard")
    parser.add_argument("--output", required=True, type=Path, help="Local node storage; use the same path on every node")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
