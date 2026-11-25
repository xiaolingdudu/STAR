# STAR-6K data contract

The released Parquet has one row per accepted pair (6,176 rows):

| Fields | Meaning |
|---|---|
| `id`, `source_row_index` | Pair identity and original source row |
| `prompt`, `question`, `options` | Actual training prompt, question, and A/B/C/D choices |
| `answer_original`, `answer_edited` | Distinct correct option letters |
| `image_original`, `image_edited` | Student full-image views |
| `crop_original`, `crop_edited` | Teacher evidence crops |
| `evidence_bbox_normalized` | Evidence annotation, normalized xyxy |
| `crop_bbox_pixels` | Actual crop footprint in the full image, xyxy; right/bottom exclusive |
| `crop_bbox_normalized` | The same footprint in normalized coordinates |
| `geometry_source`, revision flags, `review_status` | Geometry and acceptance provenance |

Do not substitute the evidence annotation for the crop footprint. The latter
can include extra context. Teacher crop resolution in this release is twice
the corresponding full-image footprint. The importer preserves those images,
and maps the footprint back to full-image coordinates. The mean-RGB control is
generated at training time and is not a fifth stored image.

The full-image Student views already contain their training evidence-box
marking. Do not draw another box, crop them to the Teacher view, or replace them
with the unmarked upstream image when reproducing the released training data.

The importer constructs the original runtime format:

```text
prompt                          [{role: user, content: <released prompt>}]
images                          [{path: <original full image>}]
bbox_images                     [{path: <original crop>}]
evidence_bbox                   annotation in normalized xyxy
extra_info.attention_gap.bbox   actual registered crop footprint
world_pair.edited_image         edited full image
world_pair.edited_teacher_image edited crop
world_pair.original_answer      original label
world_pair.edited_answer        edited label
```

Both worlds retain the exact released question/options order, including
reviewed option corrections. Image bytes are copied without re-encoding.
The original 6,241-row source and the final 6,176-pair STAR dataset are different
training sets; do not silently restore the 65 excluded rows.

The dataset's own README, license and provenance accompany its release. They
are not replaced by the Apache code license in this repository.
