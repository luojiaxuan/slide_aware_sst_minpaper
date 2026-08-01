# MCIF Beyond-OCR Candidate Inventory V1

这是 private、outcome-side、reference-aware 的 R1/R2 候选池，用于回答：MCIF 中是否存在
flat OCR 不能表达、但 structure-preserving text 或 raw visual semantics 可提前提供的
En→Zh target events。它不是人工标注、模型结果或 `vision > OCR` 证据。

## Canonical Artifact

- Hugging Face：
  [`gavinlaw/slide-aware-sst-mcif-outcomes@01defe41/beyond_ocr_candidate_inventory_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/01defe410b4fde07c647d8ed241dfbe501b5d691/beyond_ocr_candidate_inventory_v1)
- revision：`01defe410b4fde07c647d8ed241dfbe501b5d691`
- tag：`mcif-beyond-ocr-candidate-inventory-v1`
- privacy：已验证 `private=True`
- Git manifest：
  [`manifests/mcif_beyond_ocr_candidate_inventory_v1_20260801.json`](manifests/mcif_beyond_ocr_candidate_inventory_v1_20260801.json)

## Candidate Contract

R1 strict 只从 `chart_markdown`、`table_html`、`formula_latex` 的实际可见内容抽取；
HTML/LaTeX markup 与 serialized JSON field names 不算 evidence。候选还必须不在当前 R0
flat OCR 中。

R2 semantic 只从 frozen source-only Qwen3-VL 的 `scene_summary`、`objects`、`actions`、
`spatial_relations` 提议。`ocr_text` 被完全排除，且出现在当前 R0 或任何实际 R1 block
内容中的候选也被排除。VLM 自带 nominal timing 不参与因果判断；时间只来自校正后的
`t+0.5s` evidence ladder。

两层候选都必须满足 official English reference 首次出现、segment start 时当前 evidence
仍因果可用，并记录 earliest contiguous evidence 与保守 lead lower bound。

## Inventory

- 919 official reference segments / 21 talks；
- R1 strict：2 candidates / 2 talks / 2 segments；
- R2 semantic：150 candidates / 21 talks / 118 segments；
- R2 中 122 个 lead 至少 5 秒，86 个至少 10 秒，最大 173.677 秒；
- 120 个 candidate-bearing segments（R1/R2 union）。

R1 的两个候选是 `graduate school` 与 `metric`。R2 同时包含可能有意义的流程/关系项和
`content`、`presentation` 等泛化描述，因此必须经过独立 human visual-correctness、
OCR-insufficiency、event/target-realization 核验，不能自动进入正式 event set。

## Provenance

- references SHA256：
  `e7840e1b0cd0589e1a643b5d450c86372c4e6c6f67de594959926f9152ae172a`
- R0/R1/R2 ladder SHA256：
  `8f77312b93562afd8a92ea0b3139fe5f91b21b08e9740d311a1fd0a83b594f7f`
- source-only VLM output SHA256：
  `55a6dafe5ebd1fc5f37226de7ce48e601d61e1fb914287a81bdb8f92b0479682`
- builder Git commit：`eb601f64efd23c03725e9d28245ad4628c4a3919`
- bundle checksum manifest SHA256：
  `1a6c172cfef948110e232560dd22e623e7b35d9fa97df1e060d07611e73fff15`

正式构建与独立第二次构建的 6 files byte-identical；private HF 的 6 files 已全量回下载并
逐字节验证。构建命令由
[`../code/scripts/build_mcif_beyond_ocr_candidate_inventory.py`](../code/scripts/build_mcif_beyond_ocr_candidate_inventory.py)
提供，输入必须显式传入上述 hashes、exact model revision 与两个 frozen prompt hashes。

## Use Boundary

该 artifact 包含 official references 和 reference-aware 候选，禁止挂载到 inference、训练、
source-side screening 或 control construction。它不能给 R0 target-event author 看，也不能用来
删除“不好看”的 R2 condition。下一步必须生成物理隔离的 R1/R2 human-validation view；只有
人工确认 visual description 正确、OCR 确实不足、event/target realization 合格的项，才可进入
后续 audio sufficiency 与 event packet 流程。
