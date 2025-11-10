# OCR Post-Processing Tuning & Evaluation

This note describes how to adjust the Indonesian plate post-processing heuristics
and how to measure their impact before deploying changes.

## Configurable Heuristics

- Default tuning lives in `configs/ocr/postproc_indonesia.yaml`.
- Parameters map directly to the dataclass in `ocr_service.postprocess.PostprocessTuning`:
  - `suffix_len_lt3_penalty`: bias against suffixes shorter than 3 chars.
  - `suffix_penalty_map`: additive penalties for specific substrings (e.g., `DC`, `TT`).
  - `suffix_bonus_map`: negative penalties (bonuses) for desirable forms like `VIN`, `PDC`, `OE`.
  - `suffix_duplicate_penalty`: cost applied when a 2-letter suffix repeats the same letter (e.g., `SS`).
  - `suffix_vowel_pair_penalty`: discourages triples with adjacent vowels (glare artefacts).
  - `insert_bias_vi_to_vin`, `insert_bias_pdc`: lower the edit cost for targeted insertions such as `VI→VIN` and `DC→PDC`.
  - `duplicate_collapse_min_len`: how many repeated letters must appear before collapsing (`SS→S`).

To override values, copy the YAML and edit the scalars. For example:

```
# configs/ocr/postproc_indonesia_night.yaml
suffix_bonus_map:
  VIN: -0.15
  OE: -0.2
insert_bias_vi_to_vin: 0.1
```

Use `load_postprocess_config(path)` at runtime or set the `--postproc-config` flag once we add it to CLI entrypoints.

## Evaluation Workflow

1. Collect a small validation file with raw OCR strings and ground truth plates.
   - Format: JSON Lines or CSV with columns `raw` and `expected` (optional `allowed_prefixes`).
2. Run the eval helper:

```
PYTHONPATH=src python3 tools/eval_postprocess.py \
  --input data/eval/postproc_samples.jsonl \
  --config configs/ocr/postproc_indonesia.yaml \
  --output export/metrics/postproc_eval_$(date +%Y%m%d).json
```

3. The script prints totals, exact-match rate, valid-rate, and the number of
   samples altered by post-processing. The optional output JSON captures failures
   for review. Archive this under `export/metrics/` and reference it in progress logs.

4. To compare configs, run the script twice (default + new config) and compare
   `exact_rate`. If the new config regresses, keep the previous YAML.

## Runtime Verification

- Run the unit tests to guarantee regressions remain covered:
  `python3 -m pytest -q tests/ocr_service/test_postprocess.py`.
- Capture field metrics by enabling debug logging in the OCR service; log the
  raw OCR, post-processed text, and whether the heuristics altered the result.
- Before adopting new weights in production, attach the eval JSON and a short
  summary to `progress/<date>_session-*.md` to keep plan.md §5 in sync.

## Next Steps

- Surface the YAML path via CLI/API flag so deployments can flip between configs
  without code changes.
- Add a constrained beam-search fallback (see `plan.md §5`, Week 2 follow-ups)
  once heuristic tuning saturates.
