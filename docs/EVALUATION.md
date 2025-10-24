# Evaluation Guide (E2E Accuracy & Quality)

This guide shows how to generate predictions, prepare ground truth, run the
`tools/eval_e2e.py` evaluator, and inspect low-confidence cases.

## Prerequisites
- API configured to persist events and snapshots:
  - `ALPR_EVENTS_DB=export/events.sqlite`
  - `ALPR_SNAPSHOTS_DIR=export/snapshots`
- Detector/OCR engines configured via env vars (TRT or ONNX per plan.md §5).
- Python environment with project installed (`pip install -e .`).

## Option A — Real API Flow
1) Start API

```
uvicorn api_server.server:create_app --factory --host 0.0.0.0 --port 8080
```

2) Send requests with stable IDs

```
curl -F "image=@/path/to/plate1.jpg" -F "request_id=req-001" http://localhost:8080/v1/alpr
curl -F "image=@/path/to/plate2.jpg" -F "request_id=req-002" http://localhost:8080/v1/alpr
```

3) Create ground-truth CSV

```
mkdir -p data/labels
cat > data/labels/e2e_gt.csv << EOF
request_id,plate
req-001,B 1234 CD
req-002,A 8628 GL
EOF
```

4) Run evaluator

```
python tools/eval_e2e.py \
  --events export/events.sqlite \
  --ground-truth data/labels/e2e_gt.csv \
  --low-conf-threshold 0.85 \
  --low-conf-dir export/low_confidence \
  --output-json export/metrics/e2e_metrics.json
```

Output includes exact-match %, average CER/SER, per-camera stats, and copies
low-confidence snapshots to `export/low_confidence/`.

## Option B — Zero‑setup Demo
Use the seeder to create a small DB + CSV instantly.

```
python tools/seed_events_demo.py
python tools/eval_e2e.py \
  --events export/events.sqlite \
  --ground-truth data/labels/e2e_gt_demo.csv \
  --low-conf-threshold 0.85 \
  --low-conf-dir export/low_confidence_demo \
  --output-json export/metrics/e2e_metrics_demo.json
```

Expected console summary (values approximate):

```
Total samples: 3
Exact matches: 2 (66.67%)
Average CER: 0.0444
Average SER: 0.3333
Missing ground-truth rows: 0
Low-confidence predictions: 1
Per camera:
  cam01: total=2 exact=50.00% cer=0.0833 ser=0.5000
  cam02: total=1 exact=100.00% cer=0.0000 ser=0.0000
```

## Tips & Pitfalls
- IDs must match: evaluator joins by `request_id`.
- Case/format: evaluator uppercases predictions; keep GT uppercased.
- Latest only: if multiple rows share an ID, evaluator uses the latest by `created_at`.
- Snapshots: low-confidence copy only works if `snapshot_path` exists.

## CLI Reference
```
python tools/eval_e2e.py \
  --events <path to SQLite> \
  --ground-truth <path to CSV> \
  --gt-id-column <column name> \
  --gt-plate-column <column name> \
  --low-conf-threshold <0..1> \
  --low-conf-dir <output dir> \
  --output-json <metrics.json>
```

```text
CSV columns default: request_id, plate
SQLite table: events (created by API persistence)
```

