# Smoke test log

The scaffold was smoke-tested on the included 3-example toy dataset with the mock translator.

Commands run from `repo/`:

```bash
PYTHONPATH=src python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition no_context
PYTHONPATH=src python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition naive_all_context
PYTHONPATH=src python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition policy
PYTHONPATH=src python scripts/evaluate.py --config configs/min_zh_en.yaml
PYTHONPATH=src python scripts/make_paper_tables.py --config configs/min_zh_en.yaml
PYTHONPATH=src python -m pytest -q
```

Toy output table:

```csv
condition,hda,term_recall,context_overuse_rate,n
no_context,0.0,0.0,0.0,3
naive_all_context,1.0,1.0,0.6666666666666666,3
policy,0.3333333333333333,0.3333333333333333,0.0,3
```

These numbers are only a sanity check. They are not paper results.
