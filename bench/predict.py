"""Run one baseline over the benchmark data and write what it found as JSON.

    uv run --project bench/presidio --locked python bench/predict.py \\
        --detector presidio_default --data bench/data/v1.jsonl --out predictions.json

A baseline that needs packages of its own runs in an environment of its own, with a lock of
its own (``bench/presidio/``), so that those packages never enter the lock of the product.
This script is what runs there. ``evaluate.py`` starts it, reads the file and does all the
counting: nothing here knows the labels.

It needs the standard library and the baseline, and nothing else: not ``airlock_pii``, which
the environment of a baseline does not have. The file it writes::

    {"detector": "presidio_default",
     "info": {"presidio-analyzer": "...", "model": "..."},
     "predictions": [{"id": "ru-loan-03#1", "spans": [["PERSON", 7, 23]], "seconds": 0.0071}]}

- ``spans`` are ``[type, start, end]``, with the positions in the text of the sample and the
  type as the baseline maps it: a type of the data, or ``OTHER``.
- ``seconds`` is the time of that one call. The detector is called once before anything is
  timed, so that what it loads on first use is not counted: the same way ``evaluate.run``
  times a detector that runs in its own process.
- ``info`` is what ``describe()`` of the baseline returns: versions, model, configuration.

Nothing from the data is written but the ids of the samples and positions in their texts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main", "predict"]


def predict(name: str, data: Path) -> dict[str, object]:
    """Return what the baseline ``name`` finds in every sample of ``data``, as it is written.

    Raises ``ValueError`` for a line that is no sample and for an answer that is no list of
    spans of the text, naming the line or the sample and never quoting it.
    """
    module = importlib.import_module(f"baselines.{name}")
    detect = module.detect
    samples = []
    with data.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                samples.append((str(record["id"]), str(record["text"])))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{data.name}, line {number}: not a benchmark sample") from None
    if samples:
        detect(samples[0][1])
    predictions = []
    for sample_id, text in samples:
        started = time.perf_counter()
        found = detect(text)
        seconds = time.perf_counter() - started
        spans = []
        for span in found:
            try:
                entity_type, start, end = span
                well_formed = (
                    isinstance(entity_type, str)
                    and isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(text)
                )
            except (TypeError, ValueError):
                well_formed = False
            if not well_formed:
                raise ValueError(f"the detector returned what is not a span, for {sample_id}")
            spans.append([entity_type, start, end])
        predictions.append({"id": sample_id, "spans": spans, "seconds": seconds})
    describe = getattr(module, "describe", None)
    return {
        "detector": name,
        "info": describe() if callable(describe) else None,
        "predictions": predictions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a baseline and write what it found.")
    parser.add_argument("--detector", required=True, help="a module of baselines/")
    parser.add_argument("--data", type=Path, required=True, help="samples from generate.py")
    parser.add_argument("--out", type=Path, required=True, help="where the JSON goes")
    args = parser.parse_args(argv)
    try:
        written = predict(args.detector, args.data)
    except (ImportError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(written, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
