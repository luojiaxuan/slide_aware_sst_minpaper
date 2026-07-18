from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: Type[T]) -> list[T]:
    items: list[T] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(model.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return items


def write_jsonl(path: str | Path, items: Iterable[BaseModel | dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, BaseModel):
                obj = item.model_dump(mode="json")
            else:
                obj = item
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
