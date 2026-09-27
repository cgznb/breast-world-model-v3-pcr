"""Strict source-available clinical tokens and ordered treatment-plan tokens."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn

CATEGORICAL = ("treatment_arm", "hr_status", "her2_status", "mammaprint", "menopausal_status", "stage_i", "stage_j")
NUMERIC = ("age", "delta_days")
ALLOWED = set(CATEGORICAL + NUMERIC) | {"interval_verified", "action_segments"}


def validate_condition(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Conditions must be a mapping")
    unknown = set(record) - ALLOWED
    if unknown:
        raise ValueError(f"Forbidden/unknown inference conditions: {sorted(unknown)}. Labels and future observations belong only in targets.")
    import copy
    result = copy.deepcopy(record)
    for key in CATEGORICAL:
        value = result.get(key)
        if value is not None and not isinstance(value, (str, int)):
            raise ValueError(f"{key} must be a categorical scalar")
    verified = result.get("interval_verified", False)
    if not isinstance(verified, bool):
        raise ValueError("interval_verified must be a JSON boolean")
    for key in NUMERIC:
        value = result.get(key)
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {key}")
            result[key] = value
    if verified and (result.get("delta_days") is None or result["delta_days"] <= 0):
        raise ValueError("A verified clinical interval must have positive delta_days")
    if not verified:
        # Unverified de-identified dates must not become biological-time inputs.
        result["delta_days"] = None
    events = result.get("action_segments", [])
    if not isinstance(events, list):
        raise ValueError("action_segments must be a list")
    for event in events:
        required = {"drug", "start", "end", "dose", "known_at_source"}
        if not isinstance(event, dict) or set(event) != required:
            raise ValueError(f"Each action segment must have exactly {sorted(required)}")
        if event["known_at_source"] is not True:
            raise ValueError("Actual future treatment is not an available baseline condition; explicitly supply an available treatment plan")
        if not isinstance(event["drug"], str) or not event["drug"].strip():
            raise ValueError("Action drug must be a nonempty category")
        if any(not math.isfinite(float(event[k])) for k in ("start", "end", "dose")):
            raise ValueError("Nonfinite action values")
        for key in ("start", "end", "dose"):
            event[key] = float(event[key])
        if event["end"] <= event["start"] or event["dose"] < 0:
            raise ValueError("Invalid action duration or dose")
    result["action_segments"] = sorted(events, key=lambda e: (e["start"], e["end"], e["drug"]))
    return result


@dataclass
class ConditionSchema:
    vocabularies: dict[str, list[str]]
    numeric: dict[str, dict[str, float]]
    drugs: list[str]
    fit_split: str = "train"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def fit(cls, records):
        if not records or any(r.get("split") != "train" for r in records):
            raise ValueError("Fit condition schema on nonempty training records ONLY")
        conditions = [validate_condition(r["conditions"]) for r in records]
        vocab = {key: sorted({str(c[key]) for c in conditions if c.get(key) is not None}) for key in CATEGORICAL}
        numeric = {}
        for key in NUMERIC:
            values = np.asarray([c[key] for c in conditions if c.get(key) is not None], dtype=float)
            numeric[key] = {"mean": float(values.mean()) if len(values) else 0.0,
                            "std": float(values.std()) if len(values) > 1 and float(values.std()) > 1e-5 else 1.0}
        drugs = sorted({e["drug"] for c in conditions for e in c.get("action_segments", [])})
        return cls(vocab, numeric, drugs)


class ClinicalEncoder(nn.Module):
    def __init__(self, schema: ConditionSchema, dim: int, heads: int):
        super().__init__()
        if schema.fit_split != "train":
            raise ValueError("Clinical vocabulary is not training-fitted")
        self.schema, self.dim = schema, dim
        self.maps = {k: {v: i+2 for i, v in enumerate(vals)} for k, vals in schema.vocabularies.items()}
        self.categorical = nn.ModuleDict({key: nn.Embedding(len(schema.vocabularies[key])+2, dim) for key in CATEGORICAL})
        self.numeric = nn.ModuleDict({key: nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)) for key in NUMERIC})
        self.field = nn.Parameter(torch.randn(len(CATEGORICAL)+len(NUMERIC), dim) * .02)
        self.direction = nn.Embedding(2, dim)
        self.drug_map = {v: i+2 for i, v in enumerate(schema.drugs)}
        self.drug = nn.Embedding(len(schema.drugs)+2, dim)
        self.action_numeric = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(dim), enable_nested_tensor=False)

    def forward(self, records: list[dict], direction=1):
        if not records:
            raise ValueError("Empty condition batch")
        values = [validate_condition(r) for r in records]
        device = self.field.device
        tokens = []
        for key in CATEGORICAL:
            ids = [0 if v.get(key) is None else self.maps[key].get(str(v[key]), 1) for v in values]
            tokens.append(self.categorical[key](torch.tensor(ids, device=device)))
        for key in NUMERIC:
            stat = self.schema.numeric[key]
            rows = [[0.0, 1.0] if v.get(key) is None else [(float(v[key])-stat["mean"])/stat["std"], 0.0] for v in values]
            tokens.append(self.numeric[key](torch.tensor(rows, device=device, dtype=torch.float32)))
        table = torch.stack(tokens, 1) + self.field[None]
        # A learnt NULL event is preferable to silently treating padding as drug 0.
        nevents = max(1, max(len(v.get("action_segments", [])) for v in values))
        if nevents > 64:
            raise ValueError("More than 64 treatment segments: summarize verified segments explicitly")
        drugs = torch.zeros((len(values), nevents), dtype=torch.long, device=device)
        numbers = torch.zeros((len(values), nevents, 4), device=device)
        numbers[..., -1] = 1
        for i, v in enumerate(values):
            for j, event in enumerate(v.get("action_segments", [])):
                drugs[i, j] = self.drug_map.get(event["drug"], 1)
                numbers[i, j] = numbers.new_tensor([float(event["start"])/365., float(event["end"])/365.,
                                                     math.log1p(float(event["dose"])), 0.0])
        action = self.drug(drugs) + self.action_numeric(numbers)
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (forecast) or -1 (retrodiction)")
        dtok = self.direction.weight[0 if direction == 1 else 1][None, None].expand(len(values), 1, -1)
        counts = torch.tensor([max(1, len(v.get("action_segments", []))) for v in values], device=device)
        action_padding = torch.arange(nevents, device=device)[None] >= counts[:, None]
        padding = torch.cat((torch.zeros(len(values), table.shape[1], dtype=torch.bool, device=device),
                             action_padding, torch.zeros(len(values), 1, dtype=torch.bool, device=device)), 1)
        result = self.context(torch.cat((table, action, dtok), 1), src_key_padding_mask=padding)
        # Avoid padding tokens being treated as extra medication conditions by U-Net.
        # Ragged batches are represented by the observed/NULL action MEAN plus typed table tokens.
        action_result = result[:, table.shape[1]:-1].masked_fill(action_padding[..., None], 0.)
        action_result = action_result.sum(1, keepdim=True) / counts[:, None, None]
        return torch.cat((result[:, :table.shape[1]], action_result, result[:, -1:]), 1)
