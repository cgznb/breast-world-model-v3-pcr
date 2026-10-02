"""Patient-normalized landmarks and unbiased stage-balanced edge sampling."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class EdgeSample:
    patient_index: int
    source_stage: int
    target_stage: int
    kind: str
    probability: float
    objective_weight: float
    effective_weight: float

    @property
    def q(self):
        return self.probability


class PatientLandmarkSampler:
    def __init__(self, store, seed, include_terminal=True):
        self.groups = [(i, store.landmarks(i, include_terminal)) for i in store.by_split["train"]]
        self.groups = [(i, stages) for i, stages in self.groups if stages]
        if not self.groups:
            raise ValueError("No eligible training patients")
        self.generator = torch.Generator().manual_seed(seed)
        grouped = {}
        for index, stages in self.groups:
            patient = store.patients[index]
            visits = patient["visits"]
            available = {v["stage"] for v in visits}
            weight = 1 / (len(self.groups) * len(stages))
            for stage in stages:
                observed = tuple(v["stage"] for v in visits
                                 if v["stage"] <= stage and v["available_at"] <= stage)
                future = tuple(j in available for j in range(stage + 1, 4))
                key = (stage, observed, future, patient["target"]["pcr"] is not None)
                grouped.setdefault(key, []).append(((index, stage), weight))
        self.batch_keys = tuple(grouped)
        self.batch_groups = tuple(tuple(task for task, _ in grouped[key]) for key in self.batch_keys)
        self.batch_task_weights = tuple(torch.tensor([weight for _, weight in grouped[key]], dtype=torch.float64)
                                        for key in self.batch_keys)
        self.batch_group_weights = torch.tensor([weights.sum() for weights in self.batch_task_weights],
                                                dtype=torch.float64)

    def sample(self, count):
        result = []
        for _ in range(count):
            index, stages = self.groups[int(torch.randint(len(self.groups), (), generator=self.generator))]
            stage = stages[int(torch.randint(len(stages), (), generator=self.generator))]
            result.append((index, stage))
        return result

    def sample_batch(self, count, *, origin_stage=None):
        if type(count) is not int or count < 1:
            raise ValueError("Batch sample count must be a positive integer")
        if origin_stage is not None and (type(origin_stage) is not int or origin_stage not in range(4)):
            raise ValueError("Origin stage must be a canonical integer stage")
        eligible = [i for i, key in enumerate(self.batch_keys) if origin_stage is None or key[0] == origin_stage]
        if not eligible:
            raise ValueError("No eligible training landmarks at requested origin stage")
        # Group mass and within-group weights preserve patient-uniform landmark marginals.
        selected = int(torch.multinomial(self.batch_group_weights[eligible], 1, generator=self.generator))
        group = eligible[selected]
        rows = torch.multinomial(self.batch_task_weights[group], count, replacement=True, generator=self.generator)
        return [self.batch_groups[group][int(row)] for row in rows]

    def state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])


class BalancedEdgeSampler:
    def __init__(self, store, seed, include_bridges=False, bridge_weight=0.25):
        self.groups = store.edge_index("train", include_bridges=include_bridges)
        self.groups = {key: value for key, value in self.groups.items() if value}
        if not self.groups:
            raise ValueError("No eligible adjacent training edges")
        if not 0 <= bridge_weight <= 1:
            raise ValueError("Bridge auxiliary weight must be in [0,1]")
        self.types = sorted(self.groups)
        adjacent = [key for key in self.types if key[1] - key[0] == 1]
        bridge = [key for key in self.types if key[1] - key[0] > 1]
        self.type_weights = {key: 1 / len(adjacent) for key in adjacent}
        self.type_weights.update({key: bridge_weight / len(bridge) for key in bridge})
        self.generator = torch.Generator().manual_seed(seed)
        self.coverage = Counter()
        self.batch_groups = {}
        self.batch_group_weights = {}
        for edge, indices in self.groups.items():
            grouped = {}
            for index in indices:
                observed = tuple(v["stage"] for v in store.patients[index]["visits"]
                                 if v["stage"] <= edge[0] and v["available_at"] <= edge[0])
                grouped.setdefault(observed, []).append(index)
            self.batch_groups[edge] = tuple(tuple(group) for group in grouped.values())
            self.batch_group_weights[edge] = torch.tensor([len(group) for group in grouped.values()], dtype=torch.float64)

    def _entry(self, edge, index, probability, exhaustive=False):
        src, dst = edge
        weight = self.type_weights[edge] / len(self.groups[edge])
        return EdgeSample(index, src, dst, "adjacent" if dst == src + 1 else "bridge_auxiliary",
                          probability, weight, weight if exhaustive else weight / probability)

    def sample(self, count):
        result = []
        for _ in range(count):
            edge = self.types[int(torch.randint(len(self.types), (), generator=self.generator))]
            group = self.groups[edge]
            index = group[int(torch.randint(len(group), (), generator=self.generator))]
            probability = 1 / (len(self.types) * len(group))
            self.coverage[f"{edge[0]}{edge[1]}"] += 1
            result.append(self._entry(edge, index, probability))
        return result

    def sample_batch(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("Batch sample count must be a positive integer")
        edge = self.types[int(torch.randint(len(self.types), (), generator=self.generator))]
        selected = int(torch.multinomial(self.batch_group_weights[edge], 1, generator=self.generator))
        group = self.batch_groups[edge][selected]
        rows = torch.randint(len(group), (count,), generator=self.generator)
        probability = 1 / (len(self.types) * len(self.groups[edge]))
        self.coverage[f"{edge[0]}{edge[1]}"] += count
        return [self._entry(edge, group[int(row)], probability) for row in rows]

    def all_edges(self):
        return [self._entry(edge, index, 1.0, exhaustive=True)
                for edge in self.types for index in self.groups[edge]]

    def state_dict(self):
        return {"generator": self.generator.get_state(), "coverage": dict(self.coverage)}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.coverage = Counter(state["coverage"])

    def audit(self):
        return {"objective": "mean_adjacent_type_mean_patient_plus_weighted_bridge_auxiliary",
                "edge_population": {f"{a}{b}": len(v) for (a, b), v in self.groups.items()},
                "empirical_coverage": dict(self.coverage),
                "exhaustive_weights": [asdict(v) for v in self.all_edges()]}
