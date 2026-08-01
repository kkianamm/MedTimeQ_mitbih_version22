"""MIT-BIH Arrhythmia beat-classification dataset for MedTsLLM.

The original repository's PTB-XL loader performs one-label-per-record
classification. MIT-BIH instead provides an annotation for each heartbeat, so
this loader extracts a fixed-length window centered on every annotated beat and
maps the original annotation symbols to the five AAMI heartbeat groups:

    N: normal and bundle-branch/escape beats
    S: supraventricular ectopic beats
    V: ventricular ectopic beats
    F: fusion of ventricular and normal beats
    Q: paced, paced-fusion, and unclassifiable beats

The standard de Chazal inter-patient partition is used: DS1 supplies train/val
records and DS2 is held out for testing. Records 102, 104, 107, and 217 are not
part of this protocol because they contain paced beats.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Reuse the generic whole-window classification base already implemented for
# PTB-XL. A later cleanup may move ClassificationDataset to its own module.
from .ptbxl import ClassificationDataset


AAMI_CLASS_ORDER_5 = ["N", "S", "V", "F", "Q"]
AAMI_CLASS_NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown or paced beat",
}

# ANSI/AAMI EC57 grouping commonly used for MIT-BIH beat classification.
AAMI_SYMBOL_TO_CLASS = {
    # N: normal, bundle branch block, atrial/nodal escape
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: atrial/junctional/supraventricular premature beats
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: premature ventricular contraction / ventricular escape
    "V": "V",
    "E": "V",
    # F: fusion of ventricular and normal beat
    "F": "F",
    # Q: paced, fusion of paced and normal, or unclassifiable
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

# Standard inter-patient split introduced by de Chazal et al.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118",
    "119", "122", "124", "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200",
    "202", "210", "212", "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
]

# Patient-disjoint validation subset selected from DS1. The remaining DS1
# records are training data, while DS2 remains untouched until final testing.
DEFAULT_VAL_RECORDS = ["114", "119", "208", "223"]


class MITDBClassificationDataset(ClassificationDataset):
    """Single-lead MIT-BIH heartbeat classification using AAMI groups."""

    description = (
        "The MIT-BIH Arrhythmia Database contains 48 approximately 30-minute, "
        "two-channel ambulatory ECG recordings sampled at 360 Hz. This task "
        "uses patient-independent heartbeat windows from the MLII lead and "
        "groups expert beat annotations into the AAMI N, S, V, F, and Q classes."
    )
    task_description = (
        "Classify the ECG window centered on the annotated heartbeat as one of "
        "the AAMI heartbeat groups: N normal, S supraventricular ectopic, "
        "V ventricular ectopic, F fusion, or Q unknown/paced."
    )

    sampling_rate = 360

    def __init__(self, config, split):
        cfg = config.get("datasets", {}).get("MITDB", {})

        self.include_q = bool(cfg.get("include_q", True))
        self.class_order = (
            list(AAMI_CLASS_ORDER_5)
            if self.include_q
            else list(AAMI_CLASS_ORDER_5[:-1])
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_order)}

        self.lead = str(cfg.get("lead", "MLII"))
        self.allow_lead_fallback = bool(cfg.get("allow_lead_fallback", True))
        self.trim_beats = int(cfg.get("trim_beats", 10))
        self.use_cache = bool(cfg.get("use_cache", True))
        self.val_records = [str(x) for x in cfg.get("val_records", DEFAULT_VAL_RECORDS)]

        root_value = str(cfg.get("root", "data/mitdb"))
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / root).resolve()
        self.root = self._resolve_data_root(root)

        super().__init__(config, split)

    @property
    def n_classes(self):
        return len(self.class_order)

    @property
    def class_names(self):
        return [AAMI_CLASS_NAMES[c] for c in self.class_order]

    def _resolve_data_root(self, requested_root: Path) -> Path:
        """Accept either a flat extraction or common nested zip directories."""
        candidates = [
            requested_root,
            requested_root / "mitdb-1.0.0",
            requested_root / "mitdb",
        ]
        for candidate in candidates:
            if (candidate / "RECORDS").exists() or (candidate / "100.hea").exists():
                return candidate
        raise FileNotFoundError(
            "MIT-BIH files were not found. Expected RECORDS and 100.hea under "
            f"{requested_root}. Download/extract PhysioNet mitdb v1.0.0 there."
        )

    def _records_for_split(self, split: str) -> List[str]:
        val_set = set(self.val_records)
        unknown = val_set.difference(DS1_RECORDS)
        if unknown:
            raise ValueError(
                f"MITDB val_records must be selected from DS1; invalid: {sorted(unknown)}"
            )

        if split == "train":
            return [record for record in DS1_RECORDS if record not in val_set]
        if split == "val":
            return [record for record in DS1_RECORDS if record in val_set]
        if split == "test":
            return list(DS2_RECORDS)
        raise ValueError(f"Unknown split: {split!r}")

    def _cache_path(self, split: str, records: Sequence[str]) -> Path:
        record_hash = hashlib.sha1(",".join(records).encode("utf-8")).hexdigest()[:8]
        lead_tag = "".join(ch if ch.isalnum() else "_" for ch in self.lead)
        q_tag = "aami5" if self.include_q else "aami4"
        filename = (
            f"{split}_h{int(self.history_len)}_{lead_tag}_{q_tag}_"
            f"trim{self.trim_beats}_{record_hash}.npz"
        )
        return self.root / "cache" / filename

    def _select_lead(self, record) -> Tuple[int, str]:
        names = list(record.sig_name or [])
        lower_names = [name.lower() for name in names]
        requested = self.lead.lower()
        if requested in lower_names:
            idx = lower_names.index(requested)
            return idx, names[idx]

        if not self.allow_lead_fallback:
            raise ValueError(
                f"Lead {self.lead!r} not present in record {record.record_name}; "
                f"available leads: {names}"
            )
        if not names:
            raise ValueError(f"Record {record.record_name} contains no signal channels")
        return 0, names[0]

    @staticmethod
    def _center_window(signal: np.ndarray, center: int, length: int) -> np.ndarray:
        """Return a fixed window, zero-padding only at record boundaries."""
        left = length // 2
        right = length - left
        start = int(center) - left
        end = int(center) + right

        src_start = max(start, 0)
        src_end = min(end, signal.shape[0])
        window = np.zeros((length,), dtype=np.float32)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        window[dst_start:dst_end] = signal[src_start:src_end]
        return window

    def _mapped_annotations(self, samples: Iterable[int], symbols: Iterable[str]):
        mapped = []
        for sample, symbol in zip(samples, symbols):
            group = AAMI_SYMBOL_TO_CLASS.get(symbol)
            if group is None or group not in self.class_to_idx:
                continue
            mapped.append((int(sample), group, symbol))

        # Common preprocessing removes unstable start/end beats. Apply trimming
        # after non-beat annotations have already been discarded.
        if self.trim_beats > 0 and len(mapped) > 2 * self.trim_beats:
            mapped = mapped[self.trim_beats:-self.trim_beats]
        return mapped

    def _build_split(self, records: Sequence[str]) -> Dict[str, np.ndarray]:
        import wfdb

        windows: List[np.ndarray] = []
        labels: List[int] = []
        descriptions: List[str] = []
        crop = int(self.history_len)

        for record_name in records:
            path = str(self.root / record_name)
            record = wfdb.rdrecord(path, physical=True)
            annotation = wfdb.rdann(path, "atr")

            lead_idx, actual_lead = self._select_lead(record)
            signal = np.asarray(record.p_signal[:, lead_idx], dtype=np.float32)
            mapped = self._mapped_annotations(annotation.sample, annotation.symbol)

            for sample, group, _original_symbol in mapped:
                beat = self._center_window(signal, sample, crop)
                windows.append(beat[:, None])  # [T, 1]
                labels.append(self.class_to_idx[group])
                # No target label or patient ID is included, preventing text leakage.
                descriptions.append(
                    "ECG context: "
                    f'{{"database": "MIT-BIH Arrhythmia", '
                    f'"lead": "{actual_lead}", '
                    f'"sampling_rate_hz": {int(record.fs)}}}'
                )

        if not windows:
            raise RuntimeError(
                f"No mapped MIT-BIH beats were produced from records: {records}"
            )

        data = np.stack(windows).astype(np.float32, copy=False)
        targets = np.asarray(labels, dtype=np.int64)
        text = np.asarray(descriptions, dtype=str)
        return {"data": data, "labels": targets, "descriptions": text}

    def get_data(self, split=None):
        split = split or self.split
        records = self._records_for_split(split)
        cache_path = self._cache_path(split, records)

        if self.use_cache and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            return {
                "data": cached["data"],
                "labels": cached["labels"],
                "descriptions": cached["descriptions"].tolist(),
            }

        result = self._build_split(records)
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                data=result["data"],
                labels=result["labels"],
                descriptions=result["descriptions"],
            )

        result["descriptions"] = result["descriptions"].tolist()
        return result


mitdb_datasets = {
    "classification": MITDBClassificationDataset,
}
