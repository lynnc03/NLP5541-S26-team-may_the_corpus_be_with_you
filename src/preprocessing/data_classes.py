# data_classes.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class CHATFeatures:
    pause_short: int = 0
    pause_medium: int = 0
    pause_long: int = 0
    pause_timed_count: int = 0
    pause_timed_total_sec: float = 0.0
    pause_total: int = 0
    filled_pause_count: int = 0
    repetition_count: int = 0
    revision_count: int = 0
    reformulation_count: int = 0
    trailing_off_count: int = 0
    interrupted_count: int = 0
    disfluency_total: int = 0
    error_count: int = 0
    target_form_count: int = 0
    uncertain_count: int = 0
    unintelligible_xxx: int = 0
    unintelligible_yyy: int = 0
    unintelligible_total: int = 0
    paralinguistic_count: int = 0

    def to_dict(self) -> dict:
        return {
            "pause_short": self.pause_short,
            "pause_medium": self.pause_medium,
            "pause_long": self.pause_long,
            "pause_timed_count": self.pause_timed_count,
            "pause_timed_total_sec": self.pause_timed_total_sec,
            "pause_total": self.pause_total,
            "filled_pause_count": self.filled_pause_count,
            "repetition_count": self.repetition_count,
            "revision_count": self.revision_count,
            "reformulation_count": self.reformulation_count,
            "trailing_off_count": self.trailing_off_count,
            "interrupted_count": self.interrupted_count,
            "disfluency_total": self.disfluency_total,
            "error_count": self.error_count,
            "target_form_count": self.target_form_count,
            "uncertain_count": self.uncertain_count,
            "unintelligible_xxx": self.unintelligible_xxx,
            "unintelligible_yyy": self.unintelligible_yyy,
            "unintelligible_total": self.unintelligible_total,
            "paralinguistic_count": self.paralinguistic_count,
        }


@dataclass
class Annotations:
    mor_raw: Optional[str] = None
    mor_token_count: Optional[int] = None
    gra_raw: Optional[str] = None
    tim_raw: Optional[str] = None
    tim_onset: Optional[int] = None
    tim_offset: Optional[int] = None
    tim_duration_ms: Optional[int] = None
    err_tier_raw: Optional[str] = None
    err_tier_count: int = 0
    com_raw: Optional[str] = None
    act_raw: Optional[str] = None
    gpx_raw: Optional[str] = None

    @property
    def has_mor(self) -> bool:
        return bool(self.mor_raw)

    @property
    def has_gra(self) -> bool:
        return bool(self.gra_raw)

    @property
    def has_tim(self) -> bool:
        return bool(self.tim_raw and self.tim_raw.strip())

    @property
    def has_err_tier(self) -> bool:
        return bool(self.err_tier_raw)

    def to_dict(self) -> dict:
        return {
            "has_mor": self.has_mor,
            "mor_raw": self.mor_raw,
            "mor_token_count": self.mor_token_count,
            "has_gra": self.has_gra,
            "gra_raw": self.gra_raw,
            "has_tim": self.has_tim,
            "tim_raw": self.tim_raw,
            "tim_onset": self.tim_onset,
            "tim_offset": self.tim_offset,
            "tim_duration_ms": self.tim_duration_ms,
            "has_err_tier": self.has_err_tier,
            "err_tier_raw": self.err_tier_raw,
            "err_tier_count": self.err_tier_count,
            "com_raw": self.com_raw,
            "act_raw": self.act_raw,
            "gpx_raw": self.gpx_raw,
        }


@dataclass
class Metadata:
    EXAMINER_CODES: ClassVar[frozenset[str]] = frozenset({"INV", "EXP", "EXA", "CLI", "RES", "ADM"})
    PARENT_CODES: ClassVar[frozenset[str]] = frozenset({"MOT", "FAT", "NAN", "GRA", "SIB", "CAR", "PAR"})

    file_id: str
    label_binary: int
    label: str
    age: Optional[int] = None
    sex: Optional[str] = None
    corpus: Optional[str] = None
    session_types: list[str] = field(default_factory=list)
    pid: Optional[str] = None
    languages: list[str] = field(default_factory=list)
    speaker_codes: list[str] = field(default_factory=list)
    examiner_codes: list[str] = field(default_factory=list)
    parent_codes: list[str] = field(default_factory=list)
    multiple_kids: bool = False
    header_warnings: list[str] = field(default_factory=list)

    @staticmethod
    def get_speaker_role(speaker_code: str) -> str:
        if not speaker_code:
            return "other"
        if speaker_code.startswith("CHI"):
            return "target_child"
        if speaker_code in Metadata.EXAMINER_CODES:
            return "examiner"
        if speaker_code in Metadata.PARENT_CODES:
            return "parent"
        return "other"

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "corpus": self.corpus,
            "label_binary": self.label_binary,
            "label": self.label,
            "age": self.age,
            "sex": self.sex,
            "session_types": "|".join(self.session_types),
            "pid": self.pid,
            "languages": "|".join(self.languages),
            "speaker_codes": "|".join(self.speaker_codes),
            "examiner_codes": "|".join(self.examiner_codes),
            "parent_codes": "|".join(self.parent_codes),
            "multiple_kids": self.multiple_kids,
            "header_warnings": "|".join(self.header_warnings) if self.header_warnings else None,
        }


@dataclass
class Utterance:
    file_id: str
    utterance_index: int
    speaker: str
    text_raw: str = ""
    text_clean: str = ""
    text_surface: str = ""
    text_disfluency_tagged: str = ""
    features: CHATFeatures = field(default_factory=CHATFeatures)
    annotations: Annotations = field(default_factory=Annotations)
    context_before: list = field(default_factory=list)
    context_after: list = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def utterance_id(self) -> str:
        file_id = self.file_id
        idx = self.utterance_index
        formatted_idx = f"{idx:04d}"
        return f"{file_id}_{formatted_idx}"

    @property
    def speaker_role(self) -> str:
        speaker = self.speaker
        role = Metadata.get_speaker_role(speaker)
        return role

    @property
    def is_target_child(self) -> bool:
        speaker = self.speaker
        if not speaker:
            return False
        return speaker.startswith("CHI")

    @property
    def has_both_error_sources(self) -> bool:
        has_inline_errors = self.features.error_count > 0
        has_err_tier = self.annotations.has_err_tier
        return has_inline_errors and has_err_tier

    def to_dict(self) -> dict:
        result = {
            "utterance_id": self.utterance_id,
            "file_id": self.file_id,
            "utterance_index": self.utterance_index,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "is_target_child": self.is_target_child,
            "utterance_raw": self.text_raw,
            "utterance_clean": self.text_clean,
            "utterance_surface": self.text_surface,
            "utterance_disfluency_tagged": self.text_disfluency_tagged,
            "has_both_error_sources": self.has_both_error_sources,
            "context_before": json.dumps(self.context_before),
            "context_after": json.dumps(self.context_after),
            "parse_warnings": "|".join(self.parse_warnings) if self.parse_warnings else None,
        }
        result.update(self.features.to_dict())
        result.update(self.annotations.to_dict())
        return result


@dataclass
class ParsedSpeech:
    metadata: Metadata
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def file_id(self) -> str:
        return self.metadata.file_id

    @property
    def child_utterances(self) -> list[Utterance]:
        return [u for u in self.utterances if u.is_target_child]

    @property
    def chi_speakers(self) -> set[str]:
        return {u.speaker for u in self.utterances if u.is_target_child}

    @property
    def speaker_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for utterance in self.utterances:
            counts[utterance.speaker] = counts.get(utterance.speaker, 0) + 1
        return counts

    @property
    def mean_mlu_morphemes(self) -> Optional[float]:
        chi_tokens = [
            u.annotations.mor_token_count
            for u in self.child_utterances
            if u.annotations.mor_token_count is not None
        ]
        if not chi_tokens:
            return None
        return round(sum(chi_tokens) / len(chi_tokens), 4)

    def utterance_rows(self) -> list[dict]:
        rows = []
        meta = self.metadata.to_dict()
        for utterance in self.utterances:
            row = meta.copy()
            row.update(utterance.to_dict())
            rows.append(row)
        return rows

    def session_row(self) -> dict:
        feature_names = list(CHATFeatures().to_dict().keys())
        agg = {name: 0 for name in feature_names}

        for utterance in self.child_utterances:
            for name, value in utterance.features.to_dict().items():
                agg[name] += value

        n = len(self.child_utterances)
        result = self.metadata.to_dict()
        result.update(
            {
                "total_utterances": len(self.utterances),
                "chi_utterances": n,
                "speaker_counts": json.dumps(self.speaker_counts),
                "mean_mlu_morphemes": self.mean_mlu_morphemes,
                "has_mor": any(u.annotations.has_mor for u in self.utterances),
                "has_gra": any(u.annotations.has_gra for u in self.utterances),
                "has_tim": any(u.annotations.has_tim for u in self.utterances),
                "has_err_tier": any(u.annotations.has_err_tier for u in self.utterances),
            }
        )
        result.update(agg)
        return result