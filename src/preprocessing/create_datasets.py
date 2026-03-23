#create_datasets.py

##master

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from parse_data import CHAFileParser
from clean_text import TextCleaner


###SET UP LOGGING FIRST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REGISTRY = "file_info/files_master.csv"
DEFAULT_RAW_ROOT = ""
DEFAULT_OUTPUT   = "data/processed/"

CONTEXT_WINDOW_BEFORE = 2
CONTEXT_WINDOW_AFTER  = 1

#if one file to process
def process_file(file_path, registry_row, raw_root, parser, cleaner):

    raw_root = Path(raw_root)
    full_path = raw_root/file_path
    file_id   = registry_row.get("file_id", Path(file_path).stem)

    try:
        session = parser.parse(full_path, registry_row)
    except FileNotFoundError:
        log.warning(f"[{file_id}] File not found: {full_path}")
        return None
    except Exception as e:
        log.warning(f"[{file_id}] Parse error: {e}")
        return None
    
    cleaner.apply(session)
    meta = session.metadata.to_dict()
    utterance_rows = []

    for utt in session.utterances:
        row = utt.to_dict()
        row.update({
            "corpus": meta.get("corpus"),
            "label_binary": meta.get("label_binary"),
            "label": meta.get("label"),
            "age": meta.get("age"),
            "sex": meta.get("sex"),
            "session_types": meta.get("session_types"),
            "pid": meta.get("pid"),
            "multi_child": meta.get("multiple_kids"),
            "examiner_codes": meta.get("examiner_codes"),
            "parent_codes": meta.get("parent_codes"),})
        utterance_rows.append(row)

    context_rows = []
    for utt in session.child_utterances:
        context_rows.append({
            "file_id": utt.file_id,
            "utterance_id": utt.utterance_id,
            "utterance_index": utt.utterance_index,
            "speaker": utt.speaker,
            "utterance_raw": utt.text_raw,
            "utterance_clean": utt.text_clean,
            "corpus": meta.get("corpus"),
            "label_binary": meta.get("label_binary"),
            "age": meta.get("age"),
            "context_before": json.dumps(utt.context_before),
            "context_after": json.dumps(utt.context_after),})

    session_row = session.session_row()
    warnings = list(session.metadata.header_warnings)

    for utt in session.utterances:
        for w in utt.parse_warnings:
            if len(warnings) < 1000:
                warnings.append(f"utt {utt.utterance_index}: {w}")

    file_data = {
        "utterance_rows": utterance_rows,
        "context_rows":   context_rows,
        "session_row":    session_row,
        "warnings":       warnings}
    
    return file_data


def build_all_datasets(registry_path, raw_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(registry_path, low_memory=False)
    v1 = registry[registry["include_v1"] == 1].copy()
    log.info(f"Registry loaded — {len(v1)} files marked include_v1=True")
    log.info(f"Label distribution:\n{v1['label_binary'].value_counts().to_string()}")

    cha_parser = CHAFileParser(
        prior_context=CONTEXT_WINDOW_BEFORE,
        later_context=CONTEXT_WINDOW_AFTER,
    )
    cleaner = TextCleaner()

    all_utterances = []
    all_context_windows = []
    all_sessions = []
    failed_files = []
    warning_log = []

    for i, (_, row) in enumerate(v1.iterrows(), 1):
        file_id = row.get("file_id", f"file_{i}")
        file_path = row.get("file_path", "")

        if not file_path:
            log.warning(f"[{file_id}] Missing file_path in registry")
            failed_files.append(file_id)
            continue


        log.info(f"[{i:>4}/{len(v1)}] {file_id}")

        result = process_file(file_path, row.to_dict(), raw_root, cha_parser, cleaner)

        if result is None:
            failed_files.append(file_id)
            continue

        all_utterances.extend(result["utterance_rows"])
        all_context_windows.extend(result["context_rows"])
        all_sessions.append(result["session_row"])

        for w in result["warnings"]:
            warning_log.append({"file_id": file_id, "warning": w})

    log.info(f"Pipeline complete — {len(all_sessions)} processed, {len(failed_files)} failed")

    df_all = pd.DataFrame(all_utterances)
    df_all.to_csv(output_dir/"all_utterances.csv", index=False)
    log.info(f"all_utterances.csv {len(df_all):>7,} rows")

    df_chi = df_all[df_all["is_target_child"]].copy()
    df_chi.to_csv(output_dir/"child_utterances.csv", index=False)
    log.info(f"child_utterances.csv {len(df_chi):>7,} rows")

    df_ctx = pd.DataFrame(all_context_windows)
    df_ctx.to_csv(output_dir/"child_context_windows.csv", index=False)
    log.info(f"child_context_windows.csv {len(df_ctx):>7,} rows")

    df_session = pd.DataFrame(all_sessions)
    df_session.to_csv(output_dir/"session_level.csv", index=False)
    log.info(f"session_level.csv {len(df_session):>7,} rows")

    if warning_log:
        pd.DataFrame(warning_log).to_csv(output_dir/"pipeline_warnings.csv", index=False)
        log.info(f"pipeline_warnings.csv {len(warning_log):>7,} warnings")

    if failed_files:
        (output_dir/"failed_files.txt").write_text("\n".join(failed_files))
        log.warning(f"{len(failed_files)} files failed please see failed_files.txt")

    log.info("finished")

    info = {
        "all_utterances": df_all,
        "child_utterances": df_chi,
        "child_context_windows": df_ctx,
        "session_level": df_session}
    
    return info

def main():
    arg_parser = argparse.ArgumentParser(
        description="Build NLP datasets from TalkBank .cha files.")
    arg_parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    arg_parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    arg_parser.add_argument("--output",   default=DEFAULT_OUTPUT)
    args = arg_parser.parse_args()

    if not Path(args.registry).exists():
        log.error(f"Registry not found: {args.registry}")
        sys.exit(1)

    build_all_datasets(registry_path = args.registry, raw_root = args.raw_root, output_dir = args.output,)


if __name__ == "__main__":
    main()
