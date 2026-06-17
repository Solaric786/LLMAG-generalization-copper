#!/usr/bin/env python3
"""
Reviewer 2, Comment 1: repeated-call consistency test for the LLM semantic stage.

This script:
1. Runs repeated DeepSeek API calls under fixed archived settings.
2. Measures raw-response consistency.
3. Applies deterministic schema, source-grounding, and duplicate checks.
4. Compares accepted source-matched labels with the frozen final reviewed mapping.
5. Writes CSV, XLSX, JSON, README, and ZIP outputs.

API key handling
----------------
Set the key outside the script:

    export DEEPSEEK_API_KEY="your_key_here"

In Kaggle, save a secret named DEEPSEEK_API_KEY. The script will use the
environment variable first and Kaggle Secrets as a fallback.

No API key is stored in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


REQUIRED_FILES = [
    "Supplementary_File_S1_DeepSeek_System_Prompt.txt",
    "Supplementary_File_S2_DeepSeek_User_Prompt.txt",
    "Supplementary_File_S3_Request_Settings.json",
    "Supplementary_File_S4_Legend_Semantic_Input.txt",
    "Supplementary_Data_S1_Final_Source_Unit_Mapping_165.csv",
]

ALLOWED_LABELS = {
    "Precambrian", "Cambrian", "Ordovician", "Silurian", "Devonian",
    "Carboniferous", "Permian", "Triassic", "Jurassic", "Cretaceous",
    "Paleocene", "Eocene", "Oligocene", "Miocene", "Pliocene",
    "Pleistocene", "Holocene", "Unassigned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reviewer 2 Comment 1 repeated-call consistency test."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Folder containing the required files or the input ZIP.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/Reviewer2_Comment1_Repeated_Call_Test"),
        help="Folder for generated outputs.",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=Path("/kaggle/working/Reviewer2_Comment1_Repeated_Call_Results.zip"),
        help="Path of the final results ZIP.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repeated API calls.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--secret-name",
        default="DEEPSEEK_API_KEY",
        help="Environment/Kaggle secret name containing the API key.",
    )
    return parser.parse_args()


def get_api_key(secret_name: str) -> str:
    """Read the API key from the environment, then Kaggle Secrets if available."""
    value = os.getenv(secret_name, "").strip()
    if value:
        return value

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        value = UserSecretsClient().get_secret(secret_name)
        if value and value.strip():
            return value.strip()
    except Exception:
        pass

    raise RuntimeError(
        f"API key not found. Set environment variable {secret_name!r} "
        f"or create a Kaggle secret with that exact name."
    )


def locate_inputs(input_root: Path, extract_dir: Path) -> dict[str, Path]:
    """Find required files directly or extract them from an uploaded ZIP."""
    found: dict[str, Path] = {}

    for name in REQUIRED_FILES:
        matches = list(input_root.rglob(name))
        if matches:
            found[name] = matches[0]

    if len(found) == len(REQUIRED_FILES):
        return found

    zip_candidates = list(
        input_root.rglob("Reviewer2_Comment1_Repeated_Call_Inputs*.zip")
    )
    if not zip_candidates:
        zip_candidates = list(input_root.rglob("*.zip"))

    extract_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_candidates:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            continue

    for name in REQUIRED_FILES:
        matches = list(extract_dir.rglob(name))
        if matches:
            found[name] = matches[0]

    missing = [name for name in REQUIRED_FILES if name not in found]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- "
            + "\n- ".join(missing)
            + "\nAdd Reviewer2_Comment1_Repeated_Call_Inputs.zip "
              "or the extracted files to the input directory."
        )

    return found


def safe_json_dump(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    work_root = args.output_root
    raw_dir = work_root / "raw_responses"
    extract_dir = work_root / "inputs"

    # Remove stale files from earlier failed runs.
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    paths = locate_inputs(args.input_root, extract_dir)

    system_prompt = paths[REQUIRED_FILES[0]].read_text(encoding="utf-8-sig")
    user_prompt = paths[REQUIRED_FILES[1]].read_text(encoding="utf-8-sig")
    settings = json.loads(paths[REQUIRED_FILES[2]].read_text(encoding="utf-8-sig"))
    legend_input = paths[REQUIRED_FILES[3]].read_text(encoding="utf-8-sig")
    source_df = pd.read_csv(paths[REQUIRED_FILES[4]], dtype=str).fillna("")

    required_columns = {
        "Abbreviation",
        "Final_Cartographic_Label",
        "Control_Label",
    }
    missing_columns = required_columns - set(source_df.columns)
    if missing_columns:
        raise ValueError(
            "Final mapping CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    model_name = settings["model"]
    max_attempts = int(settings.get("max_retries", 3))
    backoff_seconds = float(settings.get("retry_backoff_seconds", 2.0))
    retry_status_codes = set(
        settings.get("retry_status_codes", [429, 500, 502, 503, 504])
    )

    api_key = get_api_key(args.secret_name)

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=float(settings.get("timeout_seconds", 120)),
        max_retries=0,
    )

    source_codes = source_df["Abbreviation"].astype(str).tolist()
    source_code_set = set(source_codes)

    casefold_index: dict[str, list[str]] = defaultdict(list)
    for code in source_codes:
        casefold_index[code.casefold()].append(code)

    final_label_by_code = dict(
        zip(source_df["Abbreviation"], source_df["Final_Cartographic_Label"])
    )
    control_label_by_code = dict(
        zip(source_df["Abbreviation"], source_df["Control_Label"])
    )

    def standalone_in_legend(abbreviation: str) -> bool:
        if not abbreviation:
            return False
        pattern = rf"(?<![\w]){re.escape(abbreviation)}(?![\w])"
        return re.search(pattern, legend_input, flags=re.UNICODE) is not None

    def match_source_code(abbreviation: str) -> tuple[str, str]:
        if abbreviation in source_code_set:
            return abbreviation, "exact"

        candidates = casefold_index.get(abbreviation.casefold(), [])
        if len(candidates) == 1:
            return candidates[0], "unique_case_insensitive"
        if len(candidates) > 1:
            return "", "ambiguous_case_insensitive"
        return "", "no_match"

    def is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status in retry_status_codes
        return True

    def call_model(run_id: int):
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(settings.get("temperature", 0.0)),
                    top_p=float(settings.get("top_p", 1.0)),
                    max_tokens=int(settings.get("max_tokens", 4000)),
                    response_format=settings.get(
                        "response_format", {"type": "json_object"}
                    ),
                    stream=bool(settings.get("stream", False)),
                )

                raw = response.model_dump(mode="json")
                safe_json_dump(
                    raw,
                    raw_dir / f"run_{run_id:02d}_raw_response.json",
                )
                return response, raw, attempt, None

            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts or not is_retryable(exc):
                    break
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

        error_payload = {
            "run_id": run_id,
            "error_type": type(last_exc).__name__ if last_exc else "UnknownError",
            "error_message": str(last_exc) if last_exc else "Unknown error",
            "traceback": traceback.format_exc(),
        }
        safe_json_dump(
            error_payload,
            raw_dir / f"run_{run_id:02d}_error.json",
        )
        return None, None, max_attempts, last_exc

    run_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    raw_label_by_run: dict[int, dict[str, str]] = {}
    accepted_label_by_run: dict[int, dict[str, str]] = {}

    for run_id in range(1, args.runs + 1):
        print(f"Running call {run_id}/{args.runs} ...")
        response, raw_response, attempts_used, error = call_model(run_id)

        if error is not None:
            run_rows.append(
                {
                    "Run": run_id,
                    "API_success": False,
                    "Parse_success": False,
                    "Attempts_used": attempts_used,
                    "Returned_model": "",
                    "Finish_reason": "",
                    "Prompt_tokens": "",
                    "Completion_tokens": "",
                    "Total_tokens": "",
                    "Records_returned": 0,
                    "Schema_valid_records": 0,
                    "Source_grounded_records": 0,
                    "Deterministically_accepted_records": 0,
                    "Unsupported_abbreviation_records": 0,
                    "Source_code_matches": 0,
                    "Accepted_source_code_matches": 0,
                    "Accepted_OCR_only_records": 0,
                    "Raw_response_hash": "",
                    "Accepted_mapping_hash": "",
                    "Error": f"{type(error).__name__}: {error}",
                }
            )
            raw_label_by_run[run_id] = {}
            accepted_label_by_run[run_id] = {}
            continue

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or ""
        usage = getattr(response, "usage", None)

        parsed = None
        parse_error = ""
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or not isinstance(
                parsed.get("records"), list
            ):
                raise ValueError(
                    'Response must be one JSON object with a list under key "records".'
                )
            parse_success = True
        except Exception as exc:
            parse_success = False
            parse_error = f"{type(exc).__name__}: {exc}"

        raw_hash = (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content else ""
        )

        run_raw_map: dict[str, str] = {}
        run_accepted_map: dict[str, str] = {}
        seen_valid_abbreviations: set[str] = set()

        if parse_success:
            for record_index, rec in enumerate(parsed["records"], start=1):
                rec = rec if isinstance(rec, dict) else {}
                abbreviation = str(rec.get("Abbreviation", "")).strip()
                label = str(rec.get("Age_category", "")).strip()
                legend_text = str(rec.get("LegendText", ""))
                notes = str(rec.get("Notes", ""))
                confidence_raw = rec.get("Confidence", "")

                try:
                    confidence = float(confidence_raw)
                    confidence_valid = 0.0 <= confidence <= 1.0
                except Exception:
                    confidence = None
                    confidence_valid = False

                nonblank_abbreviation = bool(abbreviation)
                allowed_label = label in ALLOWED_LABELS
                required_text_fields = (
                    isinstance(rec.get("LegendText", ""), str)
                    and isinstance(rec.get("Notes", ""), str)
                )
                schema_valid = (
                    nonblank_abbreviation
                    and allowed_label
                    and confidence_valid
                    and required_text_fields
                )

                source_grounded = standalone_in_legend(abbreviation)
                matched_source_code, source_match_type = (
                    match_source_code(abbreviation)
                    if abbreviation else ("", "no_match")
                )

                duplicate_after_first_valid = False
                accepted = False

                if schema_valid and source_grounded:
                    if abbreviation in seen_valid_abbreviations:
                        duplicate_after_first_valid = True
                    else:
                        seen_valid_abbreviations.add(abbreviation)
                        accepted = True

                if abbreviation and abbreviation not in run_raw_map:
                    run_raw_map[abbreviation] = label

                if accepted:
                    run_accepted_map[abbreviation] = label

                final_reference_label = (
                    final_label_by_code.get(matched_source_code, "")
                    if matched_source_code else ""
                )
                agrees_with_final = (
                    bool(matched_source_code)
                    and bool(final_reference_label)
                    and label == final_reference_label
                )

                record_rows.append(
                    {
                        "Run": run_id,
                        "Record_index": record_index,
                        "Abbreviation": abbreviation,
                        "Age_category": label,
                        "Confidence": (
                            confidence if confidence is not None else ""
                        ),
                        "LegendText": legend_text,
                        "Notes": notes,
                        "Schema_valid": schema_valid,
                        "Source_grounded_in_OCR_input": source_grounded,
                        "Duplicate_after_first_valid": duplicate_after_first_valid,
                        "Deterministically_accepted": accepted,
                        "Source_match_type": source_match_type,
                        "Matched_source_abbreviation": matched_source_code,
                        "Control_label": (
                            control_label_by_code.get(matched_source_code, "")
                            if matched_source_code else ""
                        ),
                        "Final_reviewed_label": final_reference_label,
                        "Accepted_label_agrees_with_final": (
                            agrees_with_final if accepted else ""
                        ),
                    }
                )

        accepted_pairs = sorted(run_accepted_map.items())
        accepted_hash = hashlib.sha256(
            json.dumps(
                accepted_pairs,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        current_records = [r for r in record_rows if r["Run"] == run_id]

        run_rows.append(
            {
                "Run": run_id,
                "API_success": True,
                "Parse_success": parse_success,
                "Attempts_used": attempts_used,
                "Returned_model": getattr(response, "model", ""),
                "Finish_reason": finish_reason,
                "Prompt_tokens": (
                    getattr(usage, "prompt_tokens", "") if usage else ""
                ),
                "Completion_tokens": (
                    getattr(usage, "completion_tokens", "") if usage else ""
                ),
                "Total_tokens": (
                    getattr(usage, "total_tokens", "") if usage else ""
                ),
                "Records_returned": len(current_records),
                "Schema_valid_records": sum(
                    bool(r["Schema_valid"]) for r in current_records
                ),
                "Source_grounded_records": sum(
                    bool(r["Source_grounded_in_OCR_input"])
                    for r in current_records
                ),
                "Deterministically_accepted_records": sum(
                    bool(r["Deterministically_accepted"])
                    for r in current_records
                ),
                "Unsupported_abbreviation_records": sum(
                    bool(r["Schema_valid"])
                    and not bool(r["Source_grounded_in_OCR_input"])
                    for r in current_records
                ),
                "Source_code_matches": sum(
                    bool(r["Matched_source_abbreviation"])
                    for r in current_records
                ),
                "Accepted_source_code_matches": sum(
                    bool(r["Deterministically_accepted"])
                    and bool(r["Matched_source_abbreviation"])
                    for r in current_records
                ),
                "Accepted_OCR_only_records": sum(
                    bool(r["Deterministically_accepted"])
                    and not bool(r["Matched_source_abbreviation"])
                    for r in current_records
                ),
                "Raw_response_hash": raw_hash,
                "Accepted_mapping_hash": accepted_hash,
                "Error": parse_error,
            }
        )

        raw_label_by_run[run_id] = run_raw_map
        accepted_label_by_run[run_id] = run_accepted_map
        time.sleep(1.0)

    run_df = pd.DataFrame(run_rows)
    records_df = pd.DataFrame(record_rows)

    parse_success_runs = (
        run_df.loc[run_df["Parse_success"] == True, "Run"]
        .astype(int)
        .tolist()
    )

    all_abbreviations = sorted(
        {abbr for mapping in raw_label_by_run.values() for abbr in mapping}
    )
    raw_matrix = pd.DataFrame(
        {
            f"Run_{run_id:02d}": {
                abbr: raw_label_by_run.get(run_id, {}).get(abbr, "MISSING")
                for abbr in all_abbreviations
            }
            for run_id in parse_success_runs
        }
    ).rename_axis("Abbreviation").reset_index()

    accepted_abbreviations = sorted(
        {
            abbr
            for mapping in accepted_label_by_run.values()
            for abbr in mapping
        }
    )
    accepted_matrix = pd.DataFrame(
        {
            f"Run_{run_id:02d}": {
                abbr: accepted_label_by_run.get(run_id, {}).get(
                    abbr, "MISSING"
                )
                for abbr in accepted_abbreviations
            }
            for run_id in parse_success_runs
        }
    ).rename_axis("Abbreviation").reset_index()

    stability_rows: list[dict[str, Any]] = []

    for abbr in all_abbreviations:
        raw_values = [
            raw_label_by_run.get(run_id, {}).get(abbr, "MISSING")
            for run_id in parse_success_runs
        ]
        accepted_values = [
            accepted_label_by_run.get(run_id, {}).get(abbr, "MISSING")
            for run_id in parse_success_runs
        ]

        raw_present = [v for v in raw_values if v != "MISSING"]
        accepted_present = [v for v in accepted_values if v != "MISSING"]

        raw_mode, raw_mode_count = (
            Counter(raw_present).most_common(1)[0]
            if raw_present else ("", 0)
        )
        accepted_mode, accepted_mode_count = (
            Counter(accepted_present).most_common(1)[0]
            if accepted_present else ("", 0)
        )

        matched_code, match_type = match_source_code(abbr)
        final_reference_label = (
            final_label_by_code.get(matched_code, "")
            if matched_code else ""
        )

        stability_rows.append(
            {
                "Abbreviation": abbr,
                "Source_grounded_in_OCR_input": standalone_in_legend(abbr),
                "Source_match_type": match_type,
                "Matched_source_abbreviation": matched_code,
                "Final_reviewed_label": final_reference_label,
                "Runs_parsed": len(parse_success_runs),
                "Runs_returned": len(raw_present),
                "Return_rate": (
                    len(raw_present) / len(parse_success_runs)
                    if parse_success_runs else 0
                ),
                "Raw_mode_label": raw_mode,
                "Raw_label_agreement_when_returned": (
                    raw_mode_count / len(raw_present)
                    if raw_present else 0
                ),
                "Raw_stable_across_all_runs": (
                    bool(parse_success_runs)
                    and len(raw_present) == len(parse_success_runs)
                    and len(set(raw_present)) == 1
                ),
                "Runs_accepted": len(accepted_present),
                "Acceptance_rate": (
                    len(accepted_present) / len(parse_success_runs)
                    if parse_success_runs else 0
                ),
                "Accepted_mode_label": accepted_mode,
                "Accepted_label_agreement_when_accepted": (
                    accepted_mode_count / len(accepted_present)
                    if accepted_present else 0
                ),
                "Accepted_stable_across_all_runs": (
                    bool(parse_success_runs)
                    and len(accepted_present) == len(parse_success_runs)
                    and len(set(accepted_present)) == 1
                ),
                "Accepted_mode_agrees_with_final": (
                    accepted_mode == final_reference_label
                    if accepted_mode and final_reference_label else ""
                ),
            }
        )

    stability_df = pd.DataFrame(stability_rows)

    api_success_count = (
        int(run_df["API_success"].sum()) if not run_df.empty else 0
    )
    parse_success_count = (
        int(run_df["Parse_success"].sum()) if not run_df.empty else 0
    )

    raw_hashes = run_df.loc[
        run_df["API_success"] == True,
        "Raw_response_hash",
    ]
    accepted_hashes = run_df.loc[
        run_df["Parse_success"] == True,
        "Accepted_mapping_hash",
    ]

    raw_exact_mode_rate = (
        raw_hashes.value_counts().iloc[0] / len(raw_hashes)
        if len(raw_hashes) else 0
    )
    accepted_mapping_mode_rate = (
        accepted_hashes.value_counts().iloc[0] / len(accepted_hashes)
        if len(accepted_hashes) else 0
    )

    summary_items = [
        ("Planned calls", args.runs),
        ("Successful API calls", api_success_count),
        ("JSON parse successes", parse_success_count),
        ("JSON parse success rate", parse_success_count / args.runs),
        ("Exact raw-response mode rate", raw_exact_mode_rate),
        (
            "Post-validation accepted-mapping mode rate",
            accepted_mapping_mode_rate,
        ),
        (
            "Unique raw abbreviations across parsed calls",
            len(all_abbreviations),
        ),
        (
            "Raw abbreviations stable across all parsed calls",
            (
                int(stability_df["Raw_stable_across_all_runs"].sum())
                if not stability_df.empty else 0
            ),
        ),
        (
            "Raw abbreviations unstable or intermittently returned",
            (
                int((~stability_df["Raw_stable_across_all_runs"]).sum())
                if not stability_df.empty else 0
            ),
        ),
        (
            "Unique accepted abbreviations across parsed calls",
            len(accepted_abbreviations),
        ),
        (
            "Accepted abbreviations stable across all parsed calls",
            (
                int(stability_df["Accepted_stable_across_all_runs"].sum())
                if not stability_df.empty else 0
            ),
        ),
        (
            "Accepted abbreviations unstable or intermittently accepted",
            (
                int(
                    (
                        stability_df["Runs_accepted"].gt(0)
                        & ~stability_df["Accepted_stable_across_all_runs"]
                    ).sum()
                )
                if not stability_df.empty else 0
            ),
        ),
        (
            "Total records returned",
            int(run_df["Records_returned"].sum())
            if not run_df.empty else 0,
        ),
        (
            "Total unsupported abbreviation records rejected",
            int(run_df["Unsupported_abbreviation_records"].sum())
            if not run_df.empty else 0,
        ),
        (
            "Mean accepted records per parsed call",
            float(
                run_df.loc[
                    run_df["Parse_success"] == True,
                    "Deterministically_accepted_records",
                ].mean()
            )
            if parse_success_count else 0,
        ),
    ]
    summary_df = pd.DataFrame(summary_items, columns=["Metric", "Value"])

    run_df.to_csv(
        work_root / "Reviewer2_Comment1_Run_Summary.csv",
        index=False,
    )
    records_df.to_csv(
        work_root / "Reviewer2_Comment1_All_Returned_Records.csv",
        index=False,
    )
    stability_df.to_csv(
        work_root / "Reviewer2_Comment1_Record_Stability.csv",
        index=False,
    )
    raw_matrix.to_csv(
        work_root / "Reviewer2_Comment1_Raw_Label_Matrix.csv",
        index=False,
    )
    accepted_matrix.to_csv(
        work_root / "Reviewer2_Comment1_Accepted_Label_Matrix.csv",
        index=False,
    )
    summary_df.to_csv(
        work_root / "Reviewer2_Comment1_Overall_Summary.csv",
        index=False,
    )

    xlsx_path = (
        work_root / "Reviewer2_Comment1_Repeated_Call_Results.xlsx"
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Overall_Summary",
            index=False,
        )
        run_df.to_excel(
            writer,
            sheet_name="Run_Summary",
            index=False,
        )
        stability_df.to_excel(
            writer,
            sheet_name="Record_Stability",
            index=False,
        )
        records_df.to_excel(
            writer,
            sheet_name="All_Records",
            index=False,
        )
        raw_matrix.to_excel(
            writer,
            sheet_name="Raw_Label_Matrix",
            index=False,
        )
        accepted_matrix.to_excel(
            writer,
            sheet_name="Accepted_Label_Matrix",
            index=False,
        )

    readme = f"""Reviewer 2 Comment 1: repeated-call consistency test

Calls requested: {args.runs}
Archived model alias: {model_name}
Temperature: {settings.get('temperature')}
Top-p: {settings.get('top_p')}
Max tokens: {settings.get('max_tokens')}
JSON response format: {settings.get('response_format')}

The test distinguishes:
1. Raw response consistency.
2. Deterministically accepted records after closed-schema, source-grounding,
   and duplicate checks.
3. Agreement of accepted source-matched labels with the frozen final reviewed mapping.

Important:
- The final reviewed 165-record mapping is not overwritten by repeated API calls.
- No API key is stored in the script or output files.
"""
    (work_root / "README_Repeated_Call_Test.txt").write_text(
        readme,
        encoding="utf-8",
    )

    args.zip_output.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_output.exists():
        args.zip_output.unlink()

    with zipfile.ZipFile(
        args.zip_output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for file_path in work_root.rglob("*"):
            if file_path.is_file():
                zf.write(
                    file_path,
                    arcname=file_path.relative_to(work_root),
                )

    print("\nCompleted.")
    print(f"Results ZIP: {args.zip_output}")
    print("\nKey summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
