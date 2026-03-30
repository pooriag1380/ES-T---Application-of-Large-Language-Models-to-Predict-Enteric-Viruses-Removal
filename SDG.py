"""
sdg_ehi.py
Single-facility (ICA) synthetic data generation using OpenAI Responses API + Structured Outputs.

What it does:
- Loads your ICA Excel file
- Drops non-data rows (e.g., "Physico-chemical parameters")
- Converts seasons to {Winter=1, Spring=2, Summer=3, Fall=4}
- Detects BDL markers like "X_isBDL"
- Adds explicit flags: <col>_is_bdl (bool)
- Applies LOD rules you provided
- Builds a facility-specific data dictionary (ranges, percentiles, BDL rates)
- Calls OpenAI API to generate schema-valid synthetic rows (but DOES NOT generate unless you run it)
- Validates constraints and retries if needed
- Saves synthetic CSV + saved spec/prompt for reproducibility

Install:
  pip install openai pydantic pandas numpy scipy openpyxl

Run:
  python sdg_ehi_structured.py --input "/mnt/data/1_Clean_Raw_Data_EHI.xlsx" --n 100 --outdir "sdg_ehi_out"

IMPORTANT:
- Paste your API key into OPENAI_API_KEY below (as requested).
- Keep this file private and do NOT commit to git if it contains a real key.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, create_model
from typing import Any

from openai import OpenAI


# =========================
# 1) USER CONFIG
# =========================

# (Requested) Hardcode API key here:
OPENAI_API_KEY = "" #Confidential API

# Model config
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.8

# Season encoding
SEASON_MAP = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
INV_SEASON_MAP = {v: k for k, v in SEASON_MAP.items()}

# BDL marker recognition
BDL_PATTERN = re.compile(r"^\s*(x[_\s-]*isbdl|bdl|<\s*bdl|#)\s*$", flags=re.IGNORECASE)

# Your detection limits (LOD)
LOD_TSS_MG_L = 0.5
LOD_NH3_MG_L = 0.007
LOD_PFU_ML = 0.01
LOD_COP_ML = 0.047

# For converting BDL-marked cells to numeric values in the *real* dataset:
# Common simple substitution: LOD/2
BDL_SUBSTITUTION_FRACTION = 0.5


# Validation thresholds
MIN_ACCEPTABLE_VALID_FRAC = 0.85

# Recommended: 8–15 calls is much more realistic for strict LOD/BDL constraints
MAX_RETRIES = 10

BATCH_SIZE = 100

# NEW: avoid tiny remainder calls (prevents the “10-row batch collapses” problem)
MIN_REQUEST_BATCH = 40

# NEW: if still short after retries, save what we have instead of raising
ALLOW_PARTIAL_OUTPUT = True


@dataclass
class SDGSettings:
    model: str = DEFAULT_MODEL        # override via env/cli if needed
    temperature: float = DEFAULT_TEMPERATURE
    batch_size: int = BATCH_SIZE
    max_retries: int = MAX_RETRIES
    min_valid_frac: float = MIN_ACCEPTABLE_VALID_FRAC
    avoid_exact_copies: bool = True
    min_request_batch: int = MIN_REQUEST_BATCH
    allow_partial_output: bool = ALLOW_PARTIAL_OUTPUT
    avoid_exact_copies: bool = True

# =========================
# 2) HELPERS
# =========================

def is_bdl_cell(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    if isinstance(x, str) and BDL_PATTERN.match(x):
        return True
    return False


def drop_nondata_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your file includes rows like:
      Season = NaN, Temp_inf = "Physico-chemical parameters"
    Keep only rows whose Season is a valid season string.
    """
    df = df.copy()
    df["Name of Facility"] = df["Name of Facility"].ffill()
    df["Season"] = df["Season"].ffill()
    df = df[df["Season"].isin(SEASON_MAP.keys())].copy()
    return df.reset_index(drop=True)


def get_lod_for_column(col: str) -> Optional[float]:
    """
    Apply your LOD rules based on column naming patterns in the EHI file.
    Columns include units in parentheses, e.g.:
      "TSS_inf (mg/L)", "Somatic_eff (PFU/ml)", "PMMoV_inf (cop/ml)"
    """
    c = col.lower()

    # TSS (mg/L)
    if "tss_" in c and "(mg/l)" in c:
        return LOD_TSS_MG_L

    # Ammonia (mg/L)
    if "ammonia_" in c and "(mg/l)" in c:
        return LOD_NH3_MG_L

    # PFU/ml (Somatic, Fspecific)
    if ("somatic_" in c or "fspecific_" in c) and "(pfu/ml)" in c:
        return LOD_PFU_ML

    # cop/ml viruses (PMMoV, ToBRFV, CrA, Ade/Ent/Nor1/Nor2, etc.)
    if "(cop/ml)" in c:
        return LOD_COP_ML

    return None


def summarize_numeric(x: pd.Series) -> Dict[str, Any]:
    vals = x.dropna().astype(float)
    if len(vals) == 0:
        return {"n": 0}
    q = vals.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    return {
        "n": int(len(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
        "p05": float(q.get(0.05)),
        "p25": float(q.get(0.25)),
        "p50": float(q.get(0.5)),
        "p75": float(q.get(0.75)),
        "p95": float(q.get(0.95)),
    }


# =========================
# 3) PREPROCESS (REAL DATA)
# =========================

def preprocess_real_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Optional[float]]]:
    """
    - Converts Season to integer code
    - For each numeric column:
        - create <col>_is_bdl (bool) if any BDL markers OR if values < LOD
        - convert to float
        - if BDL marker -> numeric substitution = LOD * BDL_SUBSTITUTION_FRACTION
    Returns:
      df_num: cleaned numeric df with *_is_bdl flags
      synth_cols: columns to synthesize (all except Name of Facility)
      lod_map: column -> lod (or None)
    """
    df = df.copy()

    # Convert seasons to int
    df["Season"] = df["Season"].map(SEASON_MAP).astype(int)

    # Synthesize all measurement columns + Season, but keep facility name as constant column.
    synth_cols = [c for c in df.columns if c not in ["Name of Facility"]]

    # Build LOD map for measurement columns (excluding Season)
    lod_map: Dict[str, Optional[float]] = {}
    for c in synth_cols:
        if c == "Season":
            lod_map[c] = None
        else:
            lod_map[c] = get_lod_for_column(c)

    # Convert each non-Season column to numeric + BDL handling
    bdl_flag_cols: List[str] = []
    for c in synth_cols:
        if c == "Season":
            continue

        lod = lod_map[c]
        flag_col = f"{c}_is_bdl"

        # mark explicit BDL strings
        bdl_str = df[c].apply(is_bdl_cell)

        # convert to numeric (non-parsable -> NaN)
        numeric = pd.to_numeric(df[c].mask(bdl_str), errors="coerce")

        # mark numeric values < LOD as BDL too (only if LOD exists)
        bdl_num = pd.Series(False, index=df.index)
        if lod is not None:
            bdl_num = numeric.notna() & (numeric < lod)

        # final BDL flag
        bdl_flag = (bdl_str | bdl_num)
        if bdl_flag.any():
            df[flag_col] = bdl_flag.astype(bool)
            bdl_flag_cols.append(flag_col)
        else:
            df[flag_col] = False
            bdl_flag_cols.append(flag_col)

        # substitute explicit BDL markers to numeric = LOD * fraction (if LOD exists)
        if lod is not None:
            numeric = numeric.mask(bdl_str, lod * BDL_SUBSTITUTION_FRACTION)

        df[c] = numeric

    return df, synth_cols, lod_map


# =========================
# 4) FACILITY SPEC (DATA DICTIONARY)
# =========================

def build_facility_spec(df_num: pd.DataFrame, synth_cols: List[str], lod_map: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """
    Builds a scientifically explainable "data dictionary" used to guide GPT generation.
    """
    allowed_seasons = sorted(df_num["Season"].dropna().unique().tolist())

    colspec: Dict[str, Any] = {}
    for c in synth_cols:
        if c == "Season":
            continue

        lod = lod_map.get(c, None)
        summ = summarize_numeric(df_num[c])

        colspec[c] = {
            "type": "float",
            "allow_null": True,
            "lod": lod,
            "summary": summ,
            "bdl_rate": float(df_num[f"{c}_is_bdl"].mean()) if f"{c}_is_bdl" in df_num.columns else 0.0,
        }

        # Extra safety constraints (optional)
        if c.lower().startswith("ph_"):
            colspec[c]["hard_min"] = 0.0
            colspec[c]["hard_max"] = 14.0
        else:
            # hard bounds from real data if available
            if summ.get("n", 0) > 0:
                colspec[c]["hard_min"] = summ["min"]
                colspec[c]["hard_max"] = summ["max"]

    return {
        "facility_code": "UCD",
        "facility_name": str(df_num["Name of Facility"].iloc[0]),
        "allowed_seasons": allowed_seasons,
        "season_encoding": "int (Winter=1, Spring=2, Summer=3, Fall=4)",
        "columns": colspec,
        "bdl_policy": f"If *_is_bdl is true, value should be in [0, LOD]. Real-data substitution uses {BDL_SUBSTITUTION_FRACTION}*LOD.",
        "n_real_rows": int(len(df_num)),
    }


# =========================
# 5) STRUCTURED OUTPUT SCHEMA
# =========================
def build_batch_json_schema(synth_cols: list[str], allowed_seasons: list[int], n_rows: int) -> dict:
    # Row schema (inline, ref-free)
    row_props = {
        "Season": {"type": "integer", "enum": allowed_seasons},
    }

    for c in synth_cols:
        if c in ["Name of Facility", "Season"]:
            continue

        # value can be number or null
        row_props[c] = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        # bdl flag is boolean
        row_props[f"{c}_is_bdl"] = {"type": "boolean"}

    row_required = list(row_props.keys())

    row_schema = {
        "type": "object",
        "properties": row_props,
        "required": row_required,
        "additionalProperties": False,
    }

    # Batch schema
    batch_schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": row_schema,
                "minItems": n_rows,
                "maxItems": n_rows,
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }

    return batch_schema

def make_row_model(synth_cols: List[str]) -> type[BaseModel]:
    """
    Pydantic model for one synthetic row.
    Includes:
      - Season (int)
      - each measurement column (float | None)
      - each measurement column's <col>_is_bdl (bool)
    """
    fields: Dict[str, Tuple[Any, Any]] = {}
    fields["Season"] = (int, Field(..., description="Season code: Winter=1, Spring=2, Summer=3, Fall=4"))

    for c in synth_cols:
        if c in ["Name of Facility", "Season"]:
            continue
        fields[c] = (Optional[float], Field(default=None))
        fields[f"{c}_is_bdl"] = (bool, Field(default=False))

    Row = create_model("SyntheticRowUCD", **fields)  # type: ignore
    return Row


def make_batch_model(RowModel: type[BaseModel]) -> type[BaseModel]:
    class SyntheticBatch(BaseModel):
        rows: list[Any]

    return SyntheticBatch


# =========================
# 6) PROMPT + VALIDATION
# =========================

def build_prompt(spec: Dict[str, Any], n_rows: int) -> List[Dict[str, str]]:
    spec_json = json.dumps(spec, indent=2)

    system = (
        "You generate synthetic environmental engineering tabular data.\n"
        "Rules:\n"
        "- Output must match the JSON schema (Structured Outputs will enforce it).\n"
        "- Generate novel rows; do not copy real rows.\n"
        "- Season must be one of allowed seasons.\n"
        "- For each column with an LOD:\n"
        "    * Use bdl_rate to decide how often *_is_bdl is true.\n"
        "    * If *_is_bdl is true, set value within [0, LOD].\n"
        "    * If *_is_bdl is false, set value >= LOD.\n"
        "- Keep values realistic and consistent with the provided summaries (min/max/percentiles).\n"
        "- Avoid overly rounded numbers.\n"
    )

    user = (
        f"Generate exactly {n_rows} synthetic rows for facility UCD.\n\n"
        "Use this facility data dictionary (summaries, bounds, and detection limits):\n"
        f"{spec_json}\n\n"
        "Return a JSON object with a single key 'rows' containing the list of row objects."
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def validate_rows(rows: List[Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    """
    Hard validation:
    - Season allowed
    - For each column:
        - if *_is_bdl: value in [0, LOD] (if LOD exists)
        - else: value >= LOD (if LOD exists)
        - value within hard_min/hard_max if provided
    """
    allowed_seasons = set(spec["allowed_seasons"])
    colspec = spec["columns"]

    valid: List[Dict[str, Any]] = []
    invalid: List[Tuple[int, str]] = []

    for i, r in enumerate(rows):
        s = r.get("Season")
        if s not in allowed_seasons:
            invalid.append((i, f"Invalid Season={s}"))
            continue

        bad = False
        for c, cs in colspec.items():
            val = r.get(c, None)
            is_bdl = r.get(f"{c}_is_bdl", False)
            lod = cs.get("lod", None)

            # LOD logic
            if lod is not None and val is not None:
                try:
                    fv = float(val)
                except Exception:
                    invalid.append((i, f"Non-numeric {c}={val}"))
                    bad = True
                    break

                if is_bdl:
                    if fv < 0 or fv > lod:
                        invalid.append((i, f"BDL out of range {c}={fv} not in [0,{lod}]"))
                        bad = True
                        break
                else:
                    if fv < lod:
                        invalid.append((i, f"Non-BDL below LOD {c}={fv} < {lod}"))
                        bad = True
                        break

                # hard bounds
                hmin = cs.get("hard_min", None)
                hmax = cs.get("hard_max", None)
                if hmin is not None and fv < hmin:
                    invalid.append((i, f"Below hard_min {c}={fv} < {hmin}"))
                    bad = True
                    break
                if hmax is not None and fv > hmax:
                    invalid.append((i, f"Above hard_max {c}={fv} > {hmax}"))
                    bad = True
                    break

            # If value is None, allow (but keep is_bdl possibly false/true)
            # You can tighten this rule if you want fewer missing values.

        if bad:
            continue

        valid.append(r)

    return valid, invalid


# =========================
# 7) GENERATION
# =========================
def generate_synthetic_ucd(
    client: OpenAI,
    df_real: pd.DataFrame,
    synth_cols: List[str],
    spec: Dict[str, Any],
    n_total: int,
    settings: SDGSettings,
    outdir: Path,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    attempts = 0
    last_invalid_reasons: List[str] = []

    audit_dir = outdir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    def _summarize_invalid(invalid: List[Tuple[int, str]], k: int = 8) -> List[str]:
        # keep unique reasons, preserve order
        seen = set()
        out = []
        for _, msg in invalid:
            if msg not in seen:
                seen.add(msg)
                out.append(msg)
            if len(out) >= k:
                break
        return out

    while len(all_rows) < n_total and attempts < settings.max_retries:
        attempts += 1
        remaining = n_total - len(all_rows)

        # ---- FIX #1: avoid tiny remainder batches ----
        # Request at least min_request_batch rows (unless batch_size is smaller).
        n_request = min(settings.batch_size, max(remaining, settings.min_request_batch))

        messages = build_prompt(spec, n_request)

        # ---- Optional: feed back the last invalid reasons to improve compliance ----
        if last_invalid_reasons:
            messages[-1]["content"] += (
                "\n\nIMPORTANT: Your previous output had invalid rows due to:\n- "
                + "\n- ".join(last_invalid_reasons)
                + "\nGenerate rows that satisfy ALL constraints (LOD/BDL + hard min/max)."
            )

        # Save prompt for audit
        with open(audit_dir / f"prompt_attempt_{attempts}.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

        schema = build_batch_json_schema(
            synth_cols=synth_cols,
            allowed_seasons=spec["allowed_seasons"],
            n_rows=n_request,
        )

        resp = client.responses.create(
            model=settings.model,
            input=messages,
            temperature=settings.temperature,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "SyntheticBatch",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        # Parse response safely
        try:
            payload = json.loads(resp.output_text)
            rows = payload["rows"]
        except Exception as e:
            # log raw output for debugging and continue
            with open(audit_dir / "raw_output_failures.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- attempt {attempts} parse failure ---\n{str(e)}\n{resp.output_text}\n")
            continue

        valid, invalid = validate_rows(rows, spec)
        valid_frac = (len(valid) / len(rows)) if rows else 0.0

        # Log attempt
        log_rec = {
            "attempt": attempts,
            "requested": n_request,
            "received": len(rows),
            "valid": len(valid),
            "invalid": len(invalid),
            "valid_frac": valid_frac,
            "accepted_so_far": len(all_rows),
            "remaining_after": max(n_total - (len(all_rows) + len(valid)), 0),
        }
        with open(audit_dir / "generation_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec) + "\n")

        # ---- FIX #2: always keep valid rows ----
        all_rows.extend(valid)

        # ---- FIX #3: do NOT discard the batch just because valid_frac is low ----
        # Instead, keep trying, and also pass feedback next time.
        if valid_frac < settings.min_valid_frac:
            last_invalid_reasons = _summarize_invalid(invalid, k=8)
            continue
        else:
            last_invalid_reasons = []

    # Build output
    if len(all_rows) < n_total:
        if settings.allow_partial_output:
            print(f"[WARN] Only generated {len(all_rows)}/{n_total} valid rows after {attempts} attempts. Saving partial output.")
            df_synth = pd.DataFrame(all_rows)
        else:
            raise RuntimeError(f"Only generated {len(all_rows)}/{n_total} valid rows after {attempts} attempts.")
    else:
        df_synth = pd.DataFrame(all_rows[:n_total])

    df_synth["Name of Facility"] = spec["facility_name"]
    df_synth["Season_str"] = df_synth["Season"].map(INV_SEASON_MAP)

    return df_synth



# =========================
# 8) MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="1_Clean_Raw_Data_UCD.xlsx")
    ap.add_argument("--outdir", type=str, default="SDG_UCD_out")
    ap.add_argument("--n", type=int, default=100, help="Synthetic rows to generate")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    if not OPENAI_API_KEY or "PASTE_" in OPENAI_API_KEY:
        raise ValueError("Please paste your real API key into OPENAI_API_KEY in the script.")

    settings = SDGSettings(
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # OpenAI client using hardcoded key (requested)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load and clean
    df_raw = pd.read_excel(input_path)
    df_data = drop_nondata_rows(df_raw)

    # Preprocess BDL + numeric conversion
    df_num, synth_cols, lod_map = preprocess_real_data(df_data)

    # Build spec
    spec = build_facility_spec(df_num, synth_cols, lod_map)

    audit_dir = outdir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open(audit_dir / "facility_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    # Generate synthetic (only when you run this script)
    df_synth = generate_synthetic_ucd(
        client=client,
        df_real=df_num,
        synth_cols=synth_cols,
        spec=spec,
        n_total=args.n,
        settings=settings,
        outdir=outdir,
    )

    out_csv = outdir / "synthetic_UCD.csv"
    df_synth.to_csv(out_csv, index=False)
    print(f"[OK] Saved synthetic data: {out_csv}")


if __name__ == "__main__":
    main()
