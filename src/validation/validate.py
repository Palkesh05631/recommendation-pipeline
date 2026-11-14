"""Stage 2 - Data validation.

Automated checks on users, items and interactions:
  missing values | duplicates | data types | invalid ratings | invalid timestamps |
  referential integrity | invalid ages

Invalid rows are NOT silently dropped: they are written to data/errors/<table>_errors.csv with
an `error_reason` column. Valid rows go to data/processed/<table>_valid.parquet.
A Pandera schema acts as a final contract check on the validated interactions.
A validation report is written to reports/validation_report.{json,md}.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config

log = get_logger("validation")


class Checker:
    """Collects boolean-mask checks, per-check counts and a per-row error reason."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reasons = pd.Series("", index=df.index)
        self.counts: dict[str, int] = {}

    def flag(self, name: str, mask: pd.Series) -> None:
        mask = mask.fillna(False).astype(bool)
        self.counts[name] = int(mask.sum())
        self.reasons.loc[mask] = self.reasons.loc[mask] + name + ";"

    def split(self):
        bad = self.reasons != ""
        valid = self.df[~bad].copy()
        errors = self.df[bad].assign(error_reason=self.reasons[bad].str.rstrip(";"))
        return valid, errors


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _summary(total, valid, errors, counts):
    return {"total_rows": int(total), "valid_rows": int(len(valid)),
            "error_rows": int(len(errors)), "checks": counts}


# --------------------------------------------------------------------------- per-table checks
def validate_users(users: pd.DataFrame, v: dict):
    c = Checker(users)
    uid, age = _num(users["user_id"]), _num(users["age"])
    c.flag("missing_user_id", users["user_id"].isna())
    c.flag("invalid_dtype_user_id", users["user_id"].notna() & uid.isna())
    c.flag("invalid_dtype_age", users["age"].notna() & age.isna())
    c.flag("invalid_age", age.notna() & ~age.between(v["age_min"], v["age_max"]))
    c.flag("duplicate_user_id", uid.notna() & uid.duplicated(keep="first"))
    valid, errors = c.split()
    valid = valid.assign(user_id=_num(valid["user_id"]).astype("int64"), age=_num(valid["age"]))
    return valid, errors, c.counts


def validate_items(items: pd.DataFrame, v: dict):
    c = Checker(items)
    iid = _num(items["item_id"])
    c.flag("missing_item_id", items["item_id"].isna())
    c.flag("invalid_dtype_item_id", items["item_id"].notna() & iid.isna())
    c.flag("missing_title", items["title"].isna() | (items["title"].astype(str).str.strip() == ""))
    c.flag("duplicate_item_id", iid.notna() & iid.duplicated(keep="first"))
    g = items[GENRES].apply(_num)
    c.flag("invalid_genre_flag", (g.notna() & ~g.isin([0, 1])).any(axis=1))
    valid, errors = c.split()
    valid = valid.assign(item_id=_num(valid["item_id"]).astype("int64"))
    for col in GENRES:
        valid[col] = _num(valid[col]).fillna(0).astype("int64")
    return valid, errors, c.counts


def validate_interactions(inter: pd.DataFrame, valid_users, valid_items, v: dict):
    c = Checker(inter)
    uid, iid = _num(inter["user_id"]), _num(inter["item_id"])
    rating, ts = _num(inter["rating"]), _num(inter["timestamp"])

    for col in ["user_id", "item_id", "rating", "timestamp"]:
        c.flag(f"missing_{col}", inter[col].isna())
    for col, parsed in [("user_id", uid), ("item_id", iid), ("rating", rating), ("timestamp", ts)]:
        c.flag(f"invalid_dtype_{col}", inter[col].notna() & parsed.isna())
    c.flag("invalid_rating", rating.notna() & ~rating.between(v["rating_min"], v["rating_max"]))
    now = time.time()
    c.flag("invalid_timestamp", ts.notna() & ((ts < v["min_timestamp"]) | (ts > now)))
    c.flag("orphan_user_id", uid.notna() & ~uid.isin(valid_users["user_id"]))
    c.flag("orphan_item_id", iid.notna() & ~iid.isin(valid_items["item_id"]))
    # Duplicates are judged only among otherwise-valid rows, so a corrupted copy of a row
    # never causes the genuine original to be thrown away.
    clean = c.reasons == ""
    keys = pd.DataFrame({"u": uid, "i": iid, "t": ts})
    c.flag("duplicate_interaction", clean & keys.duplicated(keep="first"))

    valid, errors = c.split()
    valid = pd.DataFrame({"user_id": _num(valid["user_id"]).astype("int64"),
                          "item_id": _num(valid["item_id"]).astype("int64"),
                          "rating": _num(valid["rating"]).astype("float64"),
                          "timestamp": _num(valid["timestamp"]).astype("int64")})
    return valid, errors, c.counts


def _pandera_contract(valid: pd.DataFrame, v: dict) -> None:
    """Final schema contract on validated interactions (fails loudly if violated)."""
    try:
        import pandera.pandas as pa
    except ImportError:                                            # older pandera
        import pandera as pa
    schema = pa.DataFrameSchema(
        {"user_id": pa.Column(int, pa.Check.gt(0)),
         "item_id": pa.Column(int, pa.Check.gt(0)),
         "rating": pa.Column(float, pa.Check.in_range(v["rating_min"], v["rating_max"])),
         "timestamp": pa.Column(int, pa.Check.ge(v["min_timestamp"]))},
        unique=["user_id", "item_id", "timestamp"], strict=True)
    schema.validate(valid, lazy=True)


# --------------------------------------------------------------------------- report
def _write_report(report: dict, reports_dir) -> None:
    (reports_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Data Validation Report", "",
             "| Table | Total rows | Valid | Rejected | % rejected |", "|---|---|---|---|---|"]
    for t, r in report.items():
        pct = 100 * r["error_rows"] / max(r["total_rows"], 1)
        lines.append(f"| {t} | {r['total_rows']} | {r['valid_rows']} | {r['error_rows']} | {pct:.2f}% |")
    for t, r in report.items():
        lines += ["", f"## {t}: violations per check", "", "| Check | Rows violating |", "|---|---|"]
        lines += [f"| {k} | {n} |" for k, n in r["checks"].items()]
    lines += ["", "_A row can violate several checks, so per-check counts may sum to more than "
              "the number of rejected rows. Rejected rows are stored in `data/errors/` with an "
              "`error_reason` column._"]
    (reports_dir / "validation_report.md").write_text("\n".join(lines))


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw, proc = get_path(cfg, "raw"), get_path(cfg, "processed")
    err, rep = get_path(cfg, "errors"), get_path(cfg, "reports")
    v = cfg["validation"]

    users = pd.read_csv(raw / "users.csv", dtype=str)
    items = pd.read_csv(raw / "items.csv", dtype=str)
    inter = pd.read_csv(raw / "interactions.csv", dtype=str)

    vu, eu, cu = validate_users(users, v)
    vi, ei, ci = validate_items(items, v)
    vx, ex, cx = validate_interactions(inter, vu, vi, v)
    _pandera_contract(vx, v)

    for name, valid, errors in [("users", vu, eu), ("items", vi, ei), ("interactions", vx, ex)]:
        valid.to_parquet(proc / f"{name}_valid.parquet", index=False)
        errors.to_csv(err / f"{name}_errors.csv", index=False)

    report = {"users": _summary(len(users), vu, eu, cu),
              "items": _summary(len(items), vi, ei, ci),
              "interactions": _summary(len(inter), vx, ex, cx)}
    _write_report(report, rep)
    log.info("Validation done: %s", {t: (r["valid_rows"], r["error_rows"]) for t, r in report.items()})
    return report


if __name__ == "__main__":
    run()
