"""Builds reports/final_report.pdf from the generated result files (run after the pipeline).

Inputs (all in reports/): validation_report.json, split_summary.json, model_comparison.csv,
architecture.png, and the screenshots prefect_run.png / mlflow_runs.png (matched case-insensitively).
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO_URL = "https://github.com/Palkesh05631/recommendation-pipeline"
R = Path(__file__).resolve().parents[1] / "reports"
val = json.loads((R / "validation_report.json").read_text())
split = json.loads((R / "split_summary.json").read_text())
cmp_ = pd.read_csv(R / "model_comparison.csv")
manifest = json.loads((R.parent / "data/raw/ingestion_manifest.json").read_text())


def find_image(*names):
    wanted = {n.lower() for n in names}
    for f in R.iterdir():
        if f.name.lower() in wanted:
            return f
    return None


ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, alignment=0)
cap = ParagraphStyle("cap", parent=small, fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=6)
bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)

def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="\u2022")
def link(url): return f'<link href="{url}" color="blue">{url}</link>'

def table(data, widths):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    return t

def picture(path, width_cm):
    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    img.hAlign = "CENTER"
    return img

k = 10
p, s = cmp_.iloc[0], cmp_.iloc[1]
inter = val["interactions"]
real = manifest["source"].startswith("movielens")

if real:
    data_text = (
        "The pipeline uses the public <b>MovieLens-100K</b> dataset (943 users, 1,682 movies, 100,000 ratings, 19 genre flags), "
        "downloaded automatically at run time. To demonstrate the validation stage, a small set of known-bad rows "
        f"(duplicates, out-of-range or non-numeric ratings, orphan ids, impossible timestamps; {manifest['rows']['interactions']:,} interaction rows in total after injection) "
        "is appended to the raw data (<font face='Courier'>data.inject_bad_rows</font>, switchable in the config). "
        "If the download is unavailable, the pipeline falls back to a documented synthetic MovieLens-like generator.")
else:
    data_text = (
        f"Results were generated with the pipeline's documented synthetic fallback ({manifest['source']}: {manifest['rows']['users']} users, "
        f"{manifest['rows']['items']} items, {manifest['rows']['interactions']:,} interactions after bad-row injection), which reproduces the MovieLens "
        "schema. Setting <font face='Courier'>data.source: movielens</font> reruns everything on the real dataset with no code changes.")

rmse_better_baseline = s["rmse"] > p["rmse"]
rmse_text = (
    f"For explicit rating prediction the simple item-mean baseline is slightly better (RMSE {p['rmse']:.3f} vs {s['rmse']:.3f}); "
    "on sparse data an untuned 30-factor SVD does not beat item averages, so its advantage is in ranking, not in rating accuracy. "
    if rmse_better_baseline else
    f"For explicit rating prediction both models are close (RMSE {p['rmse']:.3f} vs {s['rmse']:.3f}), so SVD's advantage is mainly in ranking. ")
closing = ("Because the data is synthetic, absolute numbers should not be compared with published MovieLens results; the relative ordering and trade-offs are the meaningful findings."
           if not real else
           "The gains are meaningful but modest in absolute terms, which is expected on sparse MovieLens data with a single untuned model; "
           "the point of the project is the pipeline, and the comparison shows the evaluation stage can detect the personalisation benefit and its diversity cost.")

unseen = split["test_rows_from_users_unseen_in_train"]
cold_text = (
    f"<b>Cold start:</b> {unseen} test rows come from users with no training history; they only receive popularity recommendations. "
    "A content-based or hybrid model using genre features and demographics would help."
    if unseen else
    "<b>Cold start is not evaluated:</b> the per-user split guarantees every test user has training history, so behaviour for brand-new users "
    "(who would only receive popularity recommendations) is untested. A content-based or hybrid model using genre features and demographics would help.")

story = [
    P("End-to-End Recommendation Data Pipeline - Project Report", h1),
    P("Data Management for Machine Learning - Assignment 2", small),
    P(f"<b>Code repository:</b> {link(REPO_URL)}", small),
    P("1. Objective and data", h2),
    P("The goal was to build a reproducible pipeline that ingests user, item and interaction data, validates and prepares it, "
      "manages features, trains and tracks recommendation models, evaluates them offline and runs as an orchestrated workflow. "
      "The emphasis is on pipeline quality and reproducibility rather than on model accuracy. " + data_text),
    P("2. Architecture", h2),
    Image(str(R / "architecture.png"), width=17 * cm, height=17 * cm * 6.2 / 13),
    P("Seven stages (ingestion, validation, preparation, transformation, feature store, training, evaluation) are separate "
      "modules under <font face='Courier'>src/</font>, each exposing a <font face='Courier'>run(cfg)</font> function. A Prefect flow "
      "(<font face='Courier'>dags/pipeline_flow.py</font>) chains them with retries on ingestion and can be scheduled daily. "
      "All parameters live in <font face='Courier'>configs/config.yaml</font>."),
    P("3. Pipeline stages and design decisions", h2),
    B(f"<b>Validation.</b> Checks cover missing values, dtypes, duplicates, rating range (1-5), timestamp range, ages, and referential "
      f"integrity (every interaction must reference a valid user and item). Of {inter['total_rows']:,} interaction rows, "
      f"{inter['error_rows']} ({100*inter['error_rows']/inter['total_rows']:.2f}%) were rejected and quarantined in "
      f"<font face='Courier'>data/errors/</font> with an <font face='Courier'>error_reason</font> column instead of being dropped silently; "
      f"{val['users']['error_rows']} user and {val['items']['error_rows']} item rows were rejected too. A Pandera schema acts as a final contract check. "
      f"Duplicates are judged only among otherwise valid rows so a corrupted copy never removes the genuine original."),
    B("<b>Preparation.</b> Missing ages are median-imputed (with an indicator flag), gender/occupation standardised, release dates parsed, "
      "timestamps converted, and repeated (user, item) pairs reduced to the latest rating."),
    B(("Each user's most recent 20% of interactions form the test set (per-user temporal split; users with fewer than 5 interactions stay in train): "
       if split.get("strategy") == "per_user_time" else
       f"The latest 20% of interactions form the test set (global cut-off {split['split_cutoff'][:10]}): ")
      .join(["<b>Time-based split.</b> ", ""]) +
      f"{split['train_rows']:,} train / {split['test_rows']:,} test rows. A random split would let the model learn from the future. "
      "A first attempt with a single global cut-off date on MovieLens left most test rows from users with no training history, so personalised models "
      "could not be evaluated meaningfully; the per-user temporal split keeps each user's test items strictly after their training items while ensuring "
      "every evaluated user has history. Every statistic, feature and matrix is computed from the training split only. "
      f"The training matrix is {split['matrix_shape'][0]}x{split['matrix_shape'][1]} with {100*split['matrix_density']:.1f}% density."),
    B("<b>Feature management.</b> A small custom feature store defines three entity levels (user, item, interaction; see "
      "<font face='Courier'>reports/feature_catalog.md</font>) in one catalogue and one code path used by both training and prediction. "
      "Features are persisted as Parquet with a version hash and reference date, and lookups return neutral defaults for cold-start users and items."),
    B("<b>Tracking.</b> Each model has an MLflow run with parameters (seed, K, split cut-off, feature version, SVD size), "
      "the serialised model artifact and, after evaluation, all metrics. A fixed seed makes reruns reproduce identical numbers."),
    P("4. Models and evaluation protocol", h2),
    P("<b>Baseline - popularity:</b> recommends the most-rated unseen items to everyone. <b>Personalised - SVD collaborative filtering:</b> "
      "truncated SVD (30 components) of the user-item rating matrix produces affinity scores for ranking; a second SVD on the user-mean-centred matrix "
      "predicts explicit ratings. Users without training history fall back to popularity. For each user, items already seen in training are excluded; "
      f"test ratings &gt;= 4 are 'relevant'; metrics are computed over the {int(p['n_eval_users'])} test users with at least one relevant item."),
    P("5. Results", h2),
]
rows = [["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "Diversity", "RMSE", "MAE"]]
for _, r in cmp_.iterrows():
    rows.append([r["model"]] + [f"{r[c]:.3f}" for c in [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "rmse", "mae"]])
story += [table(rows, [4.6 * cm] + [1.8 * cm] * 7), Spacer(1, 6),
    P(f"SVD collaborative filtering improves precision@{k} from {p[f'precision@{k}']:.3f} to {s[f'precision@{k}']:.3f} "
      f"(about {s[f'precision@{k}']/p[f'precision@{k}']:.1f}x) and NDCG@{k} from {p[f'ndcg@{k}']:.3f} to {s[f'ndcg@{k}']:.3f}. "
      f"Catalogue coverage rises from {100*p['coverage']:.1f}% to {100*s['coverage']:.1f}% because the baseline shows the same handful of popular items to everyone, "
      f"while the personalised model surfaces a much wider part of the catalogue. The price is lower intra-list diversity ({s['diversity']:.2f} vs {p['diversity']:.2f}): "
      f"SVD concentrates each list on the user's preferred genres. " + rmse_text + closing)]

shots = []
pf, ml = find_image("prefect_run.png"), find_image("mlflow_run.png", "mlflow_runs.png")
if pf:
    shots.append(KeepTogether([
        P("<b>Figure 1.</b> Successful Prefect run of the <i>recsys-pipeline</i> flow: all 7 task runs completed (ingest, validate, prepare, transform, build_features, train, evaluate).", cap),
        picture(pf, 13.5)]))
if ml:
    shots.append(KeepTogether([
        P("<b>Figure 2.</b> MLflow experiment <i>recsys-pipeline</i>: one run per model with logged metrics. The newest pair of runs (top) are the results reported above; "
          "older rows are earlier executions during development, including the initial global-split attempt.", cap),
        picture(ml, 13.5)]))
if shots:
    story += [P("6. Execution evidence", h2)] + shots

story += [
    P("7. Limitations and future work" if shots else "6. Limitations and future work", h2),
    B(cold_text),
    B("<b>Cross-user time overlap:</b> with a per-user split, one user's test interactions can precede another user's training interactions, so the model may use "
      "collaborative signal that is 'from the future' relative to some test events. This is standard leave-last-out practice but weaker than a single global cut-off."),
    B("<b>Offline, single-split evaluation:</b> one temporal split gives no confidence intervals; rolling-window evaluation and online A/B tests are needed. "
      "Relevance is inferred from ratings, so unrated-but-liked items count as misses."),
    B("<b>Scale and serving:</b> everything is batch and in-memory (dense score matrices); production would need ANN retrieval, incremental feature refresh, "
      "a managed feature store (e.g. Feast) and model registry/monitoring for drift."),
    B("<b>Model scope:</b> no hyper-parameter tuning (only 30 SVD components tried) and no implicit-feedback loss such as ALS or BPR."),
    Spacer(1, 4),
    P(f"Source code, configuration, validation report, feature catalog and run instructions: {link(REPO_URL)}", small),
]
SimpleDocTemplate(str(R / "final_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                  topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Recommendation Pipeline Report").build(story)
print("saved", R / "final_report.pdf")
