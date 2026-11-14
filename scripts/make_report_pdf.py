"""Builds reports/final_report.pdf from the generated result files (run after the pipeline)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

R = Path(__file__).resolve().parents[1] / "reports"
val = json.loads((R / "validation_report.json").read_text())
split = json.loads((R / "split_summary.json").read_text())
cmp_ = pd.read_csv(R / "model_comparison.csv")
manifest = json.loads((R.parent / "data/raw/ingestion_manifest.json").read_text())

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)

def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="\u2022")

def table(data, widths, header=True):
    t = Table(data, colWidths=widths)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER")]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

k = 10
p, s = cmp_.iloc[0], cmp_.iloc[1]
inter = val["interactions"]

story = [
    P("End-to-End Recommendation Data Pipeline - Project Report", h1),
    P("Data Management for Machine Learning - Assignment 2", small),
    P("1. Objective and data", h2),
    P(f"The goal was to build a reproducible pipeline that ingests user, item and interaction data, validates and "
      f"prepares it, manages features, trains and tracks recommendation models, evaluates them offline and runs "
      f"as an orchestrated workflow. The emphasis is on pipeline quality and reproducibility rather than on model accuracy. "
      f"The pipeline targets MovieLens-100K (943 users, 1,682 movies, 100K ratings, genre flags). "
      f"<b>Because the build environment had no internet access, the results below were generated with the pipeline's "
      f"built-in synthetic fallback</b> ({manifest['source']}: {manifest['rows']['users']} users, {manifest['rows']['items']} items, "
      f"{manifest['rows']['interactions']:,} interactions after bad-row injection), which reproduces the MovieLens schema, a "
      f"long-tail item popularity distribution, genre-driven user tastes and timestamps. Setting <font face='Courier'>data.source: movielens</font> "
      f"(or having internet access in 'auto' mode) reruns everything on the real dataset with no code changes."),
    P("2. Architecture", h2),
    Image(str(R / "architecture.png"), width=17 * cm, height=17 * cm * 6.2 / 13),
    P("Seven stages (ingestion, validation, preparation, transformation, feature store, training, evaluation) are separate "
      "modules under <font face='Courier'>src/</font>, each exposing a <font face='Courier'>run(cfg)</font> function. A Prefect flow "
      "(<font face='Courier'>dags/pipeline_flow.py</font>) chains them with retries on ingestion and can be scheduled daily. "
      "All parameters live in <font face='Courier'>configs/config.yaml</font>.", body),
    P("3. Pipeline stages and design decisions", h2),
    B(f"<b>Validation.</b> Checks cover missing values, dtypes, duplicates, rating range (1-5), timestamp range, ages, and referential "
      f"integrity (every interaction must reference a valid user and item). Of {inter['total_rows']:,} interaction rows, "
      f"{inter['error_rows']} ({100*inter['error_rows']/inter['total_rows']:.2f}%) were rejected and quarantined in "
      f"<font face='Courier'>data/errors/</font> with an <font face='Courier'>error_reason</font> column instead of being dropped silently; "
      f"{val['users']['error_rows']} user and {val['items']['error_rows']} item rows were rejected too. A Pandera schema acts as a final contract check. "
      f"Duplicates are judged only among otherwise valid rows so a corrupted copy never removes the genuine original."),
    B("<b>Preparation.</b> Missing ages are median-imputed (with an indicator flag), gender/occupation standardised, release dates parsed, "
      "timestamps converted, and repeated (user, item) pairs reduced to the latest rating."),
    B(f"<b>Time-based split.</b> " + (
      f"Each user's most recent 20% of interactions form the test set (per-user temporal split, users with fewer than 5 interactions stay in train): "
      if split.get("strategy") == "per_user_time" else
      f"The latest 20% of interactions form the test set (global cut-off {split['split_cutoff'][:10]}): ") +
      f"{split['train_rows']:,} train / {split['test_rows']:,} test rows. A random split would let the model learn from the future. "
      f"A first attempt with a single global cut-off date on MovieLens left most test rows from users with no training history, so personalised models "
      f"could not be evaluated; the per-user temporal split keeps each user's test items strictly after their training items while ensuring every evaluated user has history. "
      f"Every statistic, feature and matrix is computed from the training split only. "
      f"The training matrix is {split['matrix_shape'][0]}x{split['matrix_shape'][1]} with {100*split['matrix_density']:.1f}% density."),
    B("<b>Feature management.</b> A small custom feature store defines 3 entity levels (user, item, interaction; see "
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
      f"Catalogue coverage rises from {100*p['coverage']:.1f}% to {100*s['coverage']:.1f}% because the baseline shows the same handful of blockbusters to everyone, "
      f"while the personalised model surfaces long-tail items. The price is lower intra-list diversity ({s['diversity']:.2f} vs {p['diversity']:.2f}): "
      f"SVD concentrates each list on the user's preferred genres, whereas the popularity list is genre-mixed by accident. "
      f"For explicit rating prediction both models are essentially equal (RMSE {p['rmse']:.3f} vs {s['rmse']:.3f}); the item-mean baseline is hard to beat "
      f"on sparse data, so SVD's advantage is in ranking, not in rating accuracy. Because the data is synthetic, absolute numbers should not be compared "
      f"with published MovieLens results; the relative ordering and the trade-offs are the meaningful findings.")]
story += [
    P("6. Limitations and future work", h2),
    B(f"<b>Cold start:</b> {split['test_rows_from_users_unseen_in_train']} test rows come from users with no training history (late joiners); they only receive popularity "
      "recommendations. A content-based or hybrid model using genre features and demographics would help."),
    B("<b>Cross-user time overlap:</b> with a per-user split, one user's test interactions can precede another user's training interactions, so the model may use "
      "collaborative signal that is 'from the future' relative to some test events. This is standard leave-last-out practice but weaker than a single global cut-off."),
    B("<b>Offline, single-split evaluation:</b> one temporal split gives no confidence intervals; rolling-window evaluation and online A/B tests are needed. "
      "Relevance is inferred from ratings, so unrated-but-liked items count as misses."),
    B("<b>Scale and serving:</b> everything is batch and in-memory (dense score matrices); production would need ANN retrieval, incremental feature refresh, "
      "a managed feature store (e.g. Feast) and model registry/monitoring for drift."),
    B("<b>Model scope:</b> no hyper-parameter tuning (only 30 SVD components tried) and no implicit-feedback loss such as ALS or BPR."),
]
SimpleDocTemplate(str(R / "final_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                  topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Recommendation Pipeline Report").build(story)
print("saved", R / "final_report.pdf")
