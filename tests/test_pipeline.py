import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation.metrics import catalog_coverage, intra_list_diversity, precision_recall_ndcg, rmse
from src.training.models import PopularityRecommender, SVDRecommender
from src.transformation.transform import per_user_time_split, time_split
from src.validation.validate import validate_interactions


def test_precision_recall_ndcg_perfect_and_miss():
    recs = np.array([[0, 1, 2], [3, 4, 5]])
    p, r, n = precision_recall_ndcg(recs, [{0, 1, 2}, {9}], k=3)
    assert p == pytest.approx((1.0 + 0.0) / 2)
    assert r == pytest.approx((1.0 + 0.0) / 2)
    assert n == pytest.approx((1.0 + 0.0) / 2)


def test_coverage_and_diversity():
    recs = np.array([[0, 1], [1, 2]])
    assert catalog_coverage(recs, 6) == pytest.approx(0.5)
    genres = np.array([[1, 0], [1, 0], [0, 1]])
    assert intra_list_diversity(np.array([[0, 1]]), genres) == pytest.approx(0.0)   # identical genres
    assert intra_list_diversity(np.array([[0, 2]]), genres) == pytest.approx(1.0)   # disjoint genres


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_time_split_has_no_leakage():
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=100, freq="D")})
    train, test, _ = time_split(df, 0.2)
    assert train["datetime"].max() < test["datetime"].min()
    assert len(test) == pytest.approx(20, abs=1)


def test_validation_flags_bad_rows():
    users = pd.DataFrame({"user_id": [1]})
    items = pd.DataFrame({"item_id": [1]})
    inter = pd.DataFrame({"user_id": ["1", "1", "1", "2"], "item_id": ["1", "1", "1", "1"],
                          "rating": ["5", "9", "x", "4"], "timestamp": ["1700000000"] * 4})
    v = {"rating_min": 1, "rating_max": 5, "min_timestamp": 631152000}
    valid, errors, counts = validate_interactions(inter, users, items, v)
    assert len(valid) == 1
    assert counts["invalid_rating"] == 1 and counts["invalid_dtype_rating"] == 1
    assert counts["orphan_user_id"] == 1


def test_recommenders_exclude_seen_items_and_handle_cold_users():
    R = sparse.csr_matrix(np.array([[5, 0, 0, 3], [0, 4, 0, 0], [0, 0, 0, 0]], float))
    itf = pd.DataFrame({"item_popularity": [1, 1, 0, 1], "item_avg_rating": [5, 4, 3, 3]})
    pop = PopularityRecommender().fit(itf, 3.5)
    svd = SVDRecommender(2, 0).fit(R, np.array([4.0, 4.0, 3.5]), 3.5, pop.scores)
    for model in (pop, svd):
        recs = model.recommend(np.array([0, 1, 2]), R, 2)
        assert 0 not in recs[0] and 3 not in recs[0]       # user 0 already saw items 0 and 3
        assert 1 not in recs[1]
        assert recs.shape == (3, 2)


def test_per_user_split_keeps_every_user_in_train_and_is_ordered():
    rows = []
    for u in (1, 2, 3):
        for d in range(10):
            rows.append({"user_id": u, "item_id": d, "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)})
    rows.append({"user_id": 4, "item_id": 0, "datetime": pd.Timestamp("2024-01-01")})   # too few -> train only
    df = pd.DataFrame(rows)
    train, test, _ = per_user_time_split(df, 0.2, min_interactions=5)
    assert set(test["user_id"]) == {1, 2, 3}
    assert set(train["user_id"]) == {1, 2, 3, 4}
    for u in (1, 2, 3):
        assert train[train.user_id == u]["datetime"].max() < test[test.user_id == u]["datetime"].min()
        assert len(test[test.user_id == u]) == 2
