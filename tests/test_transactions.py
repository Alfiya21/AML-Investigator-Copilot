from aml_copilot.transactions import db, explain, scorer
from aml_copilot.transactions.features import FEATURES, build_features


def test_features_have_expected_columns(trained):
    feats = build_features(db.all_transactions(db.sample_account_ids(1)[0]))
    assert list(feats.columns) == FEATURES
    assert len(feats) == 1


def test_model_separates_suspicious_accounts(trained):
    assert trained["roc_auc"] > 0.9


def test_scores_are_higher_for_suspicious_accounts(trained):
    bad = [scorer.score_account(a)["risk_score"] for a in db.sample_account_ids(10, suspicious_only=True)]
    ok = [scorer.score_account(a)["risk_score"] for a in db.sample_account_ids(30)]
    assert sum(bad) / len(bad) > sum(ok) / len(ok)


def test_explanations_name_real_features(trained):
    account = db.sample_account_ids(1, suspicious_only=True)[0]
    reasons = explain.top_reasons(account, top_k=3)
    assert len(reasons) == 3
    assert all(r["feature"] in FEATURES for r in reasons)


def test_unknown_account_returns_nothing(trained):
    assert scorer.score_account("A99999") is None
    assert db.get_profile("A99999") is None


def test_profile_does_not_leak_ground_truth(trained):
    profile = db.get_profile(db.sample_account_ids(1)[0])
    assert set(profile) == {"account_id", "segment", "region"}
