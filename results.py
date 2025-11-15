import json, os, glob, math
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau

# ---------- تنظیمات ----------
DATA_DIR = "/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/evaluations"  # این مسیر را عوض کن
USE_CALIBRATED = True  # اگر داری، از jury_calibrated_confidence استفاده شود
BIN_COUNT = 10         # برای ECE

def load_records(data_dir):
    rows = []
    for fp in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read {fp}: {e}")
            continue

        jury_verdict = (j.get("jury_verdict") or j.get("jury_decision",{}).get("majority_verdict"))
        conf = j.get("jury_calibrated_confidence") if USE_CALIBRATED else j.get("jury_confidence")
        vd = j.get("jury_decision", {}).get("vote_distribution") or {}
        correct_votes = vd.get("correct", None)
        incorrect_votes = vd.get("incorrect", None)
        uncertain_votes = vd.get("uncertain", None)

        # optional human annotations
        human_verdict = j.get("human_verdict", None)  # "correct"/"incorrect"/"uncertain"
        human_score = j.get("human_score", None)      # e.g., float 0..1 or 1..5

        rows.append({
            "eval_id": j.get("eval_id"),
            "jury_verdict": jury_verdict, 
            "jury_confidence": conf,
            "votes_c": correct_votes, "votes_i": incorrect_votes, "votes_u": uncertain_votes,
            "human_verdict": human_verdict,
            "human_score": human_score
        })
    return pd.DataFrame(rows)

def map_label(x):
    if x is None:
        return None
    x = str(x).lower()
    if x in ["correct", "true", "1", "right"]:
        return 1
    if x in ["incorrect", "false", "0", "wrong"]:
        return 0
    if x in ["uncertain", "unknown", "skip"]:
        return None
    return None

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    return float(np.mean((y_prob - y_true)**2))

def ece_score(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    # clamp
    y_prob = np.clip(y_prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins+1)
    ece = 0.0
    n = len(y_true)
    for b in range(bins):
        lo, hi = edges[b], edges[b+1]
        idx = (y_prob >= lo) & (y_prob < hi) if b < bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if np.any(idx):
            conf = np.mean(y_prob[idx])
            acc  = np.mean(y_true[idx])
            ece += (np.sum(idx)/n) * abs(acc - conf)
    return float(ece)

def fleiss_kappa_from_counts(counts):
    """
    counts: array of shape (N_items, K_categories), entries are integer counts per item per category.
    در اینجا K=3 (C/IC/U) و تعداد رأی‌دهنده‌ها n باید در همه آیتم‌ها یکسان باشد.
    """
    counts = np.asarray(counts, int)
    N, K = counts.shape
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("All items must have the same number of raters for Fleiss' kappa.")
    n = n[0]
    # نسبت هر دسته
    p_j = counts.sum(axis=0) / (N * n)
    # توافق مشاهده‌شده برای هر آیتم
    P_i = ( (counts*(counts-1)).sum(axis=1) ) / (n*(n-1))
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    if (1 - P_e) == 0:
        return np.nan
    return float((P_bar - P_e) / (1 - P_e))

# ---------- اجرا ----------
df = load_records(DATA_DIR)

# نگاشت برچسب‌ها
df["sys_label"]   = df["jury_verdict"].apply(map_label)
df["hum_label"]   = df["human_verdict"].apply(map_label)   # ممکن است None باشد
df["sys_conf"]    = pd.to_numeric(df["jury_confidence"], errors="coerce")

# 1) Accuracy و Cohen’s κ (روی نمونه‌هایی که هر دو برچسب دارند و هیچ‌کدام None نیست)
mask_bin = df["sys_label"].notna() & df["hum_label"].notna()
acc = cohen_kappa = np.nan
if mask_bin.any():
    a = df.loc[mask_bin, "sys_label"].astype(int).to_numpy()
    b = df.loc[mask_bin, "hum_label"].astype(int).to_numpy()
    acc = float(np.mean(a == b))
    cohen_kappa = float(cohen_kappa_score(a, b))

# 2) Spearman/Kendall بین نمره‌های پیوسته (اگر human_score وجود داشته باشد)
spearman_r = spearman_p = kendall_t = kendall_p = np.nan
mask_cont = df["sys_conf"].notna() & df["human_score"].notna()
if mask_cont.any():
    # اگر human_score در بازه‌ی 1..5 باشد، نرمال‌سازی‌اش کنیم به 0..1
    hs = df.loc[mask_cont, "human_score"].astype(float).to_numpy()
    if hs.min() >= 1.0 and hs.max() <= 5.0:
        hs = (hs - 1.0) / 4.0
    sysc = df.loc[mask_cont, "sys_conf"].astype(float).to_numpy()
    spearman_r, spearman_p = spearmanr(sysc, hs)
    kendall_t, kendall_p   = kendalltau(sysc, hs, variant="b")  # مناسب برای ties

# 3) Brier و ECE — نیاز به y_true دوتایی دارد:
brier = ece = np.nan
mask_cal = df["sys_conf"].notna() & df["sys_label"].notna() & df["hum_label"].notna()
if mask_cal.any():
    y_true = (df.loc[mask_cal, "sys_label"] == df.loc[mask_cal, "hum_label"]).astype(int).to_numpy()
    y_prob = df.loc[mask_cal, "sys_conf"].astype(float).to_numpy()
    brier = brier_score(y_true, y_prob)
    ece   = ece_score(y_true, y_prob, bins=BIN_COUNT)

# 4) Fleiss’ κ از رأی jurorها (C/IC/U) — مستقل از برچسب انسانی
fkappa = np.nan
sub = df[["votes_c","votes_i","votes_u"]].dropna()
if not sub.empty:
    counts = sub.astype(int).to_numpy()
    fkappa = fleiss_kappa_from_counts(counts)

# خروجی خلاصه
summary = pd.DataFrame([{
    "N_total": len(df),
    "N_for_accuracy": int(mask_bin.sum()),
    "Accuracy": acc,
    "Cohen_kappa": cohen_kappa,
    "N_for_corr": int(mask_cont.sum()),
    "Spearman_r": spearman_r, "Spearman_p": spearman_p,
    "Kendall_tau_b": kendall_t, "Kendall_p": kendall_p,
    "N_for_calibration": int(mask_cal.sum()),
    "Brier": brier, "ECE@10": ece,
    "Fleiss_kappa_jurors": fkappa
}])

print(summary.to_string(index=False))

# ذخیره CSVها
os.makedirs("/home/zeus/Projects/hb/multi_agent_llm_judge/metrics_out", exist_ok=True)
summary.to_csv("/home/zeus/Projects/hb/multi_agent_llm_judge/metrics_out/summary_metrics.csv", index=False)
df.to_csv("/home/zeus/Projects/hb/multi_agent_llm_judge/metrics_out/per_item_dump.csv", index=False)
print("\nSaved to /home/zeus/Projects/hb/multi_agent_llm_judge/metrics_out/")
