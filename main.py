# # -*- coding: utf-8 -*-
# """
# Evaluate multi-agent judge on chattify/hotpot_qa-answers (config: gpt_4o_mini)

# - question: HF field "question"
# - model answer to be judged: HF field "judge_output"  (goes to EvaluationRequest.answer)
# - gold/human answer: HF field "answer"               (for metrics; NOT shown to judge)

# Outputs:
#   calibration_data/
#     ├─ evaluations/*.json  (full per-item logs + gold metrics)
#     ├─ per_item_dump.csv
#     └─ summary_metrics.csv
# """

# import os, sys, json, math
# from pathlib import Path
# from typing import Dict, Any, List, Tuple
# import numpy as np
# import pandas as pd

# from datetime import datetime
# from dotenv import load_dotenv
# from loguru import logger

# # --- project imports (adjusted to your repo layout)
# sys.path.insert(0, str(Path(__file__).parent.parent))
# from multi_agent_llm_judge.core.data_models import EvaluationRequest, Verdict
# from multi_agent_llm_judge.core.round_manager import RoundManager
# from multi_agent_llm_judge.core.model_manager import ModelManager
# from multi_agent_llm_judge.core.cache_manager import CacheManager, MemoryCache, DiskCache
# from multi_agent_llm_judge.providers.openrouter import OpenRouterClient
# from multi_agent_llm_judge.utils.config_loader import load_config
# from multi_agent_llm_judge.utils.logging_config import configure_logging

# # ======================== CONFIG ===========================
# # Source can be: "hf" | "csv" | "jsonl"
# DATA_SOURCE = "hf"
# DATA_PATH_OR_HF = {
#     "hf_dataset": "chattify/hotpot_qa-answers",
#     "hf_config":  "gpt-4o-mini",   # ← اسم درست (با hyphen)
#     "hf_split":   "validation"          # اگر اسپیلیت دیگری داری عوضش کن
# }


# OUT_ROOT = Path(__file__).parent / "calibration_data"
# USE_CALIBRATED_CONF = True
# ECE_BINS = 10
# # ===========================================================


# # ----------------- Normalizers for EM/F1 -------------------
# import string
# ARTICLES = {"a", "an", "the"}
# PUNCT = set(string.punctuation)

# def _normalize_text(s: str) -> str:
#     if s is None: return ""
#     s = s.lower().strip()
#     s = "".join(ch for ch in s if ch not in PUNCT)
#     s = " ".join(w for w in s.split() if w not in ARTICLES)
#     return s

# def exact_match(pred: str, gold: str) -> int:
#     return int(_normalize_text(pred) == _normalize_text(gold))

# def _tokenize(s: str) -> List[str]:
#     return _normalize_text(s).split()

# def f1_score_str(pred: str, gold: str) -> float:
#     p, g = _tokenize(pred), _tokenize(gold)
#     if len(p) == 0 and len(g) == 0: return 1.0
#     if len(p) == 0 or len(g) == 0:  return 0.0
#     common = {}
#     for w in g: common[w] = common.get(w, 0) + 1
#     num_same = 0
#     for w in p:
#         if common.get(w, 0) > 0:
#             num_same += 1
#             common[w] -= 1
#     if num_same == 0: return 0.0
#     precision = num_same / len(p)
#     recall    = num_same / len(g)
#     return 2 * precision * recall / (precision + recall)


# # ---------------- Calibration / Agreement ------------------
# def brier_score(y_true, y_prob):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.asarray(y_prob, float)
#     return float(np.mean((y_prob - y_true)**2))

# def ece_score(y_true, y_prob, bins=10):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.clip(np.asarray(y_prob, float), 0.0, 1.0)
#     edges = np.linspace(0.0, 1.0, bins+1)
#     ece = 0.0
#     n = len(y_true)
#     for b in range(bins):
#         lo, hi = edges[b], edges[b+1]
#         idx = (y_prob >= lo) & (y_prob < hi) if b < bins-1 else (y_prob >= lo) & (y_prob <= hi)
#         if np.any(idx):
#             conf = float(np.mean(y_prob[idx]))
#             acc  = float(np.mean(y_true[idx]))
#             ece += (np.sum(idx)/n) * abs(acc - conf)
#     return float(ece)

# def cohen_kappa(y_sys, y_hum):
#     a = np.asarray(y_sys, int)
#     b = np.asarray(y_hum, int)
#     Po = (a == b).mean()
#     p1 = a.mean(); q1 = 1 - p1
#     p2 = b.mean(); q2 = 1 - p2
#     Pe = p1*p2 + q1*q2
#     return (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan

# def fleiss_kappa_from_counts(counts):
#     counts = np.asarray(counts, int)
#     if counts.size == 0: return np.nan
#     N, K = counts.shape
#     n = counts.sum(axis=1)
#     if not np.all(n == n[0]):
#         raise ValueError("All items must have the same number of raters for Fleiss' kappa.")
#     n = n[0]
#     p_j = counts.sum(axis=0) / (N * n)
#     P_i = ((counts*(counts-1)).sum(axis=1)) / (n*(n-1))
#     P_bar = P_i.mean()
#     P_e = (p_j**2).sum()
#     return (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

# def kripp_alpha_nominal_from_counts(counts):
#     C = np.asarray(counts, float)
#     if C.size == 0: return np.nan
#     n_i = C.sum(axis=1)
#     valid = n_i > 1
#     C = C[valid]; n_i = n_i[valid]
#     if len(n_i) == 0: return np.nan
#     coincidence = (C * (C - 1)).sum(axis=0)
#     Do = 0.0; denom = 0.0
#     for row, n in zip(C, n_i):
#         Do += (n*(n-1) - (row*(row-1)).sum())
#         denom += n*(n-1)
#     Do = Do/denom if denom != 0 else np.nan
#     p = coincidence / coincidence.sum() if coincidence.sum() != 0 else np.zeros_like(coincidence)
#     De = 1.0 - (p**2).sum()
#     if De == 0 or np.isnan(Do): return np.nan
#     return 1.0 - (Do / De)

# def _normalize_cfg(s: str) -> str:
#     return (s or "").strip().lower().replace("_", "-")

# def resolve_hf_config_name(dataset_name: str, requested: str) -> str:
#     """
#     برمی‌گرداند اسم کانفیگ معتبر در HF را؛ اگر 'requested' نبود:
#     - underscore→hyphen
#     - مقایسهٔ case-insensitive
#     - در نهایت فازی (difflib)
#     """
#     from datasets import get_dataset_config_names
#     avail = get_dataset_config_names(dataset_name)
#     if not avail:
#         raise ValueError(f"No configs found for dataset {dataset_name}")

#     req_norm = _normalize_cfg(requested)
#     # 1) تطبیق مستقیم با نرمال‌سازی
#     for c in avail:
#         if _normalize_cfg(c) == req_norm:
#             return c

#     # 2) تطبیق ساده: اگر آندرلاین بود، با هایفن امتحان کن
#     req_dash = requested.replace("_", "-")
#     for c in avail:
#         if c == req_dash:
#             return c

#     # 3) فازی
#     import difflib
#     best = difflib.get_close_matches(requested, avail, n=1, cutoff=0.0)
#     if best:
#         print(f"[WARN] HF config '{requested}' not found. Using closest match '{best[0]}'. "
#               f"Available: {avail}")
#         return best[0]

#     raise ValueError(f"BuilderConfig '{requested}' not found. Available: {avail}")

# # ---------------------- IO: load dataset -------------------
# def load_chattify_hotpot(source: str, cfg: Dict[str, Any]) -> pd.DataFrame:
#     if source == "hf":
#         try:
#             from datasets import load_dataset
#         except Exception as e:
#             raise RuntimeError("Install `datasets` (pip install datasets) or switch DATA_SOURCE to csv/jsonl") from e

#         ds_name = cfg["hf_dataset"]
#         raw_cfg = cfg["hf_config"]
#         cfg_name = resolve_hf_config_name(ds_name, raw_cfg)  # ← مهم
#         split = cfg.get("hf_split", "train")

#         print(f"[INFO] Loading HF dataset={ds_name}, config={cfg_name}, split={split}")
#         ds = load_dataset(ds_name, cfg_name)
#         tab = ds[split]
#         df = tab.to_pandas()
#     elif source == "csv":
#         df = pd.read_csv(cfg["file"])
#     elif source == "jsonl":
#         rows = []
#         with open(cfg["file"], "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line: continue
#                 rows.append(json.loads(line))
#         df = pd.DataFrame(rows)
#     else:
#         raise ValueError("DATA_SOURCE must be one of {hf,csv,jsonl}")

#     # ستون‌های ضروری
#     for col in ["question", "judge_output", "answer"]:
#         if col not in df.columns:
#             raise ValueError(f"Missing required column: {col}")

#     # id را تنظیم کن
#     if "id" not in df.columns:
#         # تلاش برای یافتن فیلد شناسه
#         for c in ["_id", "example_id", "uid", "sample_id"]:
#             if c in df.columns:
#                 df = df.rename(columns={c: "id"})
#                 break
#         if "id" not in df.columns:
#             # اگر هیچ‌کدام نبود، خودمان بسازیم
#             df["id"] = [str(i) for i in range(len(df))]

#     # تایپ‌ها
#     df["id"] = df["id"].astype(str)
#     df["question"] = df["question"].astype(str)
#     df["judge_output"] = df["judge_output"].astype(str)
#     df["answer"] = df["answer"].astype(str)

#     # فیلدهای اختیاری
#     for c in ["supporting_facts", "context"]:
#         if c not in df.columns:
#             df[c] = [{}] * len(df)

#     return df


# # ---------------------- Context builder --------------------
# def build_context_blob(context: Dict[str, Any]) -> str:
#     """
#     If the dataset has wiki-style context {title[], sentences[[]]}, pretty-print it.
#     If not available, return empty string.
#     """
#     if not isinstance(context, dict) or "title" not in context or "sentences" not in context:
#         return ""
#     titles = context["title"]; sents = context["sentences"]
#     parts = []
#     for t, lst in zip(titles, sents):
#         joined = " ".join(str(x).strip() for x in (lst or []))
#         parts.append(f"[{t}] {joined}")
#     return "\n".join(parts)


# # ---------------------- Main pipeline ----------------------
# async def main():
#     # --- infra
#     env_path = Path(__file__).parent / ".env"
#     load_dotenv(env_path)
#     configure_logging()

#     try:
#         config = load_config()
#         logger.info("Configuration loaded.")
#     except Exception as e:
#         logger.error(f"Failed to load config: {e}")
#         return

#     # cache
#     if config.cache.backend == "memory":
#         cache_backend = MemoryCache(max_size=config.cache.max_size)
#     elif config.cache.backend == "disk":
#         cache_backend = DiskCache(cache_dir=config.cache.cache_dir)
#     else:
#         cache_backend = MemoryCache(max_size=10000)

#     cache_manager = CacheManager(primary_backend=cache_backend, namespace="judge", stats_enabled=True)

#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         logger.error("OPENROUTER_API_KEY is missing.")
#         return

#     provider = OpenRouterClient(api_key=api_key)
#     model_manager = ModelManager(provider=provider)
#     await model_manager.initialize()
#     round_manager = RoundManager(config=config, model_manager=model_manager, cache_manager=cache_manager)

#     # --- IO
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     eval_dir = OUT_ROOT / "evaluations"
#     eval_dir.mkdir(exist_ok=True)

#     df = load_chattify_hotpot(DATA_SOURCE, DATA_PATH_OR_HF)
#     logger.info(f"Loaded {len(df)} rows from {DATA_SOURCE}.")

#     per_item = []

#     for idx, row in df.iterrows():
#         q = row["question"]
#         pred = row["judge_output"]   # ← به داور می‌دهیم
#         gold = row["answer"]         # ← فقط برای متریک

#         req = EvaluationRequest(
#             question=q,
#             answer=pred,
#             context=None,             # اگر context dict داری، build_context_blob(row["context"])
#             metadata={"hotpot_id": row["id"], "source": "chattify/gpt_4o_mini"}
#         )

#         try:
#             result = await round_manager.evaluate(req)
#         except Exception as e:
#             logger.error(f"Judge failed on id={row['id']}: {e}")
#             continue

#         if not (result and result.success and result.judgment):
#             logger.error(f"Empty judgment on id={row['id']}")
#             continue

#         jd = result.judgment
#         eval_id = f"eval_{jd.timestamp.strftime('%Y%m%d_%H%M%S')}_{jd.id[:8]}"

#         # Hotpot answer metrics
#         ans_em = exact_match(pred, gold)
#         ans_f1 = f1_score_str(pred, gold)

#         # human verdict from EM
#         human_verdict = "correct" if ans_em == 1 else "incorrect"
#         human_score   = float(ans_f1)

#         # system verdict
#         sys_verdict = jd.is_correct.value.lower()  # "correct"/"incorrect"
#         raw_conf    = float(jd.raw_confidence or 0.0)
#         calib_conf  = float(jd.calibrated_confidence or raw_conf)
#         sys_conf    = calib_conf if USE_CALIBRATED_CONF else raw_conf

#         vd = jd.jury_decision.vote_distribution or {}
#         votes_c = int(vd.get("correct", 0))
#         votes_i = int(vd.get("incorrect", 0))
#         votes_u = int(vd.get("uncertain", 0))

#         # Save rich JSON
#         item = {
#             "eval_id": eval_id,
#             "timestamp": jd.timestamp.isoformat(),
#             "id": row["id"],
#             "question": q,
#             "answer": pred,                 # model answer (judged)
#             "gold_answer": gold,            # human gold
#             "jury_verdict": sys_verdict,
#             "jury_says_correct": (sys_verdict == "correct"),
#             "jury_confidence": raw_conf,
#             "jury_calibrated_confidence": sys_conf,
#             "human_verdict": human_verdict,
#             "human_score": human_score,
#             "ground_truth": { "Answer_EM": ans_em, "Answer_F1": ans_f1 },
#             "jury_decision": {
#                 "majority_verdict": jd.jury_decision.majority_verdict.value,
#                 "consensus_level": jd.jury_decision.consensus_level,
#                 "vote_distribution": jd.jury_decision.vote_distribution,
#                 "weighted_confidence": jd.jury_decision.weighted_confidence,
#                 "key_agreements": jd.jury_decision.key_agreements,
#                 "key_disagreements": jd.jury_decision.key_disagreements
#             },
#             "metrics": {
#                 "total_cost": jd.total_cost,
#                 "total_tokens": jd.total_tokens,
#                 "processing_time_ms": jd.processing_time_ms,
#                 "models_used": jd.models_used,
#                 "num_rounds": len(jd.round_summaries),
#                 "num_agents": sum(len(rs.agent_analyses) for rs in jd.round_summaries)
#             },
#             "rounds": [
#                 {
#                     "round_number": rs.round_number,
#                     "average_confidence": rs.average_confidence,
#                     "confidence_variance": rs.confidence_variance,
#                     "consensus_points": rs.consensus_points,
#                     "disagreement_points": rs.disagreement_points,
#                     "agents": [
#                         {
#                             "name": a.agent_name,
#                             "type": a.agent_type.value,
#                             "model": a.model_used,
#                             "verdict": a.verdict.value if a.verdict else None,
#                             "confidence": a.confidence,
#                             "reasoning_preview": a.analysis[:200] + '...' if len(a.analysis) > 200 else a.analysis
#                         } for a in rs.agent_analyses
#                     ]
#                 } for rs in jd.round_summaries
#             ],
#             "executive_summary": jd.executive_summary,
#             "detailed_rationale_preview": jd.detailed_rationale[:500] + '...' if len(jd.detailed_rationale) > 500 else jd.detailed_rationale
#         }

#         with open(eval_dir / f"{eval_id}.json", "w", encoding="utf-8") as f:
#             json.dump(item, f, indent=2, ensure_ascii=False)

#         per_item.append({
#             "id": row["id"],
#             "eval_id": eval_id,
#             "sys_verdict": sys_verdict,
#             "sys_conf": sys_conf,
#             "votes_c": votes_c, "votes_i": votes_i, "votes_u": votes_u,
#             "Answer_EM": ans_em, "Answer_F1": ans_f1,
#             "human_verdict": human_verdict,
#             "human_score": human_score
#         })

#     # ================= Aggregation =================
#     per = pd.DataFrame(per_item)
#     if not per.empty:
#         per.to_csv(OUT_ROOT / "per_item_dump.csv", index=False)

#         # binary labels
#         per["hum_label"] = (per["human_verdict"].str.lower() == "correct").astype(int)
#         per["sys_label"] = (per["sys_verdict"].str.lower() == "correct").astype(int)

#         # Accuracy / Cohen κ
#         mask_bin = per["hum_label"].notna() & per["sys_label"].notna()
#         acc = float((per.loc[mask_bin,"hum_label"] == per.loc[mask_bin,"sys_label"]).mean()) if mask_bin.any() else np.nan
#         kappa = cohen_kappa(per.loc[mask_bin,"sys_label"], per.loc[mask_bin,"hum_label"]) if mask_bin.any() else np.nan

#         # Spearman/Kendall: sys_conf vs human_score(=Answer_F1)
#         from scipy.stats import spearmanr, kendalltau
#         mask_cont = per["sys_conf"].notna() & per["human_score"].notna()
#         spearman_r = spearman_p = kendall_t = kendall_p = np.nan
#         if mask_cont.any():
#             sr = spearmanr(per.loc[mask_cont,"sys_conf"], per.loc[mask_cont,"human_score"])
#             spearman_r, spearman_p = float(sr.correlation), float(sr.pvalue)
#             kt = kendalltau(per.loc[mask_cont,"sys_conf"], per.loc[mask_cont,"human_score"], variant="b")
#             kendall_t, kendall_p = float(kt.correlation), float(kt.pvalue)

#         # Brier / ECE
#         brier = ece = np.nan
#         if mask_bin.any():
#             y_true = per.loc[mask_bin,"hum_label"].astype(int).to_numpy()
#             y_prob = per.loc[mask_bin,"sys_conf"].astype(float).to_numpy()
#             brier = brier_score(y_true, y_prob)
#             ece   = ece_score(y_true, y_prob, bins=ECE_BINS)

#         # Fleiss κ (needs fixed #jurors) + Kripp α (nominal)
#         fkappa = np.nan; alpha_nominal = np.nan
#         if {"votes_c","votes_i","votes_u"}.issubset(per.columns):
#             tmp = per[["votes_c","votes_i","votes_u"]].fillna(0).astype(int)
#             # Fleiss: فقط آیتم‌هایی با تعداد رأی یکسان
#             n_votes = tmp.sum(axis=1)
#             if not tmp.empty:
#                 try:
#                     n_mode = int(n_votes.mode(dropna=True)[0])
#                     sub = tmp[n_votes == n_mode]
#                     if not sub.empty:
#                         fkappa = fleiss_kappa_from_counts(sub.to_numpy())
#                 except Exception as e:
#                     logger.warning(f"Fleiss κ skipped: {e}")
#             alpha_nominal = kripp_alpha_nominal_from_counts(tmp.to_numpy())

#         summary = pd.DataFrame([{
#             "N_total": len(per),
#             "Answer_EM": float(per["Answer_EM"].mean()) if len(per) else np.nan,
#             "Answer_F1": float(per["Answer_F1"].mean()) if len(per) else np.nan,
#             "Verdict_Accuracy": acc,
#             "Cohen_kappa": kappa,
#             "Spearman_r": spearman_r, "Spearman_p": spearman_p,
#             "Kendall_tau_b": kendall_t, "Kendall_p": kendall_p,
#             "Brier": brier, "ECE@10": ece,
#             "Fleiss_kappa_jurors": fkappa,
#             "Krippendorff_alpha_nominal": alpha_nominal
#         }])
#         summary.to_csv(OUT_ROOT / "summary_metrics.csv", index=False)
#         print(summary.to_string(index=False))
#     else:
#         logger.warning("No rows evaluated; per_item is empty.")

#     # cleanup
#     await round_manager.agent_manager.cleanup()
#     await provider.close()


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())




###################################################################################################################################################################

# import os
# import asyncio
# import sys
# import json
# import pickle
# from pathlib import Path
# from typing import Optional, List, Tuple, Dict
# from datetime import datetime
# from dotenv import load_dotenv
# from loguru import logger
# from datasets import load_dataset
# import random

# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from multi_agent_llm_judge.core.agent_manager import AgentManager
# from multi_agent_llm_judge.core.round_manager import RoundManager
# from multi_agent_llm_judge.core.model_manager import ModelManager
# from multi_agent_llm_judge.core.cache_manager import CacheManager, MemoryCache, DiskCache
# from multi_agent_llm_judge.core.data_models import EvaluationRequest, EvaluationResult, FinalJudgment, Verdict
# from multi_agent_llm_judge.providers.openrouter import OpenRouterClient
# from multi_agent_llm_judge.utils.config_loader import load_config
# from multi_agent_llm_judge.utils.logging_config import configure_logging

# class CalibrationDataCollector:
#     """Collects evaluation data for future calibration training."""
    
#     def __init__(self, data_dir: Path):
#         self.data_dir = data_dir
#         self.data_dir.mkdir(parents=True, exist_ok=True)
        
#         # Different directories for data types
#         self.raw_dir = self.data_dir / "raw"
#         self.evaluations_dir = self.data_dir / "evaluations"
#         self.unlabeled_dir = self.data_dir / "unlabeled"
        
#         for dir in [self.raw_dir, self.evaluations_dir, self.unlabeled_dir]:
#             dir.mkdir(exist_ok=True)
    
#     def save_unlabeled_evaluation(self, result: EvaluationResult) -> str:
#         """
#         Save evaluation result without ground truth label for future labeling.
        
#         Args:
#             result: Evaluation result from jury system
            
#         Returns:
#             Unique ID of saved evaluation
#         """
#         if not result.success or not result.judgment:
#             logger.error("Cannot save failed evaluation")
#             return None
            
#         # Generate unique ID for this evaluation
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         eval_id = f"eval_{timestamp}_{result.judgment.id[:8]}"
        
#         # 1. Save raw result (pickle for complete object preservation)
#         raw_path = self.raw_dir / f"{eval_id}.pkl"
#         with open(raw_path, 'wb') as f:
#             pickle.dump({
#                 'eval_id': eval_id,
#                 'result': result,
#                 'timestamp': datetime.now(),
#                 'labeled': False,
#                 'ground_truth': None  # To be filled later
#             }, f)
#         logger.info(f"Saved raw evaluation to {raw_path}")
        
#         # 2. Save evaluation details (JSON for readability and manual labeling)
#         eval_data = self._extract_evaluation_data(result, eval_id)
#         eval_path = self.evaluations_dir / f"{eval_id}.json"
#         with open(eval_path, 'w', encoding='utf-8') as f:
#             json.dump(eval_data, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved evaluation details to {eval_path}")
        
#         # 3. Save unlabeled data summary for easy review
#         unlabeled_data = self._create_unlabeled_summary(result, eval_id)
#         unlabeled_path = self.unlabeled_dir / f"{eval_id}_unlabeled.json"
#         with open(unlabeled_path, 'w', encoding='utf-8') as f:
#             json.dump(unlabeled_data, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved unlabeled summary to {unlabeled_path}")
        
#         return eval_id
    
#     def _extract_evaluation_data(self, result: EvaluationResult, eval_id: str) -> dict:
#         """Extract comprehensive evaluation data for analysis and labeling."""
#         judgment = result.judgment
        
#         # Convert Verdict to boolean for jury's decision
#         jury_says_correct = judgment.is_correct == Verdict.CORRECT
        
#         return {
#             'eval_id': eval_id,
#             'timestamp': judgment.timestamp.isoformat(),
            
#             # Question and answer
#             'question': judgment.question,
#             'answer': judgment.answer,
#             'context': judgment.context,
            
#             # Jury results
#             'jury_verdict': judgment.is_correct.value,
#             'jury_says_correct': jury_says_correct,
#             'jury_confidence': judgment.raw_confidence,
#             'jury_calibrated_confidence': judgment.calibrated_confidence,
            
#             # Ground truth placeholder
#             'ground_truth': None,  # To be filled manually later
#             'labeling_notes': "",  # For annotator notes
            
#             # Jury decision details
#             'jury_decision': {
#                 'majority_verdict': judgment.jury_decision.majority_verdict.value,
#                 'consensus_level': judgment.jury_decision.consensus_level,
#                 'vote_distribution': judgment.jury_decision.vote_distribution,
#                 'weighted_confidence': judgment.jury_decision.weighted_confidence,
#                 'key_agreements': judgment.jury_decision.key_agreements,
#                 'key_disagreements': judgment.jury_decision.key_disagreements
#             },
            
#             # Performance metrics
#             'metrics': {
#                 'total_cost': judgment.total_cost,
#                 'total_tokens': judgment.total_tokens,
#                 'processing_time_ms': judgment.processing_time_ms,
#                 'models_used': judgment.models_used,
#                 'num_rounds': len(judgment.round_summaries),
#                 'num_agents': sum(len(rs.agent_analyses) for rs in judgment.round_summaries)
#             },
            
#             # Calibration features (for future training)
#             'calibration_features': {
#                 'base_confidence': judgment.calibration_features.base_confidence,
#                 'agent_agreement': judgment.calibration_features.agent_agreement,
#                 'evidence_strength': judgment.calibration_features.evidence_strength,
#                 'reasoning_coherence': judgment.calibration_features.reasoning_coherence,
#                 'model_diversity': judgment.calibration_features.model_diversity,
#                 'consensus_strength': judgment.calibration_features.consensus_strength,
#                 'token_efficiency': judgment.calibration_features.token_efficiency,
#                 'round_consistency': judgment.calibration_features.round_consistency
#             },
            
#             # Detailed round information
#             'rounds': [
#                 {
#                     'round_number': rs.round_number,
#                     'average_confidence': rs.average_confidence,
#                     'confidence_variance': rs.confidence_variance,
#                     'consensus_points': rs.consensus_points,
#                     'disagreement_points': rs.disagreement_points,
#                     'agents': [
#                         {
#                             'name': a.agent_name,
#                             'type': a.agent_type.value,
#                             'model': a.model_used,
#                             'verdict': a.verdict.value if a.verdict else None,
#                             'confidence': a.confidence,
#                             'reasoning_preview': a.analysis[:200] + '...' if len(a.analysis) > 200 else a.analysis
#                         }
#                         for a in rs.agent_analyses
#                     ]
#                 }
#                 for rs in judgment.round_summaries
#             ],
            
#             # Executive summary
#             'executive_summary': judgment.executive_summary,
#             'detailed_rationale_preview': judgment.detailed_rationale[:500] + '...' if len(judgment.detailed_rationale) > 500 else judgment.detailed_rationale
#         }
    
#     def _create_unlabeled_summary(self, result: EvaluationResult, eval_id: str) -> dict:
#         """Create a simplified summary for easy manual review and labeling."""
#         judgment = result.judgment
#         jury_says_correct = judgment.is_correct == Verdict.CORRECT
        
#         return {
#             'eval_id': eval_id,
#             'timestamp': datetime.now().isoformat(),
            
#             # Core information for labeling
#             'question': judgment.question,
#             'answer': judgment.answer,
#             'jury_verdict': 'CORRECT' if jury_says_correct else 'INCORRECT',
#             'jury_confidence': f"{judgment.raw_confidence:.1%}",
#             'consensus': f"{judgment.jury_decision.consensus_level:.1%}",
            
#             # Placeholder for manual labeling
#             'ground_truth': None,  # Should be filled as True/False
#             'labeling_confidence': None,  # How confident is the labeler? (0-1)
#             'labeling_notes': "",
#             'labeled_by': "",
#             'labeled_at': None,
            
#             # Quick reference
#             'executive_summary': judgment.executive_summary,
#             'file_path': f"evaluations/{eval_id}.json"
#         }
    
#     def create_labeling_batch(self, output_file: str = "labeling_batch.json") -> int:
#         """
#         Create a batch file for manual labeling of all unlabeled evaluations.
        
#         Returns:
#             Number of evaluations in the batch
#         """
#         unlabeled_files = list(self.unlabeled_dir.glob("*_unlabeled.json"))
        
#         batch_data = {
#             'created_at': datetime.now().isoformat(),
#             'total_evaluations': len(unlabeled_files),
#             'evaluations': []
#         }
        
#         for file_path in sorted(unlabeled_files):
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 eval_data = json.load(f)
            
#             # Only include unlabeled evaluations
#             if eval_data.get('ground_truth') is None:
#                 batch_data['evaluations'].append(eval_data)
        
#         # Save batch file
#         batch_path = self.data_dir / output_file
#         with open(batch_path, 'w', encoding='utf-8') as f:
#             json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"Created labeling batch with {len(batch_data['evaluations'])} evaluations at {batch_path}")
#         return len(batch_data['evaluations'])


# def prepare_hotpotqa_test_cases(num_samples: int = 20, include_distractors: bool = True) -> List[Dict]:
#     """
#     Load and prepare test cases from HotPotQA dataset.
    
#     Args:
#         num_samples: Number of samples to prepare
#         include_distractors: Whether to include distractor contexts
        
#     Returns:
#         List of test cases with questions, correct/incorrect answers, and contexts
#     """
#     logger.info("Loading HotPotQA dataset...")
    
#     # Load the distractor split which includes misleading contexts
#     dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    
#     test_cases = []
    
#     # Take a sample from the dataset
#     sample_count = 0
#     for example in dataset:
#         if sample_count >= num_samples:
#             break
            
#         # Extract question and correct answer
#         question = example['question']
#         correct_answer = example['answer']
        
#         # Extract supporting facts and contexts
#         supporting_facts = example['supporting_facts']
#         contexts = example['context']
        
#         # Create context string from relevant passages
#         context_parts = []
#         if include_distractors:
#             # Include all contexts (both supporting and distracting)
#             for title, sentences in zip(contexts['title'], contexts['sentences']):
#                 context_parts.append(f"{title}: {' '.join(sentences)}")
#         else:
#             # Include only supporting contexts
#             supporting_titles = set([fact[0] for fact in supporting_facts['title']])
#             for title, sentences in zip(contexts['title'], contexts['sentences']):
#                 if title in supporting_titles:
#                     context_parts.append(f"{title}: {' '.join(sentences)}")
        
#         context_str = "\n\n".join(context_parts[:3])  # Limit context length
        
#         # Create test cases with both correct and incorrect answers
        
#         # 1. Correct answer case
#         test_cases.append({
#             "question": question,
#             "answer": correct_answer,
#             "category": "multi_hop_reasoning",
#             "context": context_str,
#             "hotpotqa_id": example['id'],
#             "expected_correct": True,  # This is the correct answer
#             "answer_type": "correct"
#         })
        
#         # 2. Generate incorrect answers (various types)
#         incorrect_strategies = [
#             # Wrong fact
#             lambda: f"The answer is {random.choice(['1945', '1776', '2000', 'Paris', 'London', 'unknown'])}.",
            
#             # Partial answer (incomplete)
#             lambda: correct_answer.split('.')[0] if '.' in correct_answer else correct_answer[:len(correct_answer)//2],
            
#             # Contradictory answer
#             lambda: f"No, {correct_answer.replace('Yes,', '').replace('is', 'is not').replace('was', 'was not')}",
            
#             # Over-complicated wrong answer
#             lambda: f"Based on the context, {correct_answer} is incorrect. The actual answer involves multiple factors that weren't considered.",
            
#             # Confident but wrong
#             lambda: f"Definitely not {correct_answer}. The correct answer is something completely different."
#         ]
        
#         # Add an incorrect answer variant
#         incorrect_answer = random.choice(incorrect_strategies)()
#         test_cases.append({
#             "question": question,
#             "answer": incorrect_answer,
#             "category": "multi_hop_reasoning",
#             "context": context_str,
#             "hotpotqa_id": example['id'],
#             "expected_correct": False,  # This is an incorrect answer
#             "answer_type": "incorrect_generated"
#         })
        
#         sample_count += 2  # We added 2 test cases
        
#     logger.info(f"Prepared {len(test_cases)} test cases from HotPotQA")
#     return test_cases


# async def main():
#     """Main entry point for data collection."""
#     # Load environment variables
#     env_path = Path(__file__).parent / ".env"
#     load_dotenv(env_path)
#     logger.info(f"Loaded environment variables from {env_path}")

#     # Configure logging
#     configure_logging()

#     # Load configuration
#     try:
#         config = load_config()
#         logger.info("Configuration loaded successfully.")
#     except Exception as e:
#         logger.error(f"Failed to load configuration: {e}")
#         return

#     # Initialize cache backend based on configuration
#     if config.cache.backend == "memory":
#         cache_backend = MemoryCache(max_size=config.cache.max_size)
#     elif config.cache.backend == "disk":
#         cache_backend = DiskCache(cache_dir=config.cache.cache_dir)
#     else:
#         cache_backend = MemoryCache(max_size=10000)

#     # Initialize cache manager
#     cache_manager = CacheManager(
#         primary_backend=cache_backend,
#         namespace="judge",
#         stats_enabled=True
#     )

#     # Initialize provider with API key
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         logger.error("OPENROUTER_API_KEY not found in environment variables")
#         return

#     provider = OpenRouterClient(api_key=api_key)

#     # Initialize model manager
#     model_manager = ModelManager(provider=provider)
#     await model_manager.initialize()

#     # Initialize round manager
#     round_manager = RoundManager(
#         config=config,
#         model_manager=model_manager,
#         cache_manager=cache_manager
#     )

#     logger.info("System components initialized.")
    
#     # Initialize data collector
#     data_dir = Path(__file__).parent / "calibration_data"
#     collector = CalibrationDataCollector(data_dir)
#     logger.info(f"Data collector initialized at {data_dir}")

#     # Prepare test cases from HotPotQA
#     print("\n" + "="*80)
#     print("LOADING HOTPOTQA TEST CASES")
#     print("="*80)
    
#     test_cases = prepare_hotpotqa_test_cases(
#         num_samples=900,  # This will create 20 test cases (10 correct + 10 incorrect)
#         include_distractors=True
#     )
    
#     print(f"Loaded {len(test_cases)} test cases from HotPotQA")
#     print(f"- Correct answers: {sum(1 for tc in test_cases if tc['expected_correct'])}")
#     print(f"- Incorrect answers: {sum(1 for tc in test_cases if not tc['expected_correct'])}")

#     # Run evaluations and collect data
#     evaluation_ids = []

#     for i, test_case in enumerate(test_cases, 1):
#         print(f"\n{'='*80}")
#         print(f"EVALUATION {i}/{len(test_cases)}")
#         print(f"{'='*80}")
#         print(f"Category: {test_case['category']}")
#         print(f"HotPotQA ID: {test_case['hotpotqa_id']}")
#         print(f"Expected: {'CORRECT' if test_case['expected_correct'] else 'INCORRECT'}")
#         print(f"Answer Type: {test_case['answer_type']}")
#         print(f"\nQuestion: {test_case['question']}")
#         print(f"\nAnswer: {test_case['answer']}")
        
#         # Print truncated context
#         context_preview = test_case['context'][:300] + "..." if len(test_case['context']) > 300 else test_case['context']
#         print(f"\nContext Preview: {context_preview}")
        
#         print("-" * 80)
        
#         # Create evaluation request
#         request = EvaluationRequest(
#             question=test_case['question'],
#             answer=test_case['answer'],
#             context=test_case['context'],
#             metadata={
#                 'category': test_case['category'],
#                 'hotpotqa_id': test_case['hotpotqa_id'],
#                 'expected_correct': test_case['expected_correct'],
#                 'answer_type': test_case['answer_type']
#             }
#         )

#         try:
#             # Execute evaluation
#             result = await round_manager.evaluate(request)

#             if result.success and result.judgment:
#                 # Save unlabeled evaluation
#                 eval_id = collector.save_unlabeled_evaluation(result)
#                 evaluation_ids.append(eval_id)
                
#                 # Display results
#                 judgment = result.judgment
#                 jury_says_correct = judgment.is_correct == Verdict.CORRECT
                
#                 print(f"\n📊 JURY EVALUATION:")
#                 print(f"Verdict: {'✓ CORRECT' if jury_says_correct else '✗ INCORRECT'}")
#                 print(f"Expected: {'✓ CORRECT' if test_case['expected_correct'] else '✗ INCORRECT'}")
#                 print(f"Match: {'✅ YES' if (jury_says_correct == test_case['expected_correct']) else '❌ NO'}")
#                 print(f"Confidence: {judgment.raw_confidence:.2%}")
#                 print(f"Consensus Level: {judgment.jury_decision.consensus_level:.2%}")
#                 print(f"Vote Distribution: {judgment.jury_decision.vote_distribution}")
#                 print(f"Processing Time: {judgment.processing_time_ms}ms")
#                 print(f"Total Cost: ${judgment.total_cost:.4f}")
                
#                 print(f"\n💾 Saved for labeling: {eval_id}")
                
#                 # Show executive summary
#                 print(f"\n📝 Executive Summary: {judgment.executive_summary}")
                
#             else:
#                 print(f"❌ Evaluation failed: {result.error}")

#         except Exception as e:
#             logger.error(f"Evaluation failed for test case {i}: {e}")
#             continue

#     # Create labeling batch
#     print(f"\n{'='*80}")
#     print("DATA COLLECTION SUMMARY")
#     print(f"{'='*80}")
    
#     num_saved = len(evaluation_ids)
#     print(f"Total evaluations saved: {num_saved}")
#     print(f"Data directory: {collector.data_dir}")
    
#     if num_saved > 0:
#         # Create batch file for labeling
#         batch_count = collector.create_labeling_batch("hotpotqa_labeling_batch.json")
#         print(f"\nCreated labeling batch with {batch_count} evaluations")
#         print(f"Batch file: {collector.data_dir}/hotpotqa_labeling_batch.json")
        
#         print("\n📋 Next Steps:")
#         print("1. The ground truth labels are already known from HotPotQA")
#         print("2. Review the jury's performance against expected answers")
#         print("3. Use this data to calibrate confidence scores")
#         print("4. Identify patterns where the jury disagrees with ground truth")
    
#     # Cleanup
#     await round_manager.agent_manager.cleanup()
#     await provider.close()

# if __name__ == "__main__":
#     asyncio.run(main())

################################################################################################################################################################

# # -*- coding: utf-8 -*-
# """
# Compute metrics from existing evaluation JSON files in evaluations/ directory
# """

# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List
# from scipy.stats import spearmanr, kendalltau
# import string

# # ======================== CONFIG ===========================
# # Path to your evaluations directory
# EVALUATIONS_DIR = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/evaluationsoo")  # Adjust this path
# OUT_ROOT = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data")
# USE_CALIBRATED_CONF = True
# ECE_BINS = 10
# # ===========================================================

# # ----------------- Normalizers for EM/F1 -------------------
# ARTICLES = {"a", "an", "the"}
# PUNCT = set(string.punctuation)

# def _normalize_text(s: str) -> str:
#     if s is None: return ""
#     s = s.lower().strip()
#     s = "".join(ch for ch in s if ch not in PUNCT)
#     s = " ".join(w for w in s.split() if w not in ARTICLES)
#     return s

# def exact_match(pred: str, gold: str) -> int:
#     return int(_normalize_text(pred) == _normalize_text(gold))

# def _tokenize(s: str) -> List[str]:
#     return _normalize_text(s).split()

# def f1_score_str(pred: str, gold: str) -> float:
#     p, g = _tokenize(pred), _tokenize(gold)
#     if len(p) == 0 and len(g) == 0: return 1.0
#     if len(p) == 0 or len(g) == 0:  return 0.0
#     common = {}
#     for w in g: common[w] = common.get(w, 0) + 1
#     num_same = 0
#     for w in p:
#         if common.get(w, 0) > 0:
#             num_same += 1
#             common[w] -= 1
#     if num_same == 0: return 0.0
#     precision = num_same / len(p)
#     recall    = num_same / len(g)
#     return 2 * precision * recall / (precision + recall)

# # ---------------- Calibration / Agreement ------------------
# def brier_score(y_true, y_prob):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.asarray(y_prob, float)
#     return float(np.mean((y_prob - y_true)**2))

# def ece_score(y_true, y_prob, bins=10):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.clip(np.asarray(y_prob, float), 0.0, 1.0)
#     edges = np.linspace(0.0, 1.0, bins+1)
#     ece = 0.0
#     n = len(y_true)
#     for b in range(bins):
#         lo, hi = edges[b], edges[b+1]
#         idx = (y_prob >= lo) & (y_prob < hi) if b < bins-1 else (y_prob >= lo) & (y_prob <= hi)
#         if np.any(idx):
#             conf = float(np.mean(y_prob[idx]))
#             acc  = float(np.mean(y_true[idx]))
#             ece += (np.sum(idx)/n) * abs(acc - conf)
#     return float(ece)

# def cohen_kappa(y_sys, y_hum):
#     a = np.asarray(y_sys, int)
#     b = np.asarray(y_hum, int)
#     Po = (a == b).mean()
#     p1 = a.mean(); q1 = 1 - p1
#     p2 = b.mean(); q2 = 1 - p2
#     Pe = p1*p2 + q1*q2
#     return (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan

# def fleiss_kappa_from_counts(counts):
#     counts = np.asarray(counts, int)
#     if counts.size == 0: return np.nan
#     N, K = counts.shape
#     n = counts.sum(axis=1)
#     if not np.all(n == n[0]):
#         # Find mode and filter
#         n_mode = pd.Series(n).mode()[0]
#         valid_idx = n == n_mode
#         counts = counts[valid_idx]
#         n = n[valid_idx]
#         if len(n) == 0: return np.nan
#     n = n[0]
#     p_j = counts.sum(axis=0) / (N * n)
#     P_i = ((counts*(counts-1)).sum(axis=1)) / (n*(n-1))
#     P_bar = P_i.mean()
#     P_e = (p_j**2).sum()
#     return (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

# def kripp_alpha_nominal_from_counts(counts):
#     C = np.asarray(counts, float)
#     if C.size == 0: return np.nan
#     n_i = C.sum(axis=1)
#     valid = n_i > 1
#     C = C[valid]; n_i = n_i[valid]
#     if len(n_i) == 0: return np.nan
#     coincidence = (C * (C - 1)).sum(axis=0)
#     Do = 0.0; denom = 0.0
#     for row, n in zip(C, n_i):
#         Do += (n*(n-1) - (row*(row-1)).sum())
#         denom += n*(n-1)
#     Do = Do/denom if denom != 0 else np.nan
#     p = coincidence / coincidence.sum() if coincidence.sum() != 0 else np.zeros_like(coincidence)
#     De = 1.0 - (p**2).sum()
#     if De == 0 or np.isnan(Do): return np.nan
#     return 1.0 - (Do / De)

# # ---------------------- Load Evaluation Files ----------------------
# def load_evaluation_files(eval_dir: Path) -> List[Dict]:
#     """Load all JSON files from evaluations directory"""
#     evaluations = []
    
#     if not eval_dir.exists():
#         raise ValueError(f"Evaluations directory not found: {eval_dir}")
    
#     json_files = list(eval_dir.glob("*.json"))
#     print(f"Found {len(json_files)} evaluation files")
    
#     for json_file in json_files:
#         try:
#             with open(json_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 evaluations.append(data)
#         except Exception as e:
#             print(f"Error loading {json_file}: {e}")
#             continue
    
#     return evaluations

# # ---------------------- Main ----------------------
# def main():
#     # Create output directory
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
#     # Load evaluation files
#     print(f"Loading evaluations from: {EVALUATIONS_DIR}")
#     evaluations = load_evaluation_files(EVALUATIONS_DIR)
    
#     if not evaluations:
#         print("No evaluation files found!")
#         return
    
#     print(f"Loaded {len(evaluations)} evaluations")
    
#     # Process each evaluation
#     per_item = []
    
#     for eval_data in evaluations:
#         try:
#             # Extract basic info
#             eval_id = eval_data.get('eval_id', 'unknown')
#             item_id = eval_data.get('id', 'unknown')
            
#             # Get answers
#             pred = eval_data.get('answer', '')  # Model answer (judged)
#             gold = eval_data.get('gold_answer', '')  # Gold answer
            
#             # If we already have computed metrics in the file, use them
#             if 'ground_truth' in eval_data:
#                 ans_em = eval_data['ground_truth'].get('Answer_EM', 0)
#                 ans_f1 = eval_data['ground_truth'].get('Answer_F1', 0.0)
#             else:
#                 # Otherwise compute them
#                 ans_em = exact_match(pred, gold)
#                 ans_f1 = f1_score_str(pred, gold)
            
#             # Human verdict (from metrics or compute)
#             human_verdict = eval_data.get('human_verdict', 'correct' if ans_em == 1 else 'incorrect')
#             human_score = eval_data.get('human_score', float(ans_f1))
            
#             # System verdict and confidence
#             sys_verdict = eval_data.get('jury_verdict', 'unknown')
#             sys_conf = float(eval_data.get('jury_calibrated_confidence', 
#                                           eval_data.get('jury_confidence', 0.0)))
            
#             # Vote distribution
#             vote_dist = {}
#             if 'jury_decision' in eval_data and 'vote_distribution' in eval_data['jury_decision']:
#                 vote_dist = eval_data['jury_decision']['vote_distribution']
            
#             votes_c = int(vote_dist.get('correct', 0))
#             votes_i = int(vote_dist.get('incorrect', 0))
#             votes_u = int(vote_dist.get('uncertain', 0))
            
#             per_item.append({
#                 "id": item_id,
#                 "eval_id": eval_id,
#                 "sys_verdict": sys_verdict,
#                 "sys_conf": sys_conf,
#                 "votes_c": votes_c,
#                 "votes_i": votes_i,
#                 "votes_u": votes_u,
#                 "Answer_EM": ans_em,
#                 "Answer_F1": ans_f1,
#                 "human_verdict": human_verdict,
#                 "human_score": human_score,
#                 "question": eval_data.get('question', ''),
#                 "model_answer": pred,
#                 "gold_answer": gold
#             })
            
#         except Exception as e:
#             print(f"Error processing evaluation {eval_data.get('eval_id', 'unknown')}: {e}")
#             continue
    
#     # Convert to DataFrame
#     per = pd.DataFrame(per_item)
    
#     if per.empty:
#         print("No valid data extracted!")
#         return
    
#     print(f"\nProcessed {len(per)} evaluations successfully")
    
#     # Save per-item results
#     per.to_csv(OUT_ROOT / "per_item_dump.csv", index=False)
#     print(f"Saved per-item results to: {OUT_ROOT / 'per_item_dump.csv'}")
    
#     # Binary labels
#     per["hum_label"] = (per["human_verdict"].str.lower() == "correct").astype(int)
#     per["sys_label"] = (per["sys_verdict"].str.lower() == "correct").astype(int)
    
#     # Compute aggregate metrics
#     mask_bin = per["hum_label"].notna() & per["sys_label"].notna()
    
#     # Accuracy and Cohen's kappa
#     acc = float((per.loc[mask_bin, "hum_label"] == per.loc[mask_bin, "sys_label"]).mean()) if mask_bin.any() else np.nan
#     kappa = cohen_kappa(per.loc[mask_bin, "sys_label"], per.loc[mask_bin, "hum_label"]) if mask_bin.any() else np.nan
    
#     # Correlation metrics
#     mask_cont = per["sys_conf"].notna() & per["human_score"].notna()
#     spearman_r = spearman_p = kendall_t = kendall_p = np.nan
#     if mask_cont.any():
#         sr = spearmanr(per.loc[mask_cont, "sys_conf"], per.loc[mask_cont, "human_score"])
#         spearman_r, spearman_p = float(sr.correlation), float(sr.pvalue)
#         kt = kendalltau(per.loc[mask_cont, "sys_conf"], per.loc[mask_cont, "human_score"], variant="b")
#         kendall_t, kendall_p = float(kt.correlation), float(kt.pvalue)
    
#     # Calibration metrics
#     brier = ece = np.nan
#     if mask_bin.any():
#         y_true = per.loc[mask_bin, "hum_label"].astype(int).to_numpy()
#         y_prob = per.loc[mask_bin, "sys_conf"].astype(float).to_numpy()
#         brier = brier_score(y_true, y_prob)
#         ece = ece_score(y_true, y_prob, bins=ECE_BINS)
    
#     # Inter-rater agreement metrics
#     fkappa = np.nan
#     alpha_nominal = np.nan
#     if {"votes_c", "votes_i", "votes_u"}.issubset(per.columns):
#         tmp = per[["votes_c", "votes_i", "votes_u"]].fillna(0).astype(int)
#         if not tmp.empty:
#             try:
#                 # Fleiss kappa
#                 fkappa = fleiss_kappa_from_counts(tmp.to_numpy())
#             except Exception as e:
#                 print(f"Fleiss κ skipped: {e}")
#             # Krippendorff's alpha
#             try:
#                 alpha_nominal = kripp_alpha_nominal_from_counts(tmp.to_numpy())
#             except Exception as e:
#                 print(f"Krippendorff's α skipped: {e}")
    
#     # Create summary
#     summary = pd.DataFrame([{
#         "N_total": len(per),
#         "Answer_EM": float(per["Answer_EM"].mean()),
#         "Answer_F1": float(per["Answer_F1"].mean()),
#         "Verdict_Accuracy": acc,
#         "Cohen_kappa": kappa,
#         "Spearman_r": spearman_r,
#         "Spearman_p": spearman_p,
#         "Kendall_tau_b": kendall_t,
#         "Kendall_p": kendall_p,
#         "Brier": brier,
#         "ECE@10": ece,
#         "Fleiss_kappa_jurors": fkappa,
#         "Krippendorff_alpha_nominal": alpha_nominal
#     }])
    
#     # Save summary
#     summary.to_csv(OUT_ROOT / "summary_metrics.csv", index=False)
#     print(f"Saved summary metrics to: {OUT_ROOT / 'summary_metrics.csv'}")
    
#     # Print results
#     print("\n" + "="*60)
#     print("SUMMARY METRICS")
#     print("="*60)
#     print(summary.to_string(index=False))
#     print("="*60)
    
#     # Print detailed breakdown
#     print(f"\nDetailed Breakdown:")
#     print(f"- Total samples: {len(per)}")
#     print(f"- Average Answer EM: {per['Answer_EM'].mean():.3f}")
#     print(f"- Average Answer F1: {per['Answer_F1'].mean():.3f}")
#     if not np.isnan(acc):
#         print(f"- System accuracy: {acc:.3f}")
#     if not np.isnan(kappa):
#         print(f"- System-Human agreement (Cohen κ): {kappa:.3f}")
#     if not np.isnan(ece):
#         print(f"- Calibration error (ECE): {ece:.3f}")
#     if not np.isnan(brier):
#         print(f"- Calibration error (Brier): {brier:.3f}")
    
#     # Print verdict distribution
#     print(f"\nVerdict Distribution:")
#     print(f"- System says correct: {(per['sys_verdict'].str.lower() == 'correct').sum()} ({(per['sys_verdict'].str.lower() == 'correct').mean()*100:.1f}%)")
#     print(f"- Human says correct: {(per['human_verdict'].str.lower() == 'correct').sum()} ({(per['human_verdict'].str.lower() == 'correct').mean()*100:.1f}%)")

# if __name__ == "__main__":
#     main()



#################################################################################################################################################################

# # -*- coding: utf-8 -*-
# """
# Compute metrics from existing evaluation JSON files in evaluations/ directory
# with OpenRouter integration for HotPotQA dataset
# """

# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List, Optional
# from scipy.stats import spearmanr, kendalltau
# import string
# import logging
# import requests
# from datasets import load_dataset
# import time
# from tqdm import tqdm
# import pickle
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# # ======================== CONFIG ===========================
# # Path to your evaluations directory
# EVALUATIONS_DIR = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/evaluations")
# OUT_ROOT = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data")
# USE_CALIBRATED_CONF = True
# ECE_BINS = 10

# # Dataset cache directory
# DATASET_CACHE_DIR = OUT_ROOT / "hotpotqa_cache"
# DATASET_CACHE_FILE = DATASET_CACHE_DIR / "hotpotqa_dataset.pkl"
# ANSWERS_CACHE_FILE = DATASET_CACHE_DIR / "gpt4_answers.json"

# # OpenRouter Configuration
# OPENROUTER_API_KEY = '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'
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "x-ai/grok-4-fast:free"  # Changed to GPT-4 as per request
# # Alternative free model if you want to save costs: "nvidia/nemotron-nano-9b-v2:free"

# # Dataset Configuration
# DATASET_NAME = "hotpotqa/hotpot_qa"
# DATASET_CONFIG = "distractor"
# DATASET_SPLIT = "validation"
# MAX_SAMPLES = None  # Set to None to process all, or a number to limit samples
# # ===========================================================

# # ----------------- Normalizers for EM/F1 -------------------
# ARTICLES = {"a", "an", "the"}
# PUNCT = set(string.punctuation)

# def _normalize_text(s: str) -> str:
#     if s is None: return ""
#     s = s.lower().strip()
#     s = "".join(ch for ch in s if ch not in PUNCT)
#     s = " ".join(w for w in s.split() if w not in ARTICLES)
#     return s

# def exact_match(pred: str, gold: str) -> int:
#     return int(_normalize_text(pred) == _normalize_text(gold))

# def _tokenize(s: str) -> List[str]:
#     return _normalize_text(s).split()

# def f1_score_str(pred: str, gold: str) -> float:
#     p, g = _tokenize(pred), _tokenize(gold)
#     if len(p) == 0 and len(g) == 0: return 1.0
#     if len(p) == 0 or len(g) == 0:  return 0.0
#     common = {}
#     for w in g: common[w] = common.get(w, 0) + 1
#     num_same = 0
#     for w in p:
#         if common.get(w, 0) > 0:
#             num_same += 1
#             common[w] -= 1
#     if num_same == 0: return 0.0
#     precision = num_same / len(p)
#     recall    = num_same / len(g)
#     return 2 * precision * recall / (precision + recall)

# # ---------------- Calibration / Agreement ------------------
# def brier_score(y_true, y_prob):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.asarray(y_prob, float)
#     return float(np.mean((y_prob - y_true)**2))

# def ece_score(y_true, y_prob, bins=10):
#     y_true = np.asarray(y_true, float)
#     y_prob = np.clip(np.asarray(y_prob, float), 0.0, 1.0)
#     edges = np.linspace(0.0, 1.0, bins+1)
#     ece = 0.0
#     n = len(y_true)
#     for b in range(bins):
#         lo, hi = edges[b], edges[b+1]
#         idx = (y_prob >= lo) & (y_prob < hi) if b < bins-1 else (y_prob >= lo) & (y_prob <= hi)
#         if np.any(idx):
#             conf = float(np.mean(y_prob[idx]))
#             acc  = float(np.mean(y_true[idx]))
#             ece += (np.sum(idx)/n) * abs(acc - conf)
#     return float(ece)

# def cohen_kappa(y_sys, y_hum):
#     a = np.asarray(y_sys, int)
#     b = np.asarray(y_hum, int)
#     Po = (a == b).mean()
#     p1 = a.mean(); q1 = 1 - p1
#     p2 = b.mean(); q2 = 1 - p2
#     Pe = p1*p2 + q1*q2
#     return (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan

# def fleiss_kappa_from_counts(counts):
#     counts = np.asarray(counts, int)
#     if counts.size == 0: return np.nan
#     N, K = counts.shape
#     n = counts.sum(axis=1)
#     if not np.all(n == n[0]):
#         # Find mode and filter
#         n_mode = pd.Series(n).mode()[0]
#         valid_idx = n == n_mode
#         counts = counts[valid_idx]
#         n = n[valid_idx]
#         if len(n) == 0: return np.nan
#     n = n[0]
#     p_j = counts.sum(axis=0) / (N * n)
#     P_i = ((counts*(counts-1)).sum(axis=1)) / (n*(n-1))
#     P_bar = P_i.mean()
#     P_e = (p_j**2).sum()
#     return (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

# def kripp_alpha_nominal_from_counts(counts):
#     C = np.asarray(counts, float)
#     if C.size == 0: return np.nan
#     n_i = C.sum(axis=1)
#     valid = n_i > 1
#     C = C[valid]; n_i = n_i[valid]
#     if len(n_i) == 0: return np.nan
#     coincidence = (C * (C - 1)).sum(axis=0)
#     Do = 0.0; denom = 0.0
#     for row, n in zip(C, n_i):
#         Do += (n*(n-1) - (row*(row-1)).sum())
#         denom += n*(n-1)
#     Do = Do/denom if denom != 0 else np.nan
#     p = coincidence / coincidence.sum() if coincidence.sum() != 0 else np.zeros_like(coincidence)
#     De = 1.0 - (p**2).sum()
#     if De == 0 or np.isnan(Do): return np.nan
#     return 1.0 - (Do / De)

# # ---------------------- Dataset Cache Management ----------------------
# def save_dataset_to_cache(dataset, cache_file: Path):
#     """Save dataset to pickle file"""
#     cache_file.parent.mkdir(parents=True, exist_ok=True)
#     with open(cache_file, 'wb') as f:
#         pickle.dump(dataset, f)
#     logger.info(f"Dataset saved to cache: {cache_file}")

# def load_dataset_from_cache(cache_file: Path):
#     """Load dataset from pickle file"""
#     if cache_file.exists():
#         logger.info(f"Loading dataset from cache: {cache_file}")
#         with open(cache_file, 'rb') as f:
#             return pickle.load(f)
#     return None

# def save_answers_to_cache(answers: Dict[str, str], cache_file: Path):
#     """Save GPT-4 answers to JSON file"""
#     cache_file.parent.mkdir(parents=True, exist_ok=True)
#     with open(cache_file, 'w', encoding='utf-8') as f:
#         json.dump(answers, f, indent=2, ensure_ascii=False)
#     logger.info(f"Answers saved to cache: {cache_file}")

# def load_answers_from_cache(cache_file: Path) -> Dict[str, str]:
#     """Load GPT-4 answers from JSON file"""
#     if cache_file.exists():
#         logger.info(f"Loading answers from cache: {cache_file}")
#         with open(cache_file, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     return {}

# # ---------------------- OpenRouter Integration ----------------------
# def format_context_for_prompt(context: Dict) -> str:
#     """Format the context dictionary into a readable string"""
#     context_text = ""
    
#     if 'title' in context and 'sentences' in context:
#         titles = context['title']
#         sentences_list = context['sentences']
        
#         for i, (title, sentences) in enumerate(zip(titles, sentences_list)):
#             context_text += f"\nDocument {i+1}: {title}\n"
#             for j, sentence in enumerate(sentences):
#                 context_text += f"  - {sentence}\n"
    
#     return context_text.strip()

# def query_openrouter(question: str, context: Dict, max_retries: int = 3) -> Optional[str]:
#     """Query OpenRouter API with the question and context"""
    
#     # Format the context
#     context_text = format_context_for_prompt(context)
    
#     # Create the prompt
#     prompt = f"""Based on the following context, please answer the question.

# Context:
# {context_text}

# Question: {question}

# Please provide a clear and concise answer based only on the information provided in the context."""
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": "https://github.com/yourusername/yourproject",  # Optional but recommended
#     }
    
#     data = {
#         "model": MODEL_NAME,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and concise."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.7,
#         "max_tokens": 200
#     }
    
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(
#                 OPENROUTER_API_URL,
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 return result['choices'][0]['message']['content'].strip()
#             else:
#                 logger.error(f"API request failed with status {response.status_code}: {response.text}")
#                 if attempt < max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff
                
#         except Exception as e:
#             logger.error(f"Error querying OpenRouter: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(2 ** attempt)
    
#     return None

# # ---------------------- Dataset Processing ----------------------
# def process_hotpotqa_dataset():
#     """Load and process HotPotQA dataset with OpenRouter queries"""
    
#     # Try to load dataset from cache first
#     dataset = load_dataset_from_cache(DATASET_CACHE_FILE)
    
#     if dataset is None:
#         logger.info(f"Loading {DATASET_NAME} dataset from Hugging Face...")
#         dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
        
#         # Convert to list for easier handling
#         dataset = list(dataset)
        
#         # Save to cache
#         save_dataset_to_cache(dataset, DATASET_CACHE_FILE)
    
#     if MAX_SAMPLES:
#         dataset = dataset[:min(MAX_SAMPLES, len(dataset))]
    
#     logger.info(f"Processing {len(dataset)} samples...")
    
#     # Load existing answers
#     existing_answers = load_answers_from_cache(ANSWERS_CACHE_FILE)
#     logger.info(f"Found {len(existing_answers)} existing answers in cache")
    
#     evaluations = []
#     all_answers = existing_answers.copy()
#     new_answers_count = 0
    
#     for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
#         try:
#             # Extract data from the dataset
#             item_id = item['id']
#             question = item['question']
#             gold_answer = item['answer']
#             context = item['context']
#             supporting_facts = item['supporting_facts']
            
#             # Check if we already have the answer
#             if item_id in existing_answers:
#                 model_answer = existing_answers[item_id]
#                 logger.debug(f"Using cached answer for {item_id}")
#             else:
#                 # Query OpenRouter for model's answer
#                 model_answer = query_openrouter(question, context)
                
#                 if model_answer is None:
#                     logger.warning(f"Failed to get response for item {item_id}")
#                     continue
                
#                 # Save the new answer
#                 all_answers[item_id] = model_answer
#                 new_answers_count += 1
                
#                 # Save periodically (every 10 new answers)
#                 if new_answers_count % 10 == 0:
#                     save_answers_to_cache(all_answers, ANSWERS_CACHE_FILE)
                
#                 # Add delay to avoid rate limiting
#                 time.sleep(0.5)
            
#             # Compute metrics
#             ans_em = exact_match(model_answer, gold_answer)
#             ans_f1 = f1_score_str(model_answer, gold_answer)
            
#             # Create evaluation entry with all dataset fields
#             eval_entry = {
#                 'eval_id': f'eval_{idx:04d}',
#                 'id': item_id,
#                 'question': question,
#                 'answer': model_answer,  # Model's answer
#                 'gold_answer': gold_answer,  # Ground truth
#                 'type': item.get('type', ''),
#                 'level': item.get('level', ''),
#                 'context': context,
#                 'supporting_facts': supporting_facts,
#                 'ground_truth': {
#                     'Answer_EM': ans_em,
#                     'Answer_F1': ans_f1
#                 },
#                 'human_verdict': 'correct' if ans_em == 1 else 'incorrect',
#                 'human_score': float(ans_f1),
#                 # These fields would be filled by your jury system
#                 'jury_verdict': 'correct' if ans_em > 0.5 else 'incorrect',  # Simple placeholder
#                 'jury_confidence': 0.8 if ans_em == 1 else 0.3,  # Placeholder confidence
#                 'jury_calibrated_confidence': 0.8 if ans_em == 1 else 0.3,
#                 'jury_decision': {
#                     'vote_distribution': {
#                         'correct': 3 if ans_em == 1 else 1,
#                         'incorrect': 1 if ans_em == 1 else 3,
#                         'uncertain': 1
#                     }
#                 }
#             }
            
#             evaluations.append(eval_entry)
            
#         except Exception as e:
#             logger.error(f"Error processing item {idx}: {e}")
#             continue
    
#     # Save final answers
#     if new_answers_count > 0:
#         save_answers_to_cache(all_answers, ANSWERS_CACHE_FILE)
#         logger.info(f"Added {new_answers_count} new answers to cache")
    
#     # Save dataset with answers as a separate file
#     dataset_with_answers = []
#     for item in dataset:
#         item_copy = item.copy()
#         item_copy['model_answer'] = all_answers.get(item['id'], '')
#         dataset_with_answers.append(item_copy)
    
#     dataset_with_answers_file = DATASET_CACHE_DIR / "hotpotqa_with_gpt4_answers.json"
#     with open(dataset_with_answers_file, 'w', encoding='utf-8') as f:
#         json.dump(dataset_with_answers, f, indent=2, ensure_ascii=False)
#     logger.info(f"Saved dataset with GPT-4 answers to: {dataset_with_answers_file}")
    
#     return evaluations

# # ---------------------- Load Evaluation Files ----------------------
# def load_evaluation_files(eval_dir: Path) -> List[Dict]:
#     """Load all JSON files from evaluations directory"""
#     evaluations = []
    
#     if not eval_dir.exists():
#         logger.error(f"Evaluations directory not found: {eval_dir}")
#         return evaluations
    
#     json_files = list(eval_dir.glob("*.json"))
#     logger.info(f"Found {len(json_files)} evaluation files in {eval_dir}")
    
#     for json_file in sorted(json_files):
#         try:
#             with open(json_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 evaluations.append(data)
#         except Exception as e:
#             logger.error(f"Error loading {json_file}: {e}")
#             continue
    
#     return evaluations

# # ---------------------- Save Evaluations ----------------------
# def save_evaluations(evaluations: List[Dict], output_dir: Path):
#     """Save evaluation results to JSON files"""
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     for i, eval_data in enumerate(evaluations):
#         output_file = output_dir / f"eval_{i:04d}.json"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
#     logger.info(f"Saved {len(evaluations)} evaluation files to {output_dir}")

# # ---------------------- Main ----------------------
# def main():
#     # Create output directories
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
#     # Check if we should process HotPotQA dataset
#     process_new_data = input("Do you want to process new data from HotPotQA dataset? (y/n): ").lower() == 'y'
    
#     if process_new_data:
#         if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
#             logger.error("Please set your OpenRouter API key in the config section!")
#             return
        
#         # Process dataset with OpenRouter
#         evaluations = process_hotpotqa_dataset()
        
#         # Save evaluations
#         if evaluations:
#             save_evaluations(evaluations, EVALUATIONS_DIR)
#             logger.info(f"Processed and saved {len(evaluations)} evaluations")
    
#     # Load evaluation files (either newly created or existing)
#     logger.info(f"Loading evaluations from: {EVALUATIONS_DIR}")
#     evaluations = load_evaluation_files(EVALUATIONS_DIR)
    
#     if not evaluations:
#         logger.warning("No evaluation files found!")
#         logger.info("Make sure to run the main script first to generate evaluations.")
#         return
    
#     logger.info(f"Loaded {len(evaluations)} evaluations")
    
#     # Process each evaluation
#     per_item = []
    
#     for eval_data in evaluations:
#         try:
#             # Extract basic info
#             eval_id = eval_data.get('eval_id', 'unknown')
#             item_id = eval_data.get('id', 'unknown')
            
#             # Get answers
#             pred = eval_data.get('answer', '')  # Model answer (judged)
#             gold = eval_data.get('gold_answer', '')  # Gold answer
            
#             # If we already have computed metrics in the file, use them
#             if 'ground_truth' in eval_data:
#                 ans_em = eval_data['ground_truth'].get('Answer_EM', 0)
#                 ans_f1 = eval_data['ground_truth'].get('Answer_F1', 0.0)
#             else:
#                 # Otherwise compute them
#                 ans_em = exact_match(pred, gold)
#                 ans_f1 = f1_score_str(pred, gold)
            
#             # Human verdict (from metrics or compute)
#             human_verdict = eval_data.get('human_verdict', 'correct' if ans_em == 1 else 'incorrect')
#             human_score = eval_data.get('human_score', float(ans_f1))
            
#             # System verdict and confidence
#             sys_verdict = eval_data.get('jury_verdict', 'unknown')
#             sys_conf = float(eval_data.get('jury_calibrated_confidence', 
#                                           eval_data.get('jury_confidence', 0.0)))
            
#             # Vote distribution
#             vote_dist = {}
#             if 'jury_decision' in eval_data and 'vote_distribution' in eval_data['jury_decision']:
#                 vote_dist = eval_data['jury_decision']['vote_distribution']
            
#             votes_c = int(vote_dist.get('correct', 0))
#             votes_i = int(vote_dist.get('incorrect', 0))
#             votes_u = int(vote_dist.get('uncertain', 0))
            
#             per_item.append({
#                 "id": item_id,
#                 "eval_id": eval_id,
#                 "sys_verdict": sys_verdict,
#                 "sys_conf": sys_conf,
#                 "votes_c": votes_c,
#                 "votes_i": votes_i,
#                 "votes_u": votes_u,
#                 "Answer_EM": ans_em,
#                 "Answer_F1": ans_f1,
#                 "human_verdict": human_verdict,
#                 "human_score": human_score,
#                 "question": eval_data.get('question', ''),
#                 "model_answer": pred,
#                 "gold_answer": gold,
#                 "type": eval_data.get('type', ''),
#                 "level": eval_data.get('level', '')
#             })
            
#         except Exception as e:
#             logger.error(f"Error processing evaluation {eval_data.get('eval_id', 'unknown')}: {e}")
#             continue
    
#     # Convert to DataFrame
#     per = pd.DataFrame(per_item)
    
#     if per.empty:
#         logger.error("No valid data extracted!")
#         return
    
#     logger.info(f"\nProcessed {len(per)} evaluations successfully")
    
#     # Save per-item results
#     per_item_file = OUT_ROOT / "per_item_dump.csv"
#     per.to_csv(per_item_file, index=False)
#     logger.info(f"Saved per-item results to: {per_item_file}")
    
#     # Binary labels
#     per["hum_label"] = (per["human_verdict"].str.lower() == "correct").astype(int)
#     per["sys_label"] = (per["sys_verdict"].str.lower() == "correct").astype(int)
    
#     # Compute aggregate metrics
#     mask_bin = per["hum_label"].notna() & per["sys_label"].notna()
    
#     # Accuracy and Cohen's kappa
#     acc = float((per.loc[mask_bin, "hum_label"] == per.loc[mask_bin, "sys_label"]).mean()) if mask_bin.any() else np.nan
#     kappa = cohen_kappa(per.loc[mask_bin, "sys_label"], per.loc[mask_bin, "hum_label"]) if mask_bin.any() else np.nan
    
#     # Correlation metrics
#     mask_cont = per["sys_conf"].notna() & per["human_score"].notna()
#     spearman_r = spearman_p = kendall_t = kendall_p = np.nan
#     if mask_cont.any():
#         sr = spearmanr(per.loc[mask_cont, "sys_conf"], per.loc[mask_cont, "human_score"])
#         spearman_r, spearman_p = float(sr.correlation), float(sr.pvalue)
#         kt = kendalltau(per.loc[mask_cont, "sys_conf"], per.loc[mask_cont, "human_score"], variant="b")
#         kendall_t, kendall_p = float(kt.correlation), float(kt.pvalue)
    
#     # Calibration metrics
#     brier = ece = np.nan
#     if mask_bin.any():
#         y_true = per.loc[mask_bin, "hum_label"].astype(int).to_numpy()
#         y_prob = per.loc[mask_bin, "sys_conf"].astype(float).to_numpy()
#         brier = brier_score(y_true, y_prob)
#         ece = ece_score(y_true, y_prob, bins=ECE_BINS)
    
#     # Inter-rater agreement metrics
#     fkappa = np.nan
#     alpha_nominal = np.nan
#     if {"votes_c", "votes_i", "votes_u"}.issubset(per.columns):
#         tmp = per[["votes_c", "votes_i", "votes_u"]].fillna(0).astype(int)
#         if not tmp.empty:
#             try:
#                 # Fleiss kappa
#                 fkappa = fleiss_kappa_from_counts(tmp.to_numpy())
#             except Exception as e:
#                 logger.warning(f"Fleiss κ skipped: {e}")
#             # Krippendorff's alpha
#             try:
#                 alpha_nominal = kripp_alpha_nominal_from_counts(tmp.to_numpy())
#             except Exception as e:
#                 logger.warning(f"Krippendorff's α skipped: {e}")
    
#     # Create summary
#     summary = pd.DataFrame([{
#         "N_total": len(per),
#         "Answer_EM": float(per["Answer_EM"].mean()),
#         "Answer_F1": float(per["Answer_F1"].mean()),
#         "Verdict_Accuracy": acc,
#         "Cohen_kappa": kappa,
#         "Spearman_r": spearman_r,
#         "Spearman_p": spearman_p,
#         "Kendall_tau_b": kendall_t,
#         "Kendall_p": kendall_p,
#         "Brier": brier,
#         "ECE@10": ece,
#         "Fleiss_kappa_jurors": fkappa,
#         "Krippendorff_alpha_nominal": alpha_nominal
#     }])
    
#     # Save summary
#     summary_file = OUT_ROOT / "summary_metrics.csv"
#     summary.to_csv(summary_file, index=False)
#     logger.info(f"Saved summary metrics to: {summary_file}")
    
#     # Print results
#     print("\n" + "="*60)
#     print("SUMMARY METRICS")
#     print("="*60)
#     print(summary.to_string(index=False))
#     print("="*60)
    
#     # Print detailed breakdown
#     print(f"\nDetailed Breakdown:")
#     print(f"- Total samples: {len(per)}")
#     print(f"- Average Answer EM: {per['Answer_EM'].mean():.3f}")
#     print(f"- Average Answer F1: {per['Answer_F1'].mean():.3f}")
#     if not np.isnan(acc):
#         print(f"- System accuracy: {acc:.3f}")
#     if not np.isnan(kappa):
#         print(f"- System-Human agreement (Cohen κ): {kappa:.3f}")
#     if not np.isnan(ece):
#         print(f"- Calibration error (ECE): {ece:.3f}")
#     if not np.isnan(brier):
#         print(f"- Calibration error (Brier): {brier:.3f}")
    
#     # Print verdict distribution
#     print(f"\nVerdict Distribution:")
#     print(f"- System says correct: {(per['sys_verdict'].str.lower() == 'correct').sum()} ({(per['sys_verdict'].str.lower() == 'correct').mean()*100:.1f}%)")
#     print(f"- Human says correct: {(per['human_verdict'].str.lower() == 'correct').sum()} ({(per['human_verdict'].str.lower() == 'correct').mean()*100:.1f}%)")
    
#     # Print breakdown by type and level if available
#     if 'type' in per.columns and per['type'].notna().any():
#         print(f"\nBreakdown by Question Type:")
#         type_stats = per.groupby('type')['Answer_EM'].agg(['count', 'mean'])
#         print(type_stats.to_string())
    
#     if 'level' in per.columns and per['level'].notna().any():
#         print(f"\nBreakdown by Difficulty Level:")
#         level_stats = per.groupby('level')['Answer_EM'].agg(['count', 'mean'])
#         print(level_stats.to_string())

# if __name__ == "__main__":
#     main()
############################################################################################





















# evaluate_hotpotqa.py
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, brier_score_loss
from scipy.stats import spearmanr, kendalltau
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff
from loguru import logger
import sys
import os
from dotenv import load_dotenv
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from the specific .env file
env_path = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/.env")
load_dotenv(env_path)

from multi_agent_llm_judge.core.round_manager import RoundManager
from multi_agent_llm_judge.core.model_manager import ModelManager
from multi_agent_llm_judge.core.cache_manager import CacheManager, MemoryCache, DiskCache
from multi_agent_llm_judge.core.data_models import EvaluationRequest, Verdict, EvaluationResult, FinalJudgment
from multi_agent_llm_judge.providers.openrouter import OpenRouterClient
from multi_agent_llm_judge.utils.config_loader import load_config

class CalibrationDataCollector:
    """Collects evaluation data for future calibration training."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Different directories for data types
        self.evaluations_dir = self.data_dir / "evaluations"
        self.evaluations_dir.mkdir(exist_ok=True)
    
    def save_evaluation_with_ground_truth(self, result: EvaluationResult, ground_truth: bool, example_id: str) -> str:
        """
        Save evaluation result with ground truth label.
        
        Args:
            result: Evaluation result from jury system
            ground_truth: Whether the answer is actually correct
            example_id: ID of the example
            
        Returns:
            Unique ID of saved evaluation
        """
        if not result.success or not result.judgment:
            logger.error("Cannot save failed evaluation")
            return None
            
        # Generate unique ID for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{timestamp}_{example_id}"
        
        # Extract evaluation data
        judgment = result.judgment
        jury_says_correct = judgment.is_correct == Verdict.CORRECT
        
        eval_data = {
            'eval_id': eval_id,
            'example_id': example_id,
            'timestamp': datetime.now().isoformat(),
            
            # Question and answer
            'question': judgment.question,
            'answer': judgment.answer,
            'context': judgment.context,
            
            # Ground truth
            'ground_truth': ground_truth,
            'is_answer_correct': ground_truth,  # For clarity
            
            # Jury results
            'jury_verdict': judgment.is_correct.value,
            'jury_says_correct': jury_says_correct,
            'verdict_correct': jury_says_correct == ground_truth,  # Whether jury was right
            'jury_confidence': judgment.raw_confidence,
            'jury_calibrated_confidence': judgment.calibrated_confidence,
            
            # Jury decision details
            'jury_decision': {
                'majority_verdict': judgment.jury_decision.majority_verdict.value,
                'consensus_level': judgment.jury_decision.consensus_level,
                'vote_distribution': judgment.jury_decision.vote_distribution,
                'weighted_confidence': judgment.jury_decision.weighted_confidence,
            },
            
            # Performance metrics
            'metrics': {
                'total_cost': judgment.total_cost,
                'total_tokens': judgment.total_tokens,
                'processing_time_ms': judgment.processing_time_ms,
                'models_used': judgment.models_used,
                'num_rounds': len(judgment.round_summaries),
                'num_agents': sum(len(rs.agent_analyses) for rs in judgment.round_summaries)
            },
            
            # Agent verdicts for inter-rater agreement
            'agent_verdicts': [
                {
                    'agent': a.agent_name,
                    'verdict': 1 if a.verdict == Verdict.CORRECT else 0
                }
                for rs in judgment.round_summaries
                for a in rs.agent_analyses
                if a.verdict is not None
            ]
        }
        
        # Save evaluation
        eval_path = self.evaluations_dir / f"{eval_id}.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation to {eval_path}")
        return eval_id

class HotPotQAEvaluator:
    def __init__(self, data_path: str, config_path: str = None):
        self.data_path = Path(data_path)
        self.results = []
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Initialize data collector
        self.collector = CalibrationDataCollector(Path("calibration_data/hotpotqa"))
        
    async def initialize_pipeline(self):
        """Initialize the multi-agent judge pipeline."""
        # Initialize provider
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        logger.info(f"Using OpenRouter API key: {api_key[:10]}...")
        
        self.provider = OpenRouterClient(api_key=api_key)
        
        # Initialize managers
        self.model_manager = ModelManager(provider=self.provider)
        await self.model_manager.initialize()
        
        # Initialize cache - using disk cache to persist between runs
        cache_backend = DiskCache(cache_dir=".cache/hotpotqa")
        self.cache_manager = CacheManager(
            primary_backend=cache_backend,
            namespace="hotpotqa_judge",
            stats_enabled=True
        )
        
        # Initialize round manager
        self.round_manager = RoundManager(
            config=self.config,
            model_manager=self.model_manager,
            cache_manager=self.cache_manager
        )
        
        logger.info("Pipeline initialized successfully")
    
    def construct_context_from_sentences(self, example: Dict[str, Any]) -> str:
        """Construct context text from your dataset format."""
        context_text = []
        
        # Get context data
        context_data = example.get("context", {})
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])
        
        # Get supporting facts for marking
        supporting_facts = example.get("supporting_facts", {})
        supporting_titles = supporting_facts.get("title", [])
        supporting_sent_ids = supporting_facts.get("sent_id", [])
        
        # Create a set of supporting facts for easier lookup
        supporting_set = set(zip(supporting_titles, supporting_sent_ids))
        
        for title, sentences in zip(titles, sentences_list):
            # Add title
            context_text.append(f"\n{title}:")
            
            # Add sentences with marking for supporting facts
            for sent_id, sentence in enumerate(sentences):
                if (title, sent_id) in supporting_set:
                    context_text.append(f"  [SUPPORTING] {sentence}")
                else:
                    context_text.append(f"  {sentence}")
        
        return "\n".join(context_text)
    
    def extract_detailed_round_analysis(self, judgment: FinalJudgment) -> Dict[str, Any]:
        """Extract detailed round-by-round analysis including agent verdicts and confidences."""
        round_details = []
        all_agent_details = []
        
        for round_idx, round_summary in enumerate(judgment.round_summaries):
            round_data = {
                'round_number': round_idx + 1,
                'agents': [],
                'round_summary': {
                    'total_agents': len(round_summary.agent_analyses),
                    'verdicts': {'correct': 0, 'incorrect': 0, 'uncertain': 0},
                    'avg_confidence': 0.0,
                    'models_used': []
                }
            }
            
            confidences = []
            
            for agent_analysis in round_summary.agent_analyses:
                # Debug: Log available attributes
                logger.debug(f"AgentAnalysis attributes for {agent_analysis.agent_name}: {dir(agent_analysis)}")
                
                # Safely get model name
                model_name = getattr(agent_analysis, 'model_name', 
                                    getattr(agent_analysis, 'model', 
                                            getattr(agent_analysis, 'agent_name', 'unknown')))
                
                # Safely get reasoning - try different possible attribute names
                reasoning = getattr(agent_analysis, 'reasoning', 
                                   getattr(agent_analysis, 'rationale',
                                           getattr(agent_analysis, 'explanation',
                                                   getattr(agent_analysis, 'analysis',
                                                           getattr(agent_analysis, 'summary', 'No reasoning available')))))
                
                if reasoning and isinstance(reasoning, str):
                    reasoning_summary = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                else:
                    reasoning_summary = "No reasoning available"
                
                agent_detail = {
                    'agent_name': agent_analysis.agent_name,
                    'model': model_name,
                    'verdict': agent_analysis.verdict.value if agent_analysis.verdict else None,
                    'verdict_bool': agent_analysis.verdict == Verdict.CORRECT if agent_analysis.verdict else None,
                    'confidence': agent_analysis.confidence if hasattr(agent_analysis, 'confidence') and agent_analysis.confidence is not None else 0.0,
                    'reasoning_summary': reasoning_summary,
                    'cost': getattr(agent_analysis, 'cost', 0.0),
                    'tokens': getattr(agent_analysis, 'tokens', 0),
                    'processing_time_ms': getattr(agent_analysis, 'processing_time_ms', 0)
                }
                
                # Add to round data
                round_data['agents'].append(agent_detail)
                
                # Add to overall agent details with round info
                agent_detail_with_round = agent_detail.copy()
                agent_detail_with_round['round'] = round_idx + 1
                all_agent_details.append(agent_detail_with_round)
                
                # Update round summary
                if agent_analysis.verdict:
                    if agent_analysis.verdict == Verdict.CORRECT:
                        round_data['round_summary']['verdicts']['correct'] += 1
                    elif agent_analysis.verdict == Verdict.INCORRECT:
                        round_data['round_summary']['verdicts']['incorrect'] += 1
                    else:
                        round_data['round_summary']['verdicts']['uncertain'] += 1
                
                if hasattr(agent_analysis, 'confidence') and agent_analysis.confidence is not None:
                    confidences.append(agent_analysis.confidence)
                
                if model_name not in round_data['round_summary']['models_used']:
                    round_data['round_summary']['models_used'].append(model_name)
            
            # Calculate average confidence for this round
            if confidences:
                round_data['round_summary']['avg_confidence'] = sum(confidences) / len(confidences)
            
            round_details.append(round_data)
        
        return {
            'round_details': round_details,
            'all_agent_details': all_agent_details
        }
    
    async def evaluate_single_example(self, example: Dict[str, Any], retry_delay: float = 2.0) -> Dict[str, Any]:
        """Evaluate a single HotPotQA example with detailed output."""
        # Extract data from your format
        question = example["question"]
        model_answer = example["model_answer"]  # This is the answer to be judged
        ground_truth_answer = example["answer"]  # This is the gold standard answer
        ground_truth_verdict = example["gold_verdict"]  # True/False for correctness
        question_type = example.get("type", "unknown")
        question_level = example.get("level", "unknown")
        
        # Construct context
        context = self.construct_context_from_sentences(example)
        
        # Skip if no model answer
        if not model_answer or not model_answer.strip():
            logger.warning(f"Skipping example {example['id']} - no model answer provided")
            return None
        
        # Create evaluation request
        request = EvaluationRequest(
            question=question,
            answer=model_answer,
            context=context,
            metadata={
                'id': example['id'],
                'type': question_type,
                'level': question_level,
                'ground_truth_answer': ground_truth_answer,
                'ground_truth_verdict': ground_truth_verdict
            }
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Run evaluation
                result = await self.round_manager.evaluate(request)
                
                if result.success and result.judgment:
                    # Save evaluation with ground truth
                    eval_id = self.collector.save_evaluation_with_ground_truth(
                        result=result,
                        ground_truth=ground_truth_verdict,
                        example_id=example['id']
                    )
                    
                    judgment = result.judgment
                    
                    # Get pipeline verdict
                    pipeline_says_correct = judgment.is_correct == Verdict.CORRECT
                    
                    # Extract detailed round and agent analysis
                    detailed_analysis = self.extract_detailed_round_analysis(judgment)
                    
                    # Collect jury verdicts for analysis
                    jury_verdicts = []
                    for agent_detail in detailed_analysis['all_agent_details']:
                        if agent_detail['verdict_bool'] is not None:
                            jury_verdicts.append({
                                'round': agent_detail['round'],
                                'agent': agent_detail['agent_name'],
                                'model': agent_detail['model'],
                                'verdict': agent_detail['verdict'],
                                'verdict_bool': agent_detail['verdict_bool'],
                                'confidence': agent_detail['confidence']
                            })
                    
                    result_data = {
                        # Basic info
                        'id': example['id'],
                        'question': question,
                        'model_answer': model_answer,
                        'ground_truth_answer': ground_truth_answer,
                        'question_type': question_type,
                        'question_level': question_level,
                        
                        # Ground truth and pipeline results
                        'ground_truth_verdict': ground_truth_verdict,
                        'pipeline_verdict': pipeline_says_correct,
                        'verdict_correct': ground_truth_verdict == pipeline_says_correct,
                        
                        # Confidence and consensus
                        'total_confidence': judgment.raw_confidence,
                        'calibrated_confidence': judgment.calibrated_confidence if judgment.calibrated_confidence else judgment.raw_confidence,
                        'consensus_level': judgment.jury_decision.consensus_level,
                        'vote_distribution': judgment.jury_decision.vote_distribution,
                        'weighted_confidence': judgment.jury_decision.weighted_confidence,
                        
                        # Round-by-round analysis
                        'round_details': detailed_analysis['round_details'],
                        'agent_details': detailed_analysis['all_agent_details'],
                        'jury_verdicts': jury_verdicts,
                        
                        # Performance metrics
                        'processing_time_ms': judgment.processing_time_ms,
                        'total_cost': judgment.total_cost,
                        'total_tokens': judgment.total_tokens,
                        'models_used': judgment.models_used,
                        'num_rounds': len(judgment.round_summaries),
                        'num_agents': len(detailed_analysis['all_agent_details']),
                        
                        # Evaluation ID for tracking
                        'eval_id': eval_id
                    }
                    
                    return result_data
                else:
                    logger.error(f"Evaluation failed for example {example['id']}: {result.error}")
                    return None
                    
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limit error, wait and retry
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Error evaluating example {example['id']}: {e}")
                    if attempt == max_retries - 1:
                        import traceback
                        traceback.print_exc()
                    return None
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all required metrics."""
        # Extract data
        verdict_corrects = [r['verdict_correct'] for r in results]
        ground_truth_verdicts = [r['ground_truth_verdict'] for r in results]
        pipeline_verdicts = [r['pipeline_verdict'] for r in results]
        confidences = [r['total_confidence'] for r in results]
        
        # Convert to numpy arrays
        verdict_corrects = np.array(verdict_corrects)
        ground_truth_verdicts = np.array(ground_truth_verdicts, dtype=int)
        pipeline_verdicts = np.array(pipeline_verdicts, dtype=int)
        confidences = np.array(confidences)
        
        # Basic metrics
        n_total = len(results)
        verdict_accuracy = float(np.mean(verdict_corrects))
        
        # Cohen's kappa
        cohen_kappa = cohen_kappa_score(ground_truth_verdicts, pipeline_verdicts)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(confidences, verdict_corrects)
        
        # Kendall's tau
        kendall_tau, kendall_p = kendalltau(confidences, verdict_corrects)
        
        # Brier score
        brier = brier_score_loss(ground_truth_verdicts, confidences)
        
        # ECE
        ece = self.calculate_ece(confidences, verdict_corrects)
        
        # Calculate inter-rater agreement metrics
        inter_rater_metrics = self.calculate_inter_rater_metrics(results)
        
        # Breakdown by question type and level
        type_breakdown = {}
        level_breakdown = {}
        
        for result in results:
            q_type = result['question_type']
            q_level = result['question_level']
            
            if q_type not in type_breakdown:
                type_breakdown[q_type] = {'count': 0, 'accuracy': []}
            type_breakdown[q_type]['count'] += 1
            type_breakdown[q_type]['accuracy'].append(result['verdict_correct'])
            
            if q_level not in level_breakdown:
                level_breakdown[q_level] = {'count': 0, 'accuracy': []}
            level_breakdown[q_level]['count'] += 1
            level_breakdown[q_level]['accuracy'].append(result['verdict_correct'])
        
        # Calculate averages
        for q_type in type_breakdown:
            type_breakdown[q_type]['accuracy'] = float(np.mean(type_breakdown[q_type]['accuracy']))
        
        for q_level in level_breakdown:
            level_breakdown[q_level]['accuracy'] = float(np.mean(level_breakdown[q_level]['accuracy']))
        
        metrics = {
            "N_total": n_total,
            "Verdict_Accuracy": verdict_accuracy,
            "Cohen_kappa": float(cohen_kappa),
            "Spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0,
            "Spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
            "Kendall_tau_b": float(kendall_tau) if not np.isnan(kendall_tau) else 0.0,
            "Kendall_p": float(kendall_p) if not np.isnan(kendall_p) else 1.0,
            "Brier": float(brier),
            "ECE@10": float(ece),
            "Inter_rater_agreement": inter_rater_metrics,
            "Type_breakdown": type_breakdown,
            "Level_breakdown": level_breakdown
        }
        
        return metrics
    
    def calculate_inter_rater_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate inter-rater agreement metrics."""
        try:
            # Collect all agent verdicts for each example
            all_verdicts = []
            
            for result in results:
                example_verdicts = []
                for agent_detail in result['agent_details']:
                    if agent_detail['verdict_bool'] is not None:
                        example_verdicts.append(1 if agent_detail['verdict_bool'] else 0)
                
                if len(example_verdicts) >= 2:  # Need at least 2 raters
                    all_verdicts.append(example_verdicts)
            
            if len(all_verdicts) < 2:
                return {"error": "Insufficient data for inter-rater agreement"}
            
            # Calculate Fleiss' kappa if we have consistent number of raters
            rater_counts = [len(v) for v in all_verdicts]
            if len(set(rater_counts)) == 1:  # All examples have same number of raters
                # Prepare data for Fleiss' kappa
                n_raters = rater_counts[0]
                fleiss_data = []
                
                for verdicts in all_verdicts:
                    # Count votes for each category (0=incorrect, 1=correct)
                    count_incorrect = verdicts.count(0)
                    count_correct = verdicts.count(1)
                    fleiss_data.append([count_incorrect, count_correct])
                
                fleiss_k = fleiss_kappa(np.array(fleiss_data))
                
                return {
                    "fleiss_kappa": float(fleiss_k),
                    "n_examples": len(all_verdicts),
                    "n_raters": n_raters
                }
            else:
                return {
                    "error": "Inconsistent number of raters across examples",
                    "rater_counts": rater_counts
                }
                
        except Exception as e:
            logger.error(f"Error calculating inter-rater metrics: {e}")
            return {"error": str(e)}
    
    def calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    async def run_evaluation(self, max_examples: int = None, delay_between_examples: float = 2.0):
        """Run evaluation on HotPotQA dataset, one question at a time."""
        # Load data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Limit examples if specified
        if max_examples:
            data = data[:max_examples]
        
        logger.info(f"Loaded {len(data)} examples from {self.data_path}")
        
        # Initialize pipeline
        await self.initialize_pipeline()
        
        # Evaluate one by one
        results = []
        for idx, example in enumerate(data):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating example {idx + 1}/{len(data)} - ID: {example['id']}")
            logger.info(f"Question: {example['question'][:100]}...")
            logger.info(f"Type: {example.get('type', 'unknown')}, Level: {example.get('level', 'unknown')}")
            logger.info(f"Gold verdict: {example['gold_verdict']}")
            
            # Evaluate single example
            result = await self.evaluate_single_example(example)
            
            if result:
                results.append(result)
                logger.info(f"✓ Evaluation complete")
                logger.info(f"  Pipeline verdict: {result['pipeline_verdict']} (Ground truth: {result['ground_truth_verdict']})")
                logger.info(f"  Verdict correct: {result['verdict_correct']}")
                logger.info(f"  Confidence: {result['total_confidence']:.2%}")
                logger.info(f"  Consensus: {result['consensus_level']:.2%}")
                logger.info(f"  Agents: {result['num_agents']} across {result['num_rounds']} rounds")
                
                # Show round-by-round details
                logger.info("  Round-by-round analysis:")
                for round_detail in result['round_details']:
                    round_num = round_detail['round_number']
                    round_summary = round_detail['round_summary']
                    logger.info(f"    Round {round_num}: {round_summary['total_agents']} agents, avg conf: {round_summary['avg_confidence']:.2%}")
                    logger.info(f"      Verdicts: {round_summary['verdicts']}")
                    
                    # Show individual agent verdicts for this round
                    for agent in round_detail['agents']:
                        conf_str = f"{agent['confidence']:.2%}" if agent['confidence'] else "N/A"
                        logger.info(f"        {agent['agent_name']} ({agent['model']}): {agent['verdict']} (conf: {conf_str})")
                        
            else:
                logger.warning(f"✗ Failed to evaluate example {example['id']}")
            
            # Add delay between examples to avoid rate limits
            if idx < len(data) - 1:  # Don't delay after the last example
                logger.info(f"Waiting {delay_between_examples}s before next example...")
                await asyncio.sleep(delay_between_examples)
        
        # Calculate metrics
        if results:
            metrics = self.calculate_metrics(results)
            
            # Save results
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_path = output_dir / f"hotpotqa_detailed_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': timestamp,
                        'data_path': str(self.data_path),
                        'n_examples': len(data),
                        'n_evaluated': len(results)
                    },
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            
            # Save metrics summary
            metrics_path = output_dir / f"hotpotqa_metrics_{timestamp}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            # Print metrics
            print("\n" + "="*80)
            print("EVALUATION METRICS")
            print("="*80)
            for metric_name, value in metrics.items():
                if metric_name in ['Type_breakdown', 'Level_breakdown', 'Inter_rater_agreement']:
                    print(f"{metric_name}:")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            print(f"  {sub_key}: {sub_value}")
                elif isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")
            print("="*80)
            
            logger.info(f"Results saved to {results_path}")
            logger.info(f"Metrics saved to {metrics_path}")
            logger.info(f"Calibration data saved to {self.collector.data_dir}")
        else:
            logger.error("No valid results to calculate metrics")
        
        # Cleanup
        await self.round_manager.agent_manager.cleanup()
        await self.provider.close()

async def main():
    """Main entry point."""
    # Configuration - Update to your dataset path
    data_path = "/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/dataset/hotpotqa_cache/hotpotqa_with_verdict_with_gold copy.json"
    
    # Create evaluator
    evaluator = HotPotQAEvaluator(data_path)
    
    # Run evaluation one by one with 2-second delay between examples
    # Start with 2 examples for testing, then increase
    await evaluator.run_evaluation(max_examples=8000, delay_between_examples=2.0)

if __name__ == "__main__":
    asyncio.run(main())
