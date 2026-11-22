"""End-to-end pipeline: generate -> parse -> chunk -> index -> tune fusion -> extract -> audit -> evaluate.

Examples
--------
    python train.py                     # quick mode (24 contracts, offline extractor)
    python train.py --full              # 80 contracts
    python train.py --llm openai        # also run the LLM extractor on the test split
    python train.py --embedder st       # sentence-transformers dense retriever

All outputs go to ``artifacts/`` and to a local MLflow store (``mlflow.db`` + ``mlruns/``).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.auditor import audit_portfolio, findings_frame, summarize_findings
from src.evaluation import (
    breach_metrics,
    evaluate_retrieval,
    extraction_metrics,
    gt_to_extraction,
    guardrail_stress_test,
    loan_terms_metrics,
    recall_by_phrasing,
    tune_fusion,
)
from src.extractor import OfflineExtractor, build_extractor
from src.pipeline import PipelineConfig, build_indices, extract_all, fit_embedder, process_pdfs
from src.providers import available_providers, load_dotenv
from src.schemas import ExtractionResult, FindingStatus
from src.synthetic import SyntheticConfig, generate_corpus
from src.tracking import MLflowTracker

REPO = Path(__file__).resolve().parent
LOG = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="quick mode (default): 24 contracts")
    mode.add_argument("--full", action="store_true", help="full mode: 80 contracts")
    p.add_argument("--n-contracts", type=int, default=None, help="override the number of contracts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-frac", type=float, default=0.4, help="share of contracts used to fit LSA / tune fusion")
    p.add_argument("--embedder", choices=["lsa", "st"], default="lsa")
    p.add_argument("--lsa-components", type=int, default=128)
    p.add_argument("--llm", choices=["none", "openai", "deepseek", "groq"], default="none",
                   help="additionally run an LLM extractor on the test split (needs the API key)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--chunk-tokens", type=int, default=160)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--semantic-breaks", action="store_true", help="add embedding-similarity chunk breakpoints")
    p.add_argument("--holdout-frac", type=float, default=0.2, help="share of prose covenants with unseen phrasing")
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "synthetic")
    p.add_argument("--out", type=Path, default=REPO / "artifacts")
    p.add_argument("--experiment", default="financial-rag-auditor")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpx2", "openai", "sentence_transformers", "mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def split_ids(ids: list[str], dev_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    n_dev = int(np.clip(round(dev_frac * len(ids)), 2, len(ids) - 2))
    return sorted(order[:n_dev]), sorted(order[n_dev:])


def covenant_rows(results: list[ExtractionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for vc in r.covenants:
            c = vc.covenant
            rows.append({
                "contract_id": r.contract_id, "extractor": r.extractor, "covenant_type": c.covenant_type.value,
                "operator": c.operator.value, "threshold": c.threshold, "unit": c.unit.value,
                "test_frequency": c.test_frequency.value, "page": vc.citation_page,
                "guardrail": vc.check.status.value, "quote_match": vc.check.quote_match,
                "match_score": vc.check.match_score, "reasons": "; ".join(vc.check.reasons),
                "evidence_quote": c.evidence_quote,
            })
    return pd.DataFrame(rows)


def save_plots(curve: pd.DataFrame, findings: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-4 of the reference palette, fixed per configuration
    colors = {(False, "rrf"): "#2a78d6", (False, "weighted"): "#eb6834", (True, "rrf"): "#1baf7a", (True, "weighted"): "#eda100"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (expand, method), g in curve.groupby(["expand", "method"]):
        label = f"{method}{' + query expansion' if expand else ''}"
        ax.plot(g["alpha"], g["mrr"], marker="o" if expand else "s", markersize=6, linewidth=2,
                linestyle="-" if expand else "--", color=colors.get((bool(expand), method)), label=label)
    ax.set_xlabel("alpha (weight of the dense retriever; 0 = BM25 only, 1 = dense only)")
    ax.set_ylabel("dev MRR")
    ax.set_title("Hybrid fusion tuning on the dev split")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fusion_tuning.png", dpi=120)
    plt.close(fig)

    if not findings.empty:
        breaches = findings.assign(breach=(findings["status"] == "BREACH").astype(int))
        pivot = breaches.pivot_table(index="contract_id", columns="period", values="breach", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(pivot))))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Reds")
        ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, fontsize=7)
        ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=7)
        ax.set_title("Covenant breaches per contract and test period")
        fig.colorbar(im, ax=ax, label="# breaches")
        fig.tight_layout()
        fig.savefig(out / "breach_heatmap.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    setup_logging()
    load_dotenv(REPO / ".env")
    t_start = time.perf_counter()
    n_contracts = args.n_contracts or (80 if args.full else 24)
    mode = "full" if args.full else "quick"
    args.out.mkdir(parents=True, exist_ok=True)

    syn_cfg = SyntheticConfig(n_contracts=n_contracts, seed=args.seed, holdout_phrasing_frac=args.holdout_frac)
    cfg = PipelineConfig(embedder=args.embedder, lsa_components=args.lsa_components, seed=args.seed)
    cfg.chunker.max_tokens = args.chunk_tokens
    cfg.chunker.overlap_tokens = args.overlap
    cfg.chunker.semantic_breaks = args.semantic_breaks
    cfg.extractor.top_k = args.top_k
    cfg.retriever.top_k = args.top_k

    tracker = MLflowTracker(REPO, args.experiment, run_name=f"{mode}-{args.embedder}-{args.llm}", enabled=not args.no_mlflow)
    metrics: dict = {"mode": mode}
    with tracker:
        tracker.log_params({"mode": mode, "llm": args.llm, "synthetic": asdict(syn_cfg), "pipeline": asdict(cfg),
                            "dev_frac": args.dev_frac})

        # 1) corpus -------------------------------------------------------------------
        LOG.info("[1/8] generating %d synthetic credit agreements + Master Ledger", n_contracts)
        corpus = generate_corpus(syn_cfg, out_dir=args.data_dir)
        gt = corpus.ground_truth
        ids = sorted(gt)

        # 2) parse + chunk ------------------------------------------------------------
        LOG.info("[2/8] parsing PDFs (PyMuPDF) and chunking")
        t0 = time.perf_counter()
        processed = process_pdfs(corpus.pdfs, cfg)
        chunks = [c for p in processed.values() for c in p.chunks]
        parse_stats = {
            "n_contracts": len(ids),
            "n_pages": sum(len(p.parsed.pages) for p in processed.values()),
            "n_tables_detected": sum(p.parsed.n_tables for p in processed.values()),
            "n_tables_expected": sum(1 + int(g["has_ratchet"]) + int(any(c["presentation"] == "table" for c in g["covenants"])) for g in gt.values()),
            "n_header_footer_lines_stripped": sum(p.parsed.n_stripped for p in processed.values()),
            "n_chunks": len(chunks),
            "avg_chunk_tokens": float(np.mean([c.n_tokens for c in chunks])),
            "max_chunk_tokens": float(max(c.n_tokens for c in chunks)),
            "n_table_chunks": sum(c.has_table for c in chunks),
            "parse_seconds": round(time.perf_counter() - t0, 2),
        }
        metrics["parsing"] = parse_stats
        LOG.info("parsing stats: %s", parse_stats)

        # 3) split + index ------------------------------------------------------------
        dev_ids, test_ids = split_ids(ids, args.dev_frac, args.seed)
        LOG.info("[3/8] dev=%d contracts (fit embedder, tune fusion), test=%d contracts", len(dev_ids), len(test_ids))
        embedder = fit_embedder([processed[i] for i in dev_ids], cfg)
        build_indices(processed, embedder, cfg)

        # 4) tune hybrid fusion on dev --------------------------------------------------
        LOG.info("[4/8] grid search on the dev split: query expansion x fusion method x alpha")
        alphas = [round(a, 1) for a in np.linspace(0, 1, 11)]
        best, curve = tune_fusion(processed, gt, dev_ids, alphas)
        curve_df = pd.DataFrame(curve)
        curve_df.to_csv(args.out / "fusion_tuning.csv", index=False)
        for row in curve:
            tag = f"{row['method']}{'_expanded' if row['expand'] else ''}"
            tracker.log_metric(f"dev_mrr_{tag}", row["mrr"], step=int(round(row["alpha"] * 10)))
            tracker.log_metric(f"dev_recall@1_{tag}", row["recall@1"], step=int(round(row["alpha"] * 10)))
        # indices hold a reference to cfg.retriever, so this switches them to the tuned setting
        cfg.retriever.method, cfg.retriever.alpha = best["method"], best["alpha"]
        cfg.retriever.expand_queries = best["expand"]
        LOG.info("best on dev: expand=%s method=%s alpha=%.1f MRR=%.3f",
                 best["expand"], best["method"], best["alpha"], best["mrr"])
        metrics["fusion_best"] = {k: best[k] for k in ("expand", "method", "alpha", "mrr", "recall@1", "recall@3")}

        # 5) retrieval comparison on test ------------------------------------------------
        LOG.info("[5/8] retrieval evaluation on the test split")
        comparison = []
        tuned_name = f"Hybrid tuned ({best['method']}, alpha={best['alpha']}{', expanded' if best['expand'] else ''})"
        for name, method, alpha, expand in [
            ("BM25 only", "bm25", 0.0, False),
            (f"Dense only ({embedder.name})", "dense", 1.0, False),
            ("Hybrid RRF (k=60, equal weights)", "rrf", 0.5, False),
            ("BM25 + query expansion", "bm25", 0.0, True),
            (f"Dense ({embedder.name}) + query expansion", "dense", 1.0, True),
            ("Hybrid RRF + query expansion", "rrf", 0.5, True),
            (tuned_name, best["method"], best["alpha"], best["expand"]),
        ]:
            m, rows = evaluate_retrieval(processed, gt, test_ids, method, alpha, expand=expand)
            comparison.append({"retriever": name, "expand": expand, **m})
            if name == tuned_name:
                pd.DataFrame(rows).to_csv(args.out / "retrieval_per_query.csv", index=False)
        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv(args.out / "retrieval_comparison.csv", index=False)
        metrics["retrieval_test"] = {r["retriever"]: {k: v for k, v in r.items() if k != "retriever"} for r in comparison}
        LOG.info("retrieval (test):\n%s", comp_df.round(3).to_string(index=False))

        # 6) extraction --------------------------------------------------------------
        LOG.info("[6/8] extraction (offline rules + guardrails)%s", f" and LLM ({args.llm})" if args.llm != "none" else "")
        offline = OfflineExtractor(cfg.extractor, cfg.guardrails)
        results = extract_all(processed, offline)
        test_results = [r for r in results if r.contract_id in test_ids]
        ex_test = extraction_metrics(test_results, gt)
        ex_test.update(loan_terms_metrics(test_results, gt))
        ex_test.update(recall_by_phrasing(test_results, gt))
        metrics["extraction_test"] = ex_test
        metrics["extraction_all"] = extraction_metrics(results, gt) | recall_by_phrasing(results, gt)
        with open(args.out / "extraction_results.json", "w", encoding="utf-8") as fh:
            json.dump([r.model_dump(mode="json") for r in results], fh, indent=1)
        covenant_rows(results).to_csv(args.out / "covenants_extracted.csv", index=False)

        llm_results: list[ExtractionResult] = []
        if args.llm != "none":
            if args.llm not in available_providers():
                LOG.warning("--llm %s requested but its API key is not set; skipping the LLM run", args.llm)
                metrics["llm"] = {"status": "skipped: missing API key"}
            else:
                llm = build_extractor(args.llm, cfg.extractor, cfg.guardrails)
                llm_results = extract_all({i: processed[i] for i in test_ids}, llm)
                llm_m = extraction_metrics(llm_results, gt) | loan_terms_metrics(llm_results, gt) | recall_by_phrasing(llm_results, gt)
                llm_m["mean_attempts"] = float(np.mean([r.attempts for r in llm_results]))
                llm_m["mean_latency_s"] = float(np.mean([r.latency_s for r in llm_results]))
                metrics["llm"] = {"extractor": llm.name, **llm_m}
                with open(args.out / "extraction_results_llm.json", "w", encoding="utf-8") as fh:
                    json.dump([r.model_dump(mode="json") for r in llm_results], fh, indent=1)
                covenant_rows(llm_results).to_csv(args.out / "covenants_extracted_llm.csv", index=False)

        # 7) guardrail stress test ---------------------------------------------------------
        LOG.info("[7/8] guardrail stress test (synthetic hallucinations)")
        stress_m, stress_rows = guardrail_stress_test(processed, gt, test_ids, seed=args.seed, cfg=cfg.guardrails)
        pd.DataFrame(stress_rows).to_csv(args.out / "guardrail_stress_test.csv", index=False)
        metrics["guardrail_stress_test"] = stress_m

        # 8) audit / reconciliation -----------------------------------------------------------
        LOG.info("[8/8] reconciliation against the Master Ledger")
        findings = audit_portfolio(results, corpus.ledger, cfg.audit)
        gt_findings = audit_portfolio([gt_to_extraction(gt[i]) for i in ids], corpus.ledger, cfg.audit)
        report = findings_frame(findings)
        report.to_csv(args.out / "audit_report.csv", index=False)
        with open(args.out / "audit_report.json", "w", encoding="utf-8") as fh:
            json.dump([f.to_row() for f in findings], fh, indent=1)
        findings_frame(gt_findings).to_csv(args.out / "audit_report_ground_truth.csv", index=False)
        test_set = set(test_ids)
        metrics["audit_test"] = breach_metrics(
            [f for f in findings if f.contract_id in test_set], [f for f in gt_findings if f.contract_id in test_set]
        )
        metrics["audit_all"] = breach_metrics(findings, gt_findings)
        metrics["audit_summary"] = summarize_findings(findings)
        breach_keys = {f.key() for f in gt_findings if f.status == FindingStatus.BREACH}
        injected = [(g["contract_id"], e["covenant_type"], e["period"]) for g in gt.values() for e in g["injected_breaches"]]
        metrics["audit_summary"]["injected_breaches"] = len(injected)
        metrics["audit_summary"]["injected_breaches_detected_with_gt_covenants"] = sum(k in breach_keys for k in injected)
        if llm_results:
            llm_findings = audit_portfolio(llm_results, corpus.ledger, cfg.audit)
            findings_frame(llm_findings).to_csv(args.out / "audit_report_llm.csv", index=False)
            metrics["llm"]["audit"] = breach_metrics(llm_findings, [f for f in gt_findings if f.contract_id in test_set])

        metrics["split"] = {"dev": dev_ids, "test": test_ids}
        metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
        with open(args.out / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        with open(args.out / "config.json", "w", encoding="utf-8") as fh:
            json.dump({"synthetic": asdict(syn_cfg), "pipeline": asdict(cfg), "args": {k: str(v) for k, v in vars(args).items()}}, fh, indent=2)
        try:
            save_plots(curve_df, report, args.out)
        except Exception as exc:  # plotting must never break the pipeline
            LOG.warning("plotting failed: %s", exc)

        tracker.log_metrics({k: v for k, v in metrics.items() if k not in ("split", "mode", "fusion_best")})
        tracker.log_metrics({"fusion_best": {k: float(v) for k, v in metrics["fusion_best"].items() if k != "method"}})
        tracker.set_tags({"fusion_method": str(best["method"]), "embedder": embedder.name, "extractor": offline.name})
        tracker.log_artifacts(args.out)

    _print_summary(metrics)
    return metrics


def _print_summary(m: dict) -> None:
    ex, au, st = m["extraction_test"], m["audit_test"], m["guardrail_stress_test"]
    tuned = [v for k, v in m["retrieval_test"].items() if k.startswith("Hybrid tuned")][0]
    bm25 = m["retrieval_test"]["BM25 only"]
    lines = [
        "",
        "=" * 72,
        f"Financial RAG Auditor - {m['mode']} run finished in {m['runtime_seconds']} s",
        f"Parsing: {m['parsing']['n_pages']} pages, {m['parsing']['n_tables_detected']}/{m['parsing']['n_tables_expected']} tables, "
        f"{m['parsing']['n_chunks']} chunks",
        f"Fusion (dev): {m['fusion_best']['method']} alpha={m['fusion_best']['alpha']} expand={m['fusion_best']['expand']}",
        f"Retrieval (test) tuned hybrid: R@1={tuned['recall@1']:.3f} R@3={tuned['recall@3']:.3f} MRR={tuned['mrr']:.3f}"
        f" | BM25: R@1={bm25['recall@1']:.3f} MRR={bm25['mrr']:.3f}",
        f"Extraction (test): field P/R/F1={ex['field_precision']:.3f}/{ex['field_recall']:.3f}/{ex['field_f1']:.3f} "
        f"numeric EM={ex['numeric_exact_match']:.3f} hallucination={ex['hallucination_rate']:.3f} loan terms acc={ex['loan_terms_acc']:.3f}",
        f"Guardrail stress test: hallucination detection={st['stress_hallucination_detection']:.3f} "
        f"false rejection={st['stress_false_rejection_rate']:.3f}",
        f"Breach detection (test): P={au['breach_precision']:.3f} R={au['breach_recall']:.3f} F1={au['breach_f1']:.3f} "
        f"({int(au['breach_tp'])}/{int(au['n_gt_breaches'])} breaches found)",
        "=" * 72,
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
