"""Streamlit dashboard for the Financial AI Agent & RAG Auditing Pipeline.

    streamlit run app.py

Works before ``train.py`` was ever run: a small demo corpus (4 contracts + ledger) is
generated in memory. When ``data/synthetic`` and ``artifacts/`` exist (after training),
the dashboard uses the trained corpus, the tuned retriever settings and the saved metrics.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymupdf
import streamlit as st

from src.auditor import audit_contract, audit_portfolio, compute_actuals, findings_frame, summarize_findings
from src.evaluation import (
    breach_metrics,
    evaluate_retrieval,
    extraction_metrics,
    gt_to_extraction,
    recall_by_phrasing,
    relevant_chunk_ids,
)
from src.extractor import build_extractor
from src.pipeline import PipelineConfig, ProcessedContract, fit_embedder, process_pdf
from src.providers import available_providers, load_dotenv
from src.retriever import Embedder, HybridIndex, LOAN_TERM_QUERIES, covenant_query
from src.schemas import COVENANT_LABELS, CovenantType, ExtractionResult
from src.synthetic import Corpus, SyntheticConfig, generate_corpus, load_corpus

REPO = Path(__file__).resolve().parent
# Overridable so tests can exercise the cold-start path (no corpus, no artifacts).
DATA_DIR = Path(os.environ.get("FRA_DATA_DIR", REPO / "data" / "synthetic"))
ART_DIR = Path(os.environ.get("FRA_ARTIFACTS_DIR", REPO / "artifacts"))

# Reference palette (dataviz skill): categorical slots 1-3 and the fixed status palette.
SERIES = {"BM25": "#2a78d6", "Dense": "#eb6834", "Fused": "#1baf7a"}
STATUS_COLORS = {"COMPLIANT": "#0ca30c", "INSUFFICIENT_DATA": "#fab219", "BREACH": "#d03b3b"}
STATUS_SHORT = {"COMPLIANT": "OK", "INSUFFICIENT_DATA": "N/A", "BREACH": "BREACH"}

st.set_page_config(page_title="Financial RAG Auditor", layout="wide")
load_dotenv(REPO / ".env")


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------
@dataclass
class Workspace:
    corpus: Corpus
    source: str
    cfg: PipelineConfig
    embedder: Embedder
    metrics: dict | None


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner="Preparing corpus and retriever...")
def get_workspace() -> Workspace:
    metrics = _load_json(ART_DIR / "metrics.json")
    if (DATA_DIR / "manifest.json").is_file():
        corpus, source = load_corpus(DATA_DIR), "trained corpus (data/synthetic)"
    else:
        corpus, source = generate_corpus(SyntheticConfig(n_contracts=4, seed=7)), "in-memory demo (run train.py for the full corpus)"
    cfg = PipelineConfig()
    if metrics and "fusion_best" in metrics:
        best = metrics["fusion_best"]
        cfg.retriever.method = best.get("method", cfg.retriever.method)
        cfg.retriever.alpha = float(best.get("alpha", cfg.retriever.alpha))
        cfg.retriever.expand_queries = bool(best.get("expand", cfg.retriever.expand_queries))
    ids = [c.contract_id for c in corpus.contracts]
    fit_ids = [i for i in (metrics or {}).get("split", {}).get("dev", []) if i in ids] or ids[:6]
    pdfs = corpus.pdfs
    embedder = fit_embedder([process_pdf(i, pdfs[i], cfg) for i in fit_ids], cfg)
    return Workspace(corpus, source, cfg, embedder, metrics)


@st.cache_resource(show_spinner="Parsing PDF and building the hybrid index...")
def get_processed(doc_id: str, digest: str, _pdf: bytes) -> ProcessedContract:
    ws = get_workspace()
    p = process_pdf(doc_id, _pdf, ws.cfg)
    if p.chunks:
        p.index = HybridIndex(p.chunks, ws.embedder, ws.cfg.retriever)
    return p


def processed_for(doc_id: str, pdf: bytes) -> ProcessedContract:
    return get_processed(doc_id, hashlib.sha1(pdf).hexdigest(), pdf)


@st.cache_resource(show_spinner="Running offline extraction + audit for the portfolio...")
def get_portfolio() -> tuple[pd.DataFrame, list[ExtractionResult], pd.DataFrame]:
    """Findings for every contract: from artifacts/ if present, else computed in memory."""
    ws = get_workspace()
    report_path = ART_DIR / "audit_report.csv"
    ids = [c.contract_id for c in ws.corpus.contracts]
    if report_path.is_file() and ws.source.startswith("trained"):
        report = pd.read_csv(report_path)
        if set(report["contract_id"]).issubset(set(ids)):
            gt_path = ART_DIR / "audit_report_ground_truth.csv"
            gt_report = pd.read_csv(gt_path) if gt_path.is_file() else pd.DataFrame()
            return report, [], gt_report
    extractor = build_extractor("offline", ws.cfg.extractor, ws.cfg.guardrails)
    results = []
    for c in ws.corpus.contracts:
        p = processed_for(c.contract_id, c.pdf_bytes)
        results.append(extractor.extract(c.contract_id, p.index, p.page_texts))
    findings = audit_portfolio(results, ws.corpus.ledger, ws.cfg.audit)
    gt_findings = audit_portfolio([gt_to_extraction(c.ground_truth) for c in ws.corpus.contracts], ws.corpus.ledger, ws.cfg.audit)
    return findings_frame(findings), results, findings_frame(gt_findings)


@st.cache_data(show_spinner=False)
def page_png(digest: str, _pdf: bytes, page_no: int, dpi: int = 80) -> bytes:
    doc = pymupdf.open(stream=_pdf, filetype="pdf")
    try:
        return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def highlight(text: str, quote: str) -> str:
    """HTML-escape ``text`` and wrap the span matching ``quote`` (pipes/whitespace-insensitive) in <mark>."""
    tokens = [t for t in re.split(r"[\s|]+", quote.strip()) if t]
    if not tokens:
        return html.escape(text)
    pattern = r"[\s|]+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return html.escape(text)
    return (
        html.escape(text[: m.start()])
        + "<mark style='background:#fab21955;padding:0 2px'>"
        + html.escape(text[m.start() : m.end()])
        + "</mark>"
        + html.escape(text[m.end() :])
    )


def box(content_html: str) -> None:
    st.markdown(
        "<div style='white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;font-size:0.82rem;"
        "border:1px solid rgba(128,128,128,0.3);border-radius:6px;padding:8px'>" + content_html + "</div>",
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    return {"verified": "VERIFIED", "rejected": "REJECTED"}.get(status, status)


def style_status(df: pd.DataFrame, col: str = "status") -> pd.io.formats.style.Styler:
    def color(v: str) -> str:
        c = STATUS_COLORS.get(v)
        return f"color:{c};font-weight:600" if c else ""

    return df.style.map(color, subset=[col])


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
ws = get_workspace()
st.sidebar.title("Financial RAG Auditor")
st.sidebar.caption(f"Corpus: {ws.source}")
source = st.sidebar.radio("Document source", ["Generated contract", "Upload a PDF"], horizontal=False)
contract_ids = [c.contract_id for c in ws.corpus.contracts]
gt: dict | None = None
uploaded = None
if source == "Generated contract":
    cid = st.sidebar.selectbox("Contract", contract_ids, format_func=lambda i: f"{i} - {ws.corpus.ground_truth[i]['borrower']}")
    pdf_bytes = ws.corpus.pdfs[cid]
    gt = ws.corpus.ground_truth[cid]
    doc_id = cid
else:
    uploaded = st.sidebar.file_uploader("Credit agreement (PDF)", type=["pdf"])
    if uploaded is None:
        st.sidebar.info("Upload a PDF, or switch back to a generated contract.")
        cid = contract_ids[0]
        pdf_bytes = ws.corpus.pdfs[cid]
        gt = ws.corpus.ground_truth[cid]
        doc_id = cid
    else:
        pdf_bytes = uploaded.getvalue()
        doc_id = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded.name).stem)[:40] or "uploaded"

providers = ["offline"] + available_providers()
provider = st.sidebar.selectbox(
    "Extractor", providers,
    help="LLM providers appear when OPENAI_API_KEY / DEEPSEEK_API_KEY / GROQ_API_KEY is set (env or .env).",
)
if len(providers) == 1:
    st.sidebar.caption("No LLM API key found - using the deterministic offline extractor.")
r = ws.cfg.retriever
st.sidebar.markdown(
    f"**Tuned retriever**: `{r.method}`, alpha={r.alpha:.1f}, query expansion={'on' if r.expand_queries else 'off'}, "
    f"embedder `{getattr(ws.embedder, 'name', 'lsa')}`"
)

processed = processed_for(doc_id, pdf_bytes)
digest = hashlib.sha1(pdf_bytes).hexdigest()

st.title("Financial AI Agent & RAG Auditing Pipeline")
st.caption(
    "Parse credit agreements -> chunk -> hybrid retrieval -> structured extraction with guardrails -> "
    "reconcile covenants against the Master Ledger. All entities are fictional."
)

tabs = st.tabs(["Document", "Chunks", "Retrieval inspector", "Extraction & guardrails", "Reconciliation", "Evaluation"])

# --------------------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------------------
with tabs[0]:
    parsed = processed.parsed
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", len(parsed.pages))
    c2.metric("Tables detected", parsed.n_tables)
    c3.metric("Header/footer lines stripped", parsed.n_stripped)
    c4.metric("Chunks", len(processed.chunks))
    page_no = st.number_input("Page", min_value=1, max_value=max(1, len(parsed.pages)), value=1, step=1)
    left, right = st.columns([1, 1])
    with left:
        st.image(page_png(digest, pdf_bytes, int(page_no)), caption=f"Rendered page {page_no}", width="stretch")
    with right:
        page = parsed.pages[int(page_no) - 1]
        st.markdown("**Parsed elements (reading order)**")
        for el in page.elements:
            if el.kind == "table" and el.rows:
                st.markdown("`table` (kept intact, exported as Markdown)")
                st.dataframe(pd.DataFrame(el.rows[1:], columns=[f"{h or '-'} ({i})" for i, h in enumerate(el.rows[0])]),
                             hide_index=True, width="stretch")
            elif el.kind == "heading":
                st.markdown(f"`heading` **{el.text}**")
            else:
                st.markdown(f"`text` {el.text}")
        with st.expander("Stripped header / footer lines"):
            st.write(page.stripped or "none")

# --------------------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------------------
with tabs[1]:
    chunks = processed.chunks
    cdf = pd.DataFrame(
        [{"chunk_id": c.chunk_id, "section": c.section, "pages": c.page_label, "tokens": c.n_tokens,
          "table": c.has_table, "preview": c.text[:120].replace("\n", " ")} for c in chunks]
    )
    st.markdown(
        f"Structure-aware chunking: max {ws.cfg.chunker.max_tokens} tokens, overlap {ws.cfg.chunker.overlap_tokens} tokens, "
        "section boundaries are hard, tables are atomic."
    )
    fig = px.histogram(cdf, x="tokens", nbins=25, color_discrete_sequence=[SERIES["BM25"]],
                       labels={"tokens": "tokens per chunk"}, title="Chunk size distribution")
    fig.update_layout(height=280, bargap=0.08, yaxis_title="chunks", margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cdf, hide_index=True, width="stretch")
    if chunks:
        pick = st.selectbox("Show chunk", [c.chunk_id for c in chunks])
        ch = next(c for c in chunks if c.chunk_id == pick)
        st.caption(f"{ch.section} | {ch.page_label} | {ch.n_tokens} tokens")
        box(html.escape(ch.context_text))

# --------------------------------------------------------------------------------------
# Retrieval inspector
# --------------------------------------------------------------------------------------
with tabs[2]:
    if processed.index is None:
        st.warning("No text could be extracted from this PDF.")
    else:
        q1, q2 = st.columns([2, 1])
        options = [t.value for t in CovenantType] + list(LOAN_TERM_QUERIES) + ["custom"]
        qtype = q1.selectbox("Query", options, format_func=lambda o: COVENANT_LABELS.get(CovenantType(o), o) if o in CovenantType._value2member_map_ else o)
        expand = q2.checkbox("Obligation-context query expansion", value=r.expand_queries)
        if qtype == "custom":
            query = st.text_input("Custom query", "minimum Tangible Net Worth")
        elif qtype in LOAN_TERM_QUERIES:
            query = LOAN_TERM_QUERIES[qtype]
        else:
            query = covenant_query(CovenantType(qtype), expand)
        m1, m2, m3 = st.columns(3)
        method = m1.selectbox("Fusion", ["rrf", "weighted", "bm25", "dense"], index=["rrf", "weighted", "bm25", "dense"].index(r.method))
        alpha = m2.slider("alpha (dense weight)", 0.0, 1.0, float(r.alpha), 0.1)
        top_k = m3.slider("top-k", 1, 10, 5)
        st.caption(f"Query sent to the retrievers: *{query}*")
        hits = processed.index.search(query, top_k=top_k, method=method, alpha=alpha)
        hdf = pd.DataFrame(
            [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "section": h.chunk.section, "pages": h.chunk.page_label,
              "bm25": round(h.bm25_score, 3), "bm25_rank": h.bm25_rank, "dense": round(h.dense_score, 3),
              "dense_rank": h.dense_rank, "fused": round(h.score, 4)} for h in hits]
        )
        # normalise each score family to [0, 1] over the shown hits so they share one axis
        long = []
        for col, name in (("bm25", "BM25"), ("dense", "Dense"), ("fused", "Fused")):
            v = hdf[col].astype(float)
            span = (v.max() - v.min()) or 1.0
            for cid_, val, raw in zip(hdf["chunk_id"], (v - v.min()) / span, v):
                long.append({"chunk": cid_, "score family": name, "normalised score": val, "raw": raw})
        fig = px.bar(pd.DataFrame(long), x="chunk", y="normalised score", color="score family", barmode="group",
                     color_discrete_map=SERIES, hover_data={"raw": True}, title="Scores of the top-k chunks (min-max normalised per family)")
        fig.update_layout(height=320, bargap=0.25, bargroupgap=0.08, margin=dict(t=40, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(hdf, hide_index=True, width="stretch")
        rel: set[str] = set()
        if gt is not None and qtype in CovenantType._value2member_map_:
            ev = next((c for c in gt["covenants"] if c["covenant_type"] == qtype), None)
            if ev is not None:
                rel = relevant_chunk_ids(processed, ev["evidence"])
                st.info(f"Ground-truth clause is in chunk(s): {', '.join(sorted(rel)) or 'none'}")
        for h in hits:
            mark = "  <- ground truth" if h.chunk.chunk_id in rel else ""
            with st.expander(f"#{h.rank} {h.chunk.chunk_id} ({h.chunk.page_label}){mark}"):
                box(html.escape(h.chunk.context_text))

# --------------------------------------------------------------------------------------
# Extraction & guardrails
# --------------------------------------------------------------------------------------
with tabs[3]:
    if processed.index is None:
        st.warning("No text could be extracted from this PDF.")
    else:
        run_llm = True
        if provider != "offline":
            run_llm = st.button(f"Run extraction with {provider}", type="primary")
            st.caption("LLM calls cost money; the offline extractor runs automatically.")
        if provider == "offline" or run_llm:
            extractor = build_extractor(provider, ws.cfg.extractor, ws.cfg.guardrails)
            with st.spinner(f"Extracting with {extractor.name}..."):
                result = extractor.extract(doc_id, processed.index, processed.page_texts)
            st.session_state[f"extraction::{doc_id}::{provider}"] = result
        result = st.session_state.get(f"extraction::{doc_id}::{provider}")
        if result is None:
            st.info("Press the button to run the LLM extractor.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Extractor", result.extractor)
            k2.metric("Covenants verified", len(result.accepted_covenants()))
            k3.metric("Rejected by guardrails", len(result.rejected_covenants()))
            k4.metric("LLM attempts", result.attempts)
            if result.errors:
                with st.expander("Validation errors fed back to the model"):
                    for e in result.errors:
                        st.code(e)
            st.subheader("Loan terms")
            lt = result.loan_terms
            if lt is not None:
                rows = []
                for field, value in [("borrower", lt.borrower), ("principal", lt.principal), ("currency", lt.currency),
                                     ("rate_type", lt.rate_type.value if lt.rate_type else None), ("base_rate", lt.base_rate),
                                     ("rate_pct", lt.rate_pct), ("maturity_date", lt.maturity_date)]:
                    chk = result.loan_term_checks.get(field) or result.loan_term_checks.get(
                        {"currency": "principal", "rate_type": "rate_pct", "base_rate": "rate_pct"}.get(field, ""))
                    shown = "" if value is None else (f"{value:,.2f}" if isinstance(value, float) else str(value))
                    rows.append({"field": field, "value": shown, "guardrail": status_badge(chk.status.value) if chk else "-",
                                 "page": chk.located_page if chk else None})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.subheader("Financial covenants")
            cov_rows = []
            for vc in result.covenants:
                c = vc.covenant
                cov_rows.append({
                    "covenant": c.covenant_type.value, "operator": c.operator.value, "threshold": c.threshold,
                    "unit": c.unit.value, "frequency": c.test_frequency.value, "page": vc.citation_page,
                    "guardrail": status_badge(vc.check.status.value), "match": f"{vc.check.quote_match} ({vc.check.match_score:.2f})",
                    "reasons": "; ".join(vc.check.reasons),
                })
            if cov_rows:
                st.dataframe(pd.DataFrame(cov_rows), hide_index=True, width="stretch")
            else:
                st.warning("No covenant was extracted.")
            chunk_map = {c.chunk_id: c for c in processed.chunks}
            for vc in result.covenants:
                label = f"{status_badge(vc.check.status.value)} | {vc.covenant.covenant_type.value} {vc.covenant.operator.value} {vc.covenant.threshold:g}"
                with st.expander(label):
                    st.markdown(f"**Evidence quote:** {vc.covenant.evidence_quote}")
                    src_chunk = chunk_map.get(vc.check.chunk_id or "")
                    if src_chunk is not None:
                        st.caption(f"Source: {src_chunk.chunk_id}, {src_chunk.page_label}, {src_chunk.section}")
                        box(highlight(src_chunk.text, vc.covenant.evidence_quote))
                    for reason in vc.check.reasons:
                        st.caption(f"- {reason}")
            if gt is not None and source == "Generated contract":
                st.subheader("Ground truth (generator)")
                gdf = pd.DataFrame(gt["covenants"])[["covenant_type", "operator", "threshold", "unit", "test_frequency", "page", "phrasing", "evidence"]]
                st.dataframe(gdf, hide_index=True, width="stretch")
                m = extraction_metrics([result], {doc_id: gt})
                st.caption(
                    f"Field P/R/F1 = {m['field_precision']:.2f}/{m['field_recall']:.2f}/{m['field_f1']:.2f}, "
                    f"hallucination rate = {m['hallucination_rate']:.2f}"
                )
            st.subheader("This contract vs. the Master Ledger")
            ledger = ws.corpus.ledger
            single = findings_frame(audit_contract(result, compute_actuals(ledger), ledger, ws.cfg.audit))
            if single.empty:
                st.info("No findings (no verified covenants).")
            else:
                st.dataframe(style_status(single[["period", "covenant_type", "operator", "threshold", "actual_value",
                                                  "headroom_pct", "status", "source_page", "note"]]),
                             hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------
# Reconciliation dashboard
# --------------------------------------------------------------------------------------
with tabs[4]:
    report, _, gt_report = get_portfolio()
    summary = report["status"].value_counts().to_dict() if not report.empty else {}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Findings", len(report))
    k2.metric("Breaches", int(summary.get("BREACH", 0)))
    k3.metric("Insufficient data", int(summary.get("INSUFFICIENT_DATA", 0)))
    k4.metric("Borrowers in breach", report.loc[report["status"] == "BREACH", "contract_id"].nunique() if not report.empty else 0)
    if not report.empty:
        f1, f2 = st.columns(2)
        types = f1.multiselect("Covenant types", sorted(report["covenant_type"].unique()), default=sorted(report["covenant_type"].unique()))
        statuses = f2.multiselect("Statuses", list(STATUS_COLORS), default=list(STATUS_COLORS))
        view = report[report["covenant_type"].isin(types) & report["status"].isin(statuses)]

        # worst status per (borrower, period): BREACH > INSUFFICIENT_DATA > COMPLIANT
        rank = {"COMPLIANT": 0, "INSUFFICIENT_DATA": 1, "BREACH": 2}
        grid = view.assign(code=view["status"].map(rank)).pivot_table(index="borrower", columns="period", values="code", aggfunc="max")
        if not grid.empty:
            inv = {v: k for k, v in rank.items()}
            text = grid.map(lambda v: "" if pd.isna(v) else STATUS_SHORT[inv[int(v)]])
            hover = grid.map(lambda v: "not tested" if pd.isna(v) else inv[int(v)])
            colorscale = [[0, STATUS_COLORS["COMPLIANT"]], [0.33, STATUS_COLORS["COMPLIANT"]],
                          [0.34, STATUS_COLORS["INSUFFICIENT_DATA"]], [0.66, STATUS_COLORS["INSUFFICIENT_DATA"]],
                          [0.67, STATUS_COLORS["BREACH"]], [1, STATUS_COLORS["BREACH"]]]
            heat = go.Figure(go.Heatmap(
                z=grid.to_numpy(), x=list(grid.columns), y=list(grid.index), zmin=0, zmax=2, colorscale=colorscale,
                text=text.to_numpy(), texttemplate="%{text}", textfont={"size": 9}, customdata=hover.to_numpy(),
                hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>", xgap=2, ygap=2,
                colorbar=dict(tickvals=[0.33, 1, 1.67], ticktext=["COMPLIANT", "INSUFFICIENT_DATA", "BREACH"], title="worst status"),
            ))
            heat.update_layout(title="Worst covenant status per borrower and test period", height=max(320, 26 * len(grid) + 120),
                               margin=dict(t=40, b=10))
            st.plotly_chart(heat, width="stretch")

        counts = view[view["status"] == "BREACH"].groupby("borrower").size().sort_values(ascending=True)
        if not counts.empty:
            bar = px.bar(counts.reset_index(name="breaches"), x="breaches", y="borrower", orientation="h",
                         color_discrete_sequence=[STATUS_COLORS["BREACH"]], title="Breaches per borrower (count of test periods)")
            bar.update_layout(height=max(260, 24 * len(counts) + 100), margin=dict(t=40, b=10), yaxis_title="")
            st.plotly_chart(bar, width="stretch")

        st.markdown("**Actual vs. threshold**")
        s1, s2 = st.columns(2)
        borrower = s1.selectbox("Borrower", sorted(view["borrower"].unique()) or sorted(report["borrower"].unique()))
        sub_types = sorted(report.loc[report["borrower"] == borrower, "covenant_type"].unique())
        ctype = s2.selectbox("Covenant", sub_types)
        series = report[(report["borrower"] == borrower) & (report["covenant_type"] == ctype)].sort_values("period")
        if not series.empty:
            line = go.Figure()
            line.add_trace(go.Scatter(x=series["period"], y=series["actual_value"], mode="lines+markers", name="actual",
                                      line=dict(color=SERIES["BM25"], width=2), marker=dict(size=8),
                                      hovertemplate="%{x}: %{y:,.3f}<extra>actual</extra>"))
            line.add_trace(go.Scatter(x=series["period"], y=series["threshold"], mode="lines", name=f"threshold ({series['operator'].iloc[0]})",
                                      line=dict(color="#898781", width=2, dash="dash"),
                                      hovertemplate="%{x}: %{y:,.3f}<extra>threshold</extra>"))
            br = series[series["status"] == "BREACH"]
            if not br.empty:
                line.add_trace(go.Scatter(x=br["period"], y=br["actual_value"], mode="markers", name="breach",
                                          marker=dict(color=STATUS_COLORS["BREACH"], size=12, symbol="x"),
                                          hovertemplate="%{x}: BREACH<extra></extra>"))
            line.update_layout(height=320, margin=dict(t=30, b=10), yaxis_title=ctype, hovermode="x unified")
            st.plotly_chart(line, width="stretch")
            quote = series["evidence_quote"].iloc[0]
            st.caption(f"Citation: page {int(series['source_page'].iloc[0])} - \"{quote}\"")

        st.markdown("**Findings**")
        st.dataframe(style_status(view[["contract_id", "borrower", "period", "covenant_type", "operator", "threshold",
                                        "actual_value", "headroom_pct", "status", "source_page", "note"]]),
                     hide_index=True, width="stretch")
        st.download_button("Download audit report (CSV)", report.to_csv(index=False).encode("utf-8"), "audit_report.csv", "text/csv")

# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
with tabs[5]:
    metrics = ws.metrics if ws.source.startswith("trained") else None
    if metrics:
        st.markdown(f"Metrics from the last `train.py` run ({metrics.get('mode')} mode, test split of "
                    f"{len(metrics['split']['test'])} contracts).")
        ex, au, stt = metrics["extraction_test"], metrics["audit_test"], metrics["guardrail_stress_test"]
        k = st.columns(5)
        k[0].metric("Field F1", f"{ex['field_f1']:.3f}")
        k[1].metric("Numeric exact match", f"{ex['numeric_exact_match']:.3f}")
        k[2].metric("Hallucination rate", f"{ex['hallucination_rate']:.3f}")
        k[3].metric("Breach recall", f"{au['breach_recall']:.3f}")
        k[4].metric("Hallucinations caught", f"{stt['stress_hallucination_detection']:.3f}")
        comp_path = ART_DIR / "retrieval_comparison.csv"
        if comp_path.is_file():
            comp = pd.read_csv(comp_path)
            long = comp.melt(id_vars=["retriever"], value_vars=["recall@1", "mrr"], var_name="metric", value_name="score")
            fig = px.bar(long, x="score", y="retriever", color="metric", barmode="group", orientation="h",
                         color_discrete_sequence=[SERIES["BM25"], SERIES["Dense"]], title="Retrieval on the test split")
            fig.update_layout(height=380, xaxis_range=[0, 1], margin=dict(t=40, b=10), yaxis_title="", legend_title_text="")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(comp, hide_index=True, width="stretch")
        tune_path = ART_DIR / "fusion_tuning.csv"
        if tune_path.is_file():
            tune = pd.read_csv(tune_path)
            tune["config"] = tune["method"] + tune["expand"].map({True: " + expansion", False: ""})
            palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
            fig = px.line(tune, x="alpha", y="mrr", color="config", markers=True, color_discrete_sequence=palette,
                          title="Dev MRR vs. alpha (weight of the dense retriever)")
            fig.update_traces(line=dict(width=2), marker=dict(size=8))
            fig.update_layout(height=340, margin=dict(t=40, b=10), legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        st.markdown("**Extraction (test split)**")
        st.json({k_: round(v, 4) for k_, v in ex.items()}, expanded=False)
        st.markdown("**Guardrail stress test**")
        st.json({k_: round(v, 4) for k_, v in stt.items()}, expanded=False)
        st.markdown("**Breach detection (test split)**")
        st.json({k_: round(v, 4) for k_, v in au.items()}, expanded=False)
    else:
        st.info("No trained metrics found - showing a small in-memory evaluation of the demo corpus. "
                "Run `python train.py` for the full evaluation.")
        ids = [c.contract_id for c in ws.corpus.contracts]
        procs = {c.contract_id: processed_for(c.contract_id, c.pdf_bytes) for c in ws.corpus.contracts}
        rows = []
        for name, method, alpha, expand in [("BM25", "bm25", 0.0, False), ("Dense", "dense", 1.0, False),
                                            ("RRF", "rrf", 0.5, False), ("RRF + expansion", "rrf", 0.5, True)]:
            m_, _ = evaluate_retrieval(procs, ws.corpus.ground_truth, ids, method, alpha, expand=expand)
            rows.append({"retriever": name, **{k_: round(v, 3) for k_, v in m_.items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        report, results, gt_report = get_portfolio()
        if results:
            em = extraction_metrics(results, ws.corpus.ground_truth) | recall_by_phrasing(results, ws.corpus.ground_truth)
            st.json({k_: round(v, 4) for k_, v in em.items()}, expanded=False)
            ws_findings = audit_portfolio(results, ws.corpus.ledger, ws.cfg.audit)
            gt_findings = audit_portfolio([gt_to_extraction(ws.corpus.ground_truth[i]) for i in ids], ws.corpus.ledger, ws.cfg.audit)
            st.json({k_: round(v, 4) for k_, v in breach_metrics(ws_findings, gt_findings).items()}, expanded=False)
            st.caption(f"Audit summary: {summarize_findings(ws_findings)}")
