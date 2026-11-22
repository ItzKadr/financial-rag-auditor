import re

import pymupdf

from src.pdf_parser import parse_pdf, table_to_markdown
from src.text_utils import compact


def test_parser_keeps_tables_and_pages(corpus, processed) -> None:
    for c in corpus.contracts:
        parsed = processed[c.contract_id].parsed
        gt = c.ground_truth
        assert len(parsed.pages) == gt["n_pages"]
        expected_tables = 1 + int(gt["has_ratchet"]) + int(any(x["presentation"] == "table" for x in gt["covenants"]))
        assert parsed.n_tables == expected_tables
        for cov in gt["covenants"]:
            page = parsed.pages[cov["page"] - 1]
            if cov["presentation"] == "table":
                cells = [x.strip() for x in cov["evidence"].split("|")]
                tables = [t for t in page.tables if any(r == cells for r in t.rows or [])]
                assert tables, f"row {cells} not found as a table row on page {cov['page']}"
                assert "|" + "|".join(cells) + "|" in tables[0].text  # Markdown row preserved
            else:
                assert compact(cov["evidence"]) in compact(page.text)


def test_headers_and_footers_are_stripped(corpus, processed, first_id) -> None:
    parsed = processed[first_id].parsed
    borrower = corpus.ground_truth[first_id]["borrower"]
    for page in parsed.pages:
        assert not re.search(r"Page \d+ of \d+", page.text)
        assert "CONFIDENTIAL - EXECUTION VERSION" not in page.text
        assert f"{borrower} - Credit Agreement" not in page.text
        assert any(s.startswith("Page ") for s in page.stripped)


def test_headings_are_detected(processed, first_id) -> None:
    headings = [e.text for e in processed[first_id].parsed.elements if e.kind == "heading"]
    assert any(h.startswith("ARTICLE 1 - DEFINITIONS") for h in headings)
    assert any("FINANCIAL COVENANTS" in h for h in headings)


def _two_column_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "A full width introductory paragraph that spans across the centre of the page.", fontname="tiro", fontsize=10)
    for i in range(4):
        page.insert_text((72, 140 + 13 * i), f"LEFT line {i}", fontname="tiro", fontsize=10)
        page.insert_text((330, 140 + 13 * i), f"RIGHT line {i}", fontname="tiro", fontsize=10)
    page.insert_text((72, 260), "A closing full width paragraph after the two column block on this page.", fontname="tiro", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_two_column_reading_order() -> None:
    parsed = parse_pdf(_two_column_pdf(), doc_id="twocol")
    text = parsed.full_text
    # MuPDF interleaves LEFT/RIGHT lines in one block; the parser must read column by column.
    positions = [text.index(f"LEFT line {i}") for i in range(4)] + [text.index(f"RIGHT line {i}") for i in range(4)]
    assert positions == sorted(positions)
    assert text.index("introductory") < positions[0] and text.index("closing") > positions[-1]


def test_table_to_markdown_pads_rows() -> None:
    md = table_to_markdown([["A", "B", "C"], ["1", "2"]])
    assert md.splitlines() == ["|A|B|C|", "|---|---|---|", "|1|2||"]
