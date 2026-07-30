"""Split navigation/layout table blocks into individual text blocks.

Paddle's markdown pipeline sometimes wraps multi-column text layouts
(navigation sidebars, style guides, tabular reference lists) into a
single ``<table>`` HTML block. Because ``block_type=table`` gates
translation with ``policy.translate=False``, all text inside such tables
is silently dropped.

This module detects those pseudo-tables by structure and splits them
back into individual ``text/body`` blocks with estimated bounding boxes
so that downstream translation and overlay rendering work normally.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_MIN_CELLS_FOR_NAV = 4
_MAX_COLS_FOR_NAV = 5


def is_navigation_table(block: dict) -> bool:
    """Return True when *block* is a pseudo-table (navigation / list layout)."""
    if block.get("type") != "table":
        return False
    raw = (block.get("text") or block.get("raw_text") or "").strip()
    if not raw.lower().startswith("<table"):
        return False
    return _classify_html_table(raw) == "navigation"


def _classify_html_table(html: str) -> str:
    parser = _TableClassifier()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return "unknown"
    if not (_MIN_CELLS_FOR_NAV <= parser.cell_count):
        return "unknown"
    if not (2 <= parser.max_cols <= _MAX_COLS_FOR_NAV):
        return "data" if parser.max_cols > _MAX_COLS_FOR_NAV else "unknown"
    # Layout/navigation tables have rowspan cells (content grouping) or
    # images (example thumbnails).  Real data tables have regular grids.
    if parser.rowspan_count > 0 or parser.image_count > 0:
        return "navigation"
    return "data"


class _TableClassifier(HTMLParser):
    """Minimal single-pass classifier — structure only."""

    def __init__(self) -> None:
        super().__init__()
        self.max_cols = 0
        self.cell_count = 0
        self.rowspan_count = 0
        self.image_count = 0
        self._cur_cols = 0
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tl = tag.lower()
        if tl == "table":
            self._depth += 1
        elif tl in ("td", "th") and self._depth > 0:
            self._cur_cols += 1
            self.cell_count += 1
            if int(dict(attrs).get("rowspan", 1)) > 1:
                self.rowspan_count += 1
        elif tl == "img" and self._depth > 0:
            self.image_count += 1

    def handle_endtag(self, tag: str) -> None:
        tl = tag.lower()
        if tl == "tr" and self._depth > 0:
            self.max_cols = max(self.max_cols, self._cur_cols)
            self._cur_cols = 0
        elif tl == "table":
            self._depth -= 1


# ---------------------------------------------------------------------------
# Cell parser
# ---------------------------------------------------------------------------

_IMG_SRC_RE = re.compile(r'<img\s[^>]*src="([^"]*)"', re.DOTALL | re.IGNORECASE)
_STRIP_TAGS_RE = re.compile(r"<[^>]*>")


def _extract_cells(html: str) -> list[list[dict]]:
    """Parse table HTML into rows of cells.

    Each cell dict::

        {"text": str, "images": list[str],
         "rowspan": int, "colspan": int}
    """
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    i = 0
    in_table = False

    while i < len(html):
        if html[i] != "<":
            i += 1
            continue
        end = html.find(">", i)
        if end == -1:
            break
        raw_tag = html[i + 1 : end].strip()
        tag_name = raw_tag.split()[0].lower() if raw_tag else ""

        if tag_name in ("table",):
            in_table = True
        elif tag_name in ("/table",):
            in_table = False
            if current_row:
                rows.append(current_row)
            break
        elif tag_name in ("tr",):
            if current_row:
                rows.append(current_row)
            current_row = []
        elif tag_name == "/tr" and in_table:
            if current_row:
                rows.append(current_row)
            current_row = []
        elif tag_name in ("td", "th") and in_table:
            j = end + 1
            inner = ""
            depth = 1
            close_td = f"</{tag_name}"
            while j < len(html) and depth > 0:
                if html[j] == "<":
                    ce = html.find(">", j)
                    if ce == -1:
                        break
                    inner_tag = html[j + 1 : ce].split()[0].lower()
                    if inner_tag == tag_name:
                        depth += 1
                    elif inner_tag == f"/{tag_name}":
                        depth -= 1
                    elif depth == 1 and inner_tag[0] != "/":
                        inner += html[j : ce + 1]
                    j = ce + 1
                else:
                    if depth == 1:
                        inner += html[j]
                    j += 1

            attrs = {}
            for part in raw_tag.split()[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    attrs[k.lower()] = v.strip('"').strip("'")
            cell = {
                "text": _STRIP_TAGS_RE.sub("", inner).strip(),
                "images": _IMG_SRC_RE.findall(inner),
                "rowspan": int(attrs.get("rowspan", 1)),
                "colspan": int(attrs.get("colspan", 1)),
            }
            current_row.append(cell)
            i = j
            continue
        i = end + 1

    if current_row and in_table:
        rows.append(current_row)
    return rows


# ---------------------------------------------------------------------------
# Bounding-box estimation
# ---------------------------------------------------------------------------

_COL_FRACTIONS: dict[int, tuple[float, ...]] = {
    2: (0.38, 0.62),
    3: (0.22, 0.56, 0.22),
    4: (0.28, 0.12, 0.32, 0.28),
    5: (0.20, 0.12, 0.28, 0.24, 0.16),
}


def _cell_bbox(
    row: int,
    col: int,
    rowspan: int,
    colspan: int,
    nrows: int,
    ncols: int,
    tbbox: list[float],
) -> list[float]:
    tx0, ty0, tx1, ty1 = tbbox
    tw = max(tx1 - tx0, 1)
    th = max(ty1 - ty0, 1)
    fracs = _COL_FRACTIONS.get(ncols)
    if fracs:
        cw = [f * tw for f in fracs[:ncols]]
        while len(cw) < ncols:
            cw.append(tw / ncols)
    else:
        cw = [tw / ncols] * ncols
    rh = th / max(nrows, 1)
    x0 = max(tx0, tx0 + sum(cw[:col]))
    x1 = max(tx0, tx0 + sum(cw[: col + colspan]))
    y0 = max(ty0, ty0 + row * rh)
    y1 = max(ty0, ty0 + (row + rowspan) * rh)
    return [round(v, 3) for v in (x0, y0, x1, y1)]


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------


def _build_text_block(
    text: str,
    bbox: list[float],
    page_index: int,
    order: int,
    raw_block_id: int,
) -> dict:
    """Return a block record (post-``build_block_record`` level) for a single
    table-cell text, with fields that let ``apply_document_defaults`` /
    ``enrich_document_contract_v1`` compute the correct translate policy."""
    lines = [
        {
            "bbox": bbox,
            "spans": [
                {
                    "type": "text",
                    "raw_type": "text",
                    "text": text,
                    "bbox": bbox,
                    "score": None,
                }
            ],
        }
    ]
    segments = [
        {
            "type": "text",
            "raw_type": "text",
            "text": text,
            "bbox": bbox,
            "score": None,
        }
    ]

    return {
        "block_id": f"p{page_index + 1:03d}-b{order:04d}",
        "page_index": page_index,
        "order": order,
        "type": "text",
        "sub_type": "body",
        "bbox": bbox,
        "text": text,
        "lines": lines,
        "segments": segments,
        "tags": [],
        # Explicitly empty so apply_document_defaults leaves them as-is
        "derived": {"role": "", "by": "", "confidence": 0.0},
        "continuation_hint": {
            "source": "",
            "group_id": "",
            "role": "",
            "scope": "",
            "reading_order": -1,
            "confidence": 0.0,
        },
        "metadata": {
            "content_is_rich": False,
            "content_length": len(text),
            "content_line_count": max(text.count("\n") + 1, 1),
            "layout_det_matched": False,
            "provider": "paddle",
            "raw_text_excerpt": text[:80],
            "raw_type": "text",
            "raw_sub_type": "",
            "raw_bbox": bbox,
        },
        "source": {
            "provider": "paddle",
            "raw_page_index": page_index,
            "raw_type": "text",
            "raw_sub_type": "",
            "raw_bbox": bbox,
            "raw_text_excerpt": text[:80],
            "raw_block_id": raw_block_id,
            "raw_path": f"table_split/{raw_block_id}",
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def split_navigation_table(table_block: dict, page_index: int, start_order: int) -> list[dict]:
    """Split a navigation-style table block into individual text blocks.

    Returns a list of block dicts matching the ``build_block_record`` output
    format.  Returns an empty list when the block is not a navigation table.
    """
    if not is_navigation_table(table_block):
        return []

    html = (table_block.get("text") or "").strip()
    tbbox = table_block.get("bbox", [0, 0, 0, 0]) or [0, 0, 0, 0]
    raw_block_id = (table_block.get("source") or {}).get("raw_block_id", -1)

    cells = _extract_cells(html)
    if not cells:
        return []

    nrows = len(cells)
    ncols = max((len(r) for r in cells), default=4)

    blocks: list[dict] = []
    order = start_order

    for row_idx, row in enumerate(cells):
        for col_idx, cell in enumerate(row):
            text = cell.get("text", "").strip()
            if not text:
                continue
            bbox = _cell_bbox(row_idx, col_idx, cell.get("rowspan", 1),
                              cell.get("colspan", 1), nrows, ncols, tbbox)
            block = _build_text_block(text, bbox, page_index, order, raw_block_id)
            blocks.append(block)
            order += 1

    return blocks
