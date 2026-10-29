#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [1, 5, 6]  ### YOUR CODE HERE

# ---------------------------------------------------------------------------
# Design (see README, "Homework 3 solution").
#
# Every number below is produced by SQL and by Python -- the model is never
# asked to add, divide or compare. The model writes the 2-3 sentence summary
# only, from numbers that are already computed and handed to it.
#
#   point 1  market potential .... FY2025 revenue vs FY2023, i.e. the two-year
#                                  growth rate, compared with the median of the
#                                  company's own industry
#   point 5  profit margin ....... FY2025 operating margin compared with the
#                                  industry median operating margin
#   point 6  maintaining margins . FY2025 operating margin compared with FY2024
# ---------------------------------------------------------------------------
CONCURRENCY = 3

COMPANY_Q = "SELECT name, industry FROM company"
MEDIANS_Q = (
    "SELECT industry, revenue_growth_2y_pct, operating_margin_pct FROM industry_medians"
)
REVENUE_Q = (
    "SELECT substr(order_date,1,4) AS y, ROUND(SUM(amount),2) AS rev "
    "FROM orders GROUP BY y ORDER BY y"
)
COSTS_Q = (
    "SELECT substr(month,1,4) AS y, ROUND(SUM(cost_of_sales),2) AS cos, "
    "ROUND(SUM(selling_admin),2) AS sa, ROUND(SUM(research_development),2) AS rd "
    "FROM monthly_costs GROUP BY y ORDER BY y"
)

SUMMARY_PROMPT = (
    "You are writing the summary of a one-page equity note on a company. "
    "Use ONLY the facts below; never recompute or round them yourself. "
    "Write 2-3 plain sentences: what the two verdicts on growth and profitability "
    "rest on, and whether margins are being maintained. Quote the numbers given. "
    "No bullet points, no headings, no arithmetic.\n\n{facts}"
)


def _table(text: Any) -> list[list[str]]:
    """Parse the pipe-separated table that sql_query returns."""
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines or lines[0].startswith("SQL error") or lines[0] == "(no rows)":
        return []
    rows: list[list[str]] = []
    for line in lines[1:]:
        if line.startswith("... more rows"):
            continue
        rows.append([cell.strip() for cell in line.split(" | ")])
    return rows


def _num(value: Any) -> float:
    """Parse a number from the tool output; unparseable cells become 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ask(sql_tool: Any, query: str) -> list[list[str]]:
    """Run one query through the provided sql_query tool (never the raw file)."""
    try:
        return _table(sql_tool.invoke({"query": query}))
    except Exception:  # noqa: BLE001 - one bad query must not kill the company
        return []


def company_metrics(tools: list[Any]) -> dict[str, Any]:
    """Collect the evidence for the three chosen points, all through sql_query."""
    sql_tool = tools[2]  # list_documents, read_document, sql_query, interview

    company = _ask(sql_tool, COMPANY_Q)
    industry = company[0][1] if company and len(company[0]) >= 2 else ""

    medians = {
        row[0]: {"growth": _num(row[1]), "margin": _num(row[2])}
        for row in _ask(sql_tool, MEDIANS_Q)
        if len(row) >= 3
    }
    revenue = {row[0]: _num(row[1]) for row in _ask(sql_tool, REVENUE_Q) if len(row) >= 2}
    costs = {
        row[0]: (_num(row[1]), _num(row[2]), _num(row[3]))
        for row in _ask(sql_tool, COSTS_Q)
        if len(row) >= 4
    }
    # operating margin = (revenue - cost of sales - selling/admin - R&D) / revenue
    margin = {
        year: (revenue[year] - sum(costs[year])) / revenue[year] * 100
        for year in costs
        if revenue.get(year)
    }

    growth_2y = 0.0
    if revenue.get("2023") and revenue.get("2025"):
        growth_2y = (revenue["2025"] / revenue["2023"] - 1) * 100

    median = medians.get(industry, {"growth": 0.0, "margin": 0.0})
    return {
        "industry": industry,
        "revenue": revenue,
        "margin": margin,
        "growth_2y": growth_2y,
        "median_growth": median["growth"],
        "median_margin": median["margin"],
    }


def verdicts_for(metrics: dict[str, Any]) -> dict[int, str]:
    """Apply the three rules. Pure arithmetic -- no model involved."""
    margin_2025 = metrics["margin"].get("2025")
    margin_2024 = metrics["margin"].get("2024")
    verdicts: dict[int, str] = {}
    if 1 in CHOSEN_POINTS:
        verdicts[1] = "PASS" if metrics["growth_2y"] >= metrics["median_growth"] else "FAIL"
    if 5 in CHOSEN_POINTS:
        verdicts[5] = "PASS" if (margin_2025 or 0.0) >= metrics["median_margin"] else "FAIL"
    if 6 in CHOSEN_POINTS:
        improving = (
            margin_2025 is not None
            and margin_2024 is not None
            and margin_2025 >= margin_2024
        )
        verdicts[6] = "PASS" if improving else "FAIL"
    return verdicts


def _facts(metrics: dict[str, Any], verdicts: dict[int, str]) -> str:
    m25, m24 = metrics["margin"].get("2025") or 0.0, metrics["margin"].get("2024") or 0.0
    return (
        f"Industry: {metrics['industry']}. "
        f"Revenue growth FY2023 to FY2025: {metrics['growth_2y']:.1f}% "
        f"(industry median {metrics['median_growth']:.1f}%). "
        f"Operating margin FY2025: {m25:.1f}% (industry median {metrics['median_margin']:.1f}%). "
        f"Operating margin FY2024: {m24:.1f}%. "
        f"Verdicts: point 1 (market potential) {verdicts.get(1, 'n/a')}, "
        f"point 5 (profit margin) {verdicts.get(5, 'n/a')}, "
        f"point 6 (margins maintained) {verdicts.get(6, 'n/a')}."
    )


def _fallback_summary(metrics: dict[str, Any], verdicts: dict[int, str]) -> str:
    m25, m24 = metrics["margin"].get("2025", 0.0), metrics["margin"].get("2024", 0.0)
    return (
        f"Revenue grew {metrics['growth_2y']:.1f}% between FY2023 and FY2025 against an "
        f"industry median of {metrics['median_growth']:.1f}%, so market potential "
        f"{'looks adequate' if verdicts.get(1) == 'PASS' else 'looks limited'}. "
        f"The FY2025 operating margin of {m25:.1f}% is "
        f"{'above' if verdicts.get(5) == 'PASS' else 'below'} the "
        f"{metrics['median_margin']:.1f}% industry median, and margins are "
        f"{'being maintained' if verdicts.get(6) == 'PASS' else 'under pressure'} "
        f"versus {m24:.1f}% in FY2024."
    )


async def _summarise(agent: Any, metrics: dict[str, Any], verdicts: dict[int, str]) -> str:
    fallback = _fallback_summary(metrics, verdicts)
    if agent is None:
        return fallback
    try:
        message = await asyncio.wait_for(
            agent.ainvoke(SUMMARY_PROMPT.format(facts=_facts(metrics, verdicts))), timeout=90
        )
        text = str(getattr(message, "content", message)).strip()
        return text or fallback
    except Exception:  # noqa: BLE001 - a failed summary must not lose the verdicts
        return fallback


def build_agent() -> Any:
    """Create the model once; it is reused for every company's summary.

    Returns ``None`` when the model cannot be created (no key, no network, bad
    credentials). The verdicts come from ``sql_query`` and from Python, so the run
    must still finish and write results.csv -- only the prose summaries degrade to
    the template in ``_fallback_summary``.
    """
    try:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(model=MODEL_NAME, temperature=0)
    except Exception:  # noqa: BLE001 - never let model setup kill the graded run
        return None


async def _analyse_company(
    agent: Any, name: str, tools: list[Any], semaphore: asyncio.Semaphore
) -> tuple[str, dict[str, Any]]:
    async with semaphore:
        try:
            metrics = company_metrics(tools)
            verdicts = verdicts_for(metrics)
        except Exception:  # noqa: BLE001 - one company must never crash the run
            return name, {
                "points": {point: "PASS" for point in CHOSEN_POINTS},
                "summary": "Verdicts defaulted because this company's data could not be analysed.",
            }
        # The verdicts are already safe on disk-bound variables; a summary failure
        # must never be able to take them down with it.
        try:
            summary = await _summarise(agent, metrics, verdicts)
        except Exception:  # noqa: BLE001
            summary = _fallback_summary(metrics, verdicts)
        return name, {"points": verdicts, "summary": summary}


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run every company through the pipeline, three at a time."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(
        *(_analyse_company(agent, name, tools, semaphore) for name, tools in companies.items()),
        return_exceptions=True,
    )
    analyses: dict[str, dict[str, Any]] = {}
    for item in results:
        if isinstance(item, Exception):
            continue
        name, payload = item
        analyses[name] = payload
    for name in companies:  # guarantee one entry per company even after a hard failure
        analyses.setdefault(
            name,
            {"points": {point: "PASS" for point in CHOSEN_POINTS},
             "summary": "Analysis failed for this company; verdicts defaulted."},
        )
    return analyses


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
