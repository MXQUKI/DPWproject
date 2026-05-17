"""Summarise the Tkinter UI operation log for filter/debug diagnosis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "ui_operation_log.jsonl"


def load_events(log_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not log_path.exists():
        return events

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def latest_session_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    app_start_indices = [index for index, event in enumerate(events) if event.get("event") == "app_started"]
    if not app_start_indices:
        return events
    return events[app_start_indices[-1] :]


def average_duration(events: list[dict[str, object]], event_name: str) -> float:
    values = [
        float(event["duration_ms"])
        for event in events
        if event.get("event") == event_name and isinstance(event.get("duration_ms"), (int, float))
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def print_report(events: list[dict[str, object]], label: str) -> None:
    if not events:
        print("No UI log events found.")
        return

    counts = Counter(str(event.get("event")) for event in events)
    print(f"UI Operation Log Report ({label})")
    print("=" * 32)
    print(f"Total events: {len(events)}")
    print(f"analysis_requested: {counts['analysis_requested']}")
    print(f"debounced: {counts['analysis_request_debounced']}")
    print(f"same_filters: {counts['analysis_request_same_filters']}")
    print(f"ignored_busy: {counts['analysis_request_ignored_busy']}")
    print(f"ignored_unchanged: {counts['analysis_request_ignored_unchanged']}")
    print(f"parse_errors: {counts['analysis_request_parse_error']}")
    print(f"discarded_results: {counts['analysis_result_discarded']}")
    print(f"reset_noop: {counts['reset_filters_noop']}")
    print(f"reset_cleared_pending: {counts['reset_filters_cleared_pending']}")
    print(f"chart_busy: {counts['chart_refresh_ignored_busy'] + counts['chart_refresh_ignored_analysis_busy']}")
    print(f"chart_empty: {counts['chart_refresh_empty_selection']}")
    print(f"page_boundary: {counts['page_change_boundary']}")
    print(f"avg analyze worker ms: {average_duration(events, 'analysis_worker_completed')}")
    print(f"avg apply result ms: {average_duration(events, 'analysis_applied')}")
    print(f"avg refresh view ms: {average_duration(events, 'view_refreshed')}")
    print(f"avg table populate ms: {average_duration(events, 'results_table_populated')}")
    print()
    print("Likely Cause")

    if counts["analysis_request_debounced"] > 0:
        print("- Some duplicate trigger paths were collapsed into a single analysis request.")
        print("  If you are checking an older run, these events may come from the legacy apply flow.")
    if counts["analysis_request_same_filters"] > 0:
        print("- Some apply actions reused the same filters, so the app correctly skipped a redundant refresh.")
    if counts["analysis_request_ignored_busy"] > 0:
        print("- Some clicks happened while a previous analysis was still running.")
    if counts["analysis_request_ignored_unchanged"] > 0:
        print("- Some clicks reused the exact same filters, so there was no data change to apply.")
    if counts["analysis_request_parse_error"] > 0:
        print("- Some requests failed input parsing before analysis started.")
    if counts["analysis_result_discarded"] > 0:
        print("- Some older results returned after a newer request had already replaced them.")
    if counts["reset_filters_noop"] > 0 or counts["reset_filters_cleared_pending"] > 0:
        print("- Some Reset clicks were valid acknowledgements but did not require a new analysis run.")
    if counts["chart_refresh_ignored_busy"] > 0 or counts["chart_refresh_ignored_analysis_busy"] > 0:
        print("- Some chart refresh clicks happened while another update was already running.")
    if counts["chart_refresh_empty_selection"] > 0:
        print("- Some chart refresh clicks were blocked because the current selection had no rows.")
    if counts["page_change_boundary"] > 0:
        print("- Some page navigation clicks hit the first/last page boundary and were acknowledged without moving.")
    if (
        counts["analysis_request_debounced"] == 0
        and counts["analysis_request_same_filters"] == 0
        and counts["analysis_request_ignored_busy"] == 0
        and counts["analysis_request_ignored_unchanged"] == 0
        and counts["analysis_request_parse_error"] == 0
        and counts["analysis_result_discarded"] == 0
        and counts["reset_filters_noop"] == 0
        and counts["reset_filters_cleared_pending"] == 0
        and counts["chart_refresh_ignored_busy"] == 0
        and counts["chart_refresh_ignored_analysis_busy"] == 0
        and counts["chart_refresh_empty_selection"] == 0
        and counts["page_change_boundary"] == 0
    ):
        print("- No obvious rejection pattern found. Check recent timings below.")

    print()
    print("Recent Events")
    for event in events[-15:]:
        timestamp = event.get("timestamp", "-")
        name = event.get("event", "-")
        request_id = event.get("request_id", event.get("analysis_request_id", "-"))
        details = []
        for key in ("source", "duration_ms", "rows", "message"):
            if key in event:
                details.append(f"{key}={event[key]}")
        print(f"- {timestamp} | {name} | request={request_id} | {'; '.join(details)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the IMDB UI operation log.")
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="Path to the ui_operation_log.jsonl file.")
    parser.add_argument("--all", action="store_true", help="Report the full log history instead of only the latest app session.")
    args = parser.parse_args()

    log_path = Path(args.log)
    all_events = load_events(log_path)
    if args.all:
        print_report(all_events, "full history")
        return

    print_report(latest_session_events(all_events), "latest session")


if __name__ == "__main__":
    main()
