"""Tkinter UI for the integrated IMDB movies analysis project."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import queue
from time import perf_counter
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk

from advanced_analytics import get_comprehensive_analysis
from data_analysis import analyze
from data_preprocessing import prepare_dataset
from data_visualization import create_project_visuals

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class IMDbExplorerApp:
    """Desktop explorer for filtering, analysing, and visualising the dataset."""

    PAGE_SIZE = 120
    FILTER_FEEDBACK_DEBOUNCE_MS = 90
    GLOBAL_CHART_KEYS = frozenset({"forecast_backtest_yearly_comparison"})
    DEFAULT_SORT = (("vote_average", False), ("revenue", False), ("title", True))
    RESULT_COLUMN_MAP = {
        "id": "id",
        "title": "title",
        "genre": "primary_genre",
        "release_date": "release_date",
        "year": "year",
        "rating": "vote_average",
        "runtime": "runtime",
        "budget": "budget",
        "revenue": "revenue",
    }
    CHART_OPTIONS = (
        (
            "genre_distribution",
            "Genre Distribution",
            "Shows which genres appear most often in the current movie selection.",
        ),
        (
            "genre_comparison",
            "Genre Comparison",
            "Compares genre volume and average rating in the same figure.",
        ),
        (
            "yearly_rating_trend",
            "Yearly Rating Trend",
            "Tracks release volume and average rating across years.",
        ),
        (
            "budget_revenue_scatter",
            "Budget vs Revenue",
            "Explores the relationship between production budget and revenue.",
        ),
        (
            "forecast_backtest_yearly_comparison",
            "Yearly Forecast vs Actual",
            "Shows each year's predicted total box office beside the actual total box office.",
        ),
    )
    CHART_FILENAMES = {
        "genre_distribution": "genre_distribution.png",
        "genre_comparison": "genre_comparison.png",
        "yearly_rating_trend": "yearly_rating_trend.png",
        "budget_revenue_scatter": "budget_revenue_scatter.png",
        "forecast_backtest_yearly_comparison": "forecast_backtest_yearly_comparison.png",
    }

    def __init__(self, root: tk.Tk, dataset_path: str | None = None) -> None:
        self.root = root
        self.root.title("IMDB Movies Dataset Explorer")
        self.root.geometry("1380x900")
        self.root.minsize(1180, 800)
        self.palette = {
            "app_bg": "#151210",
            "content_bg": "#ECE3D6",
            "sidebar_bg": "#1E1916",
            "sidebar_card": "#2A231F",
            "card_bg": "#FBF7F0",
            "card_alt": "#F6EFE4",
            "text_dark": "#2A211B",
            "text_muted": "#7A6A5B",
            "text_light": "#F3EBDD",
            "accent_gold": "#C9A45C",
            "accent_red": "#B35249",
            "accent_olive": "#71816D",
            "line": "#D8CCBD",
        }

        self.output_dir = Path(__file__).resolve().parent / "outputs"
        self.chart_manifest_path = self.output_dir / "chart_manifest.json"
        self.logs_dir = Path(__file__).resolve().parent / "logs"
        self.operation_log_path = self.logs_dir / "ui_operation_log.jsonl"
        self.dataset, self.quality_report = prepare_dataset(dataset_path, prefer_cleaned=True)

        self.analysis_request_id = 0
        self.chart_request_token = 0
        self.current_page = 0
        self.total_pages = 1
        self.analysis_in_progress = False
        self.chart_generation_in_progress = False
        self.charts_match_current_filters = False
        self.current_advanced: dict[str, object] = {}
        self.chart_paths: dict[str, str] = {}
        self.current_sorted_results = pd.DataFrame()
        self.current_display_rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        self.current_chart_photo: ImageTk.PhotoImage | None = None
        self.chart_render_job: str | None = None
        self.chart_canvas_image_id: int | None = None
        self.chart_preview_cache: dict[tuple[str, int], tuple[ImageTk.PhotoImage, tuple[int, int]]] = {}
        self.chart_canvas_width = 0
        self.current_filter_key: tuple[tuple[str, object], ...] = ()
        self.analysis_cache: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
        self.advanced_cache: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
        self.log_lock = threading.Lock()
        self.applied_control_snapshot: tuple[tuple[str, str], ...] = ()
        self.analysis_request_snapshots: dict[int, tuple[tuple[str, str], ...]] = {}
        self.filter_dirty = False
        self.sort_state: list[tuple[str, bool]] = list(self.DEFAULT_SORT)
        self.sorted_results_cache: dict[tuple[tuple[str, bool], ...], pd.DataFrame] = {}
        self.results_row_items: list[str] = []
        self.filter_feedback_job: str | None = None
        self.last_action_state_log_snapshot: tuple[bool, bool, bool] | None = None
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.log_writer_thread = threading.Thread(target=self._log_writer_worker, name="UILogWriter", daemon=True)
        self.log_writer_thread.start()

        self.current_analysis = analyze(self.dataset)
        self.current_results = self.current_analysis["filtered_df"]
        self.current_result_identity = self._build_result_identity(self.current_results)
        self.analysis_cache[()] = self.current_analysis
        self._reset_sort_cache()

        self.genre_var = tk.StringVar(value="All")
        self.year_from_var = tk.StringVar()
        self.year_to_var = tk.StringVar()
        self.rating_var = tk.StringVar()
        self.keyword_var = tk.StringVar()

        self.status_var = tk.StringVar(value="Preparing interface...")
        self.status_badge_var = tk.StringVar(value="WORKING")
        self.results_info_var = tk.StringVar(value="")
        self.last_action_var = tk.StringVar(value="No filter action yet.")
        self.pending_state_var = tk.StringVar(value="Filters are up to date.")
        self.applied_filters_var = tk.StringVar(value="Applied filters: All titles")
        self.results_banner_var = tk.StringVar(value="Dataset ready. Adjust filters to refine the result set.")
        self.page_var = tk.StringVar(value="Page 1 / 1")
        self.chart_status_var = tk.StringVar(value="")
        self.chart_title_var = tk.StringVar(value=self.CHART_OPTIONS[0][1])
        self.chart_info_var = tk.StringVar(value="")
        self.chart_file_var = tk.StringVar(value="")
        self.chart_description_var = tk.StringVar(value=self.CHART_OPTIONS[0][2])
        self.chart_choice_var = tk.StringVar(value=self.CHART_OPTIONS[0][0])
        self.dataset_scope_var = tk.StringVar(value="")
        self.selection_scope_var = tk.StringVar(value="")

        self.summary_vars = {
            "movie_count": tk.StringVar(),
            "genre_count": tk.StringVar(),
            "average_rating": tk.StringVar(),
            "total_revenue": tk.StringVar(),
        }

        self.chart_refresh_buttons: list[ttk.Button] = []

        self._configure_style()
        self._build_layout()
        self._bind_events()
        self._populate_genres()
        self._bind_filter_traces()
        self.applied_control_snapshot = self._current_control_snapshot_key()
        self._refresh_filter_feedback()
        self._render_quality_report()
        self._load_existing_charts()
        self._refresh_view(include_advanced=False)
        self._log_operation(
            "app_started",
            dataset_rows=int(len(self.dataset)),
            clean_rows=int(self.quality_report["clean_rows"]),
            log_path=str(self.operation_log_path),
        )
        self._set_status("Dataset ready.")
        self._start_advanced_analysis(self.current_results.copy(), self.analysis_request_id)

    def _configure_style(self) -> None:
        self.root.configure(bg=self.palette["app_bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.palette["content_bg"])
        style.configure("App.TFrame", background=self.palette["app_bg"])
        style.configure("Content.TFrame", background=self.palette["content_bg"])
        style.configure("HeroCard.TFrame", background=self.palette["sidebar_bg"])
        style.configure("HeroPanel.TFrame", background="#2A231F")
        style.configure("Sidebar.TFrame", background=self.palette["sidebar_bg"])
        style.configure("SidebarCard.TFrame", background=self.palette["sidebar_card"])
        style.configure("Card.TFrame", background=self.palette["card_bg"])
        style.configure("CardAlt.TFrame", background=self.palette["card_alt"])
        style.configure("Inset.TFrame", background="#FFFDFC")

        style.configure("TLabel", background=self.palette["content_bg"], foreground=self.palette["text_dark"], font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Georgia", 26, "bold"), background=self.palette["sidebar_bg"], foreground=self.palette["text_light"])
        style.configure("HeroBody.TLabel", font=("Helvetica", 11), background=self.palette["sidebar_bg"], foreground="#D7CCBF")
        style.configure("HeroKicker.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["sidebar_bg"], foreground=self.palette["accent_gold"])
        style.configure("HeroTitle.TLabel", font=("Georgia", 28, "bold"), background=self.palette["sidebar_bg"], foreground=self.palette["text_light"])
        style.configure("HeroSub.TLabel", font=("Helvetica", 11), background=self.palette["sidebar_bg"], foreground="#D8CDBF")
        style.configure("HeroStatus.TLabel", font=("Helvetica", 11), background=self.palette["sidebar_bg"], foreground=self.palette["text_light"])
        style.configure("HeroPanelTitle.TLabel", font=("Helvetica", 10, "bold"), background="#2A231F", foreground="#D8CDBF")
        style.configure("HeroPanelValue.TLabel", font=("Helvetica", 12, "bold"), background="#2A231F", foreground=self.palette["text_light"])
        style.configure("HeroPanelBody.TLabel", font=("Helvetica", 10), background="#2A231F", foreground="#D8CDBF")
        style.configure("SubHeader.TLabel", font=("Helvetica", 13, "bold"), background=self.palette["content_bg"], foreground=self.palette["text_dark"])
        style.configure("Muted.TLabel", font=("Helvetica", 10), background=self.palette["content_bg"], foreground=self.palette["text_muted"])
        style.configure("SidebarTitle.TLabel", font=("Helvetica", 12, "bold"), background=self.palette["sidebar_card"], foreground=self.palette["text_light"])
        style.configure("SidebarBody.TLabel", font=("Helvetica", 10), background=self.palette["sidebar_card"], foreground="#D8CDBF")
        style.configure("SidebarValue.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["sidebar_bg"], foreground=self.palette["accent_gold"])
        style.configure("SidebarStatus.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["sidebar_bg"], foreground=self.palette["text_light"])
        style.configure("StatusChip.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["accent_gold"], foreground=self.palette["sidebar_bg"])
        style.configure("CardTitle.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["card_bg"], foreground=self.palette["text_muted"])
        style.configure("CardValue.TLabel", font=("Georgia", 22, "bold"), background=self.palette["card_bg"], foreground=self.palette["text_dark"])
        style.configure("CardBody.TLabel", font=("Helvetica", 10), background=self.palette["card_bg"], foreground=self.palette["text_muted"])
        style.configure("SectionTitle.TLabel", font=("Helvetica", 13, "bold"), background=self.palette["card_bg"], foreground=self.palette["text_dark"])
        style.configure("SectionBody.TLabel", font=("Helvetica", 10), background=self.palette["card_bg"], foreground=self.palette["text_muted"])
        style.configure("PanelTitle.TLabel", font=("Helvetica", 12, "bold"), background=self.palette["card_bg"], foreground=self.palette["text_dark"])
        style.configure("PanelBody.TLabel", font=("Helvetica", 10), background=self.palette["card_bg"], foreground=self.palette["text_muted"])
        style.configure("AltPanelTitle.TLabel", font=("Helvetica", 12, "bold"), background=self.palette["card_alt"], foreground=self.palette["text_dark"])
        style.configure("AltPanelBody.TLabel", font=("Helvetica", 10), background=self.palette["card_alt"], foreground=self.palette["text_muted"])
        style.configure("Pill.TLabel", font=("Helvetica", 10, "bold"), background=self.palette["card_alt"], foreground=self.palette["text_dark"])

        style.configure("TButton", font=("Helvetica", 11), padding=(12, 8), borderwidth=0)
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), padding=(14, 9), background=self.palette["accent_gold"], foreground=self.palette["sidebar_bg"])
        style.map("Accent.TButton", background=[("active", "#D8B775"), ("disabled", "#8A7A5B")], foreground=[("disabled", "#E8DDCC")])
        style.configure("Danger.TButton", font=("Helvetica", 11, "bold"), padding=(14, 9), background=self.palette["accent_red"], foreground="#FFF7F5")
        style.map("Danger.TButton", background=[("active", "#C56157"), ("disabled", "#8E5B56")], foreground=[("disabled", "#F0DDDA")])
        style.configure("Soft.TButton", font=("Helvetica", 11), padding=(12, 8), background=self.palette["card_alt"], foreground=self.palette["text_dark"])
        style.map("Soft.TButton", background=[("active", "#E6D9C9"), ("disabled", "#D8CEC1")], foreground=[("disabled", "#9B8E80")])
        style.configure("TabAction.TButton", font=("Helvetica", 10, "bold"), padding=(10, 6), background=self.palette["card_alt"], foreground=self.palette["text_dark"])
        style.configure("Filter.TMenubutton", font=("Helvetica", 11), padding=(10, 7), background="#FFFDFC", foreground=self.palette["text_dark"])
        style.map("Filter.TMenubutton", background=[("active", "#F5EEE4"), ("disabled", "#E2D8CB")])

        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"), background=self.palette["card_alt"], foreground=self.palette["text_dark"], relief="flat")
        style.map("Treeview.Heading", background=[("active", "#E8D9C6")])
        style.configure(
            "Treeview",
            rowheight=30,
            font=("Helvetica", 10),
            background="#FFFDFC",
            foreground=self.palette["text_dark"],
            fieldbackground="#FFFDFC",
            bordercolor=self.palette["line"],
            lightcolor=self.palette["line"],
            darkcolor=self.palette["line"],
        )
        style.map("Treeview", background=[("selected", "#D8C6A0")], foreground=[("selected", self.palette["text_dark"])])

        style.configure("TNotebook", background=self.palette["content_bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab",
            font=("Helvetica", 11, "bold"),
            padding=(18, 10),
            background="#D8CDBF",
            foreground=self.palette["text_dark"],
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.palette["card_bg"]), ("active", "#E6D9C9")],
            foreground=[("selected", self.palette["text_dark"])],
        )

        style.configure("TCombobox", fieldbackground="#FFFDFC", background=self.palette["card_bg"], foreground=self.palette["text_dark"], padding=6)
        style.configure("TEntry", fieldbackground="#FFFDFC", foreground=self.palette["text_dark"], padding=6)
        style.configure("TRadiobutton", background=self.palette["card_bg"], foreground=self.palette["text_dark"], font=("Helvetica", 10))
        style.map("TRadiobutton", background=[("active", self.palette["card_bg"])])
        style.configure("Alt.TRadiobutton", background=self.palette["card_alt"], foreground=self.palette["text_dark"], font=("Helvetica", 10))
        style.map("Alt.TRadiobutton", background=[("active", self.palette["card_alt"])])

    def _build_layout(self) -> None:
        app_shell = ttk.Frame(self.root, style="App.TFrame", padding=18)
        app_shell.pack(fill="both", expand=True)

        sidebar = ttk.Frame(app_shell, style="Sidebar.TFrame", padding=(18, 18, 18, 18))
        sidebar.pack(side="left", fill="y")

        content = ttk.Frame(app_shell, style="Content.TFrame", padding=(18, 0, 0, 0))
        content.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)

        hero = ttk.Frame(content, style="HeroCard.TFrame", padding=20)
        hero.pack(fill="x", pady=(0, 14))

        hero_left = ttk.Frame(hero, style="HeroCard.TFrame")
        hero_left.pack(side="left", fill="both", expand=True, padx=(0, 18))

        ttk.Label(hero_left, text="IMDB MOVIE LAB", style="HeroKicker.TLabel").pack(anchor="w")
        ttk.Label(hero_left, text="Movie Intelligence Desk", style="HeroTitle.TLabel").pack(anchor="w", pady=(6, 2))
        ttk.Label(
            hero_left,
            text="A single workspace for filtering the cleaned dataset, reviewing narrative insights, and previewing publication-ready charts.",
            style="HeroSub.TLabel",
            wraplength=640,
            justify="left",
        ).pack(anchor="w", pady=(0, 14))

        hero_status = ttk.Frame(hero_left, style="HeroCard.TFrame")
        hero_status.pack(anchor="w")
        ttk.Label(hero_status, textvariable=self.status_badge_var, style="StatusChip.TLabel").pack(side="left")
        ttk.Label(hero_status, textvariable=self.status_var, style="HeroStatus.TLabel", wraplength=520, justify="left").pack(side="left", padx=(10, 0))

        hero_right = ttk.Frame(hero, style="HeroPanel.TFrame", padding=16)
        hero_right.pack(side="right", fill="y")
        ttk.Label(hero_right, text="Dataset Scope", style="HeroPanelTitle.TLabel").pack(anchor="w")
        ttk.Label(hero_right, textvariable=self.dataset_scope_var, style="HeroPanelBody.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(6, 14))
        ttk.Label(hero_right, text="Current Focus", style="HeroPanelTitle.TLabel").pack(anchor="w")
        ttk.Label(hero_right, textvariable=self.selection_scope_var, style="HeroPanelValue.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(6, 0))

        summary_frame = ttk.Frame(content, style="Content.TFrame")
        summary_frame.pack(fill="x", pady=(0, 14))
        self._build_summary_card(summary_frame, "Movies", self.summary_vars["movie_count"], "Current filtered titles", 0, self.palette["accent_gold"])
        self._build_summary_card(summary_frame, "Genres", self.summary_vars["genre_count"], "Distinct primary genres", 1, self.palette["accent_red"])
        self._build_summary_card(summary_frame, "Average Rating", self.summary_vars["average_rating"], "Mean vote average", 2, self.palette["accent_olive"])
        self._build_summary_card(summary_frame, "Total Revenue", self.summary_vars["total_revenue"], "Summed box office", 3, self.palette["accent_gold"])

        notebook_shell = ttk.Frame(content, style="Card.TFrame", padding=12)
        notebook_shell.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(notebook_shell)
        self.notebook.pack(fill="both", expand=True)

        self.results_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.results_tab, text="Results")
        insights_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(insights_tab, text="Insights")
        self.charts_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.charts_tab, text="Charts")
        quality_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(quality_tab, text="Data Quality")

        self._build_results_tab(self.results_tab)
        self._build_insights_tab(insights_tab)
        self._build_charts_tab(self.charts_tab)
        self._build_quality_tab(quality_tab)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        hero_card = ttk.Frame(parent, style="Sidebar.TFrame")
        hero_card.pack(fill="x")
        ttk.Label(hero_card, text="IMDB", style="Header.TLabel").pack(anchor="w")
        ttk.Label(hero_card, text="Cinema Dataset Workbench", style="HeroBody.TLabel").pack(anchor="w", pady=(2, 10))

        status_row = ttk.Frame(hero_card, style="Sidebar.TFrame")
        status_row.pack(anchor="w")
        ttk.Label(status_row, textvariable=self.status_badge_var, style="StatusChip.TLabel").pack(side="left")
        ttk.Label(status_row, textvariable=self.status_var, style="SidebarStatus.TLabel", wraplength=200, justify="left").pack(side="left", padx=(10, 0))
        ttk.Label(
            hero_card,
            text="Built for cleaning, exploration, chart review, and quick dataset inspection.",
            style="HeroBody.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(12, 18))

        filters_card = ttk.Frame(parent, style="SidebarCard.TFrame", padding=16)
        filters_card.pack(fill="x", pady=(0, 14))
        ttk.Label(filters_card, text="Filters", style="SidebarTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(filters_card, text="Genre", style="SidebarBody.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.genre_selector = ttk.OptionMenu(filters_card, self.genre_var, "All")
        self.genre_selector.configure(style="Filter.TMenubutton")
        self.genre_selector.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(filters_card, text="Year Range", style="SidebarBody.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.year_from_entry = ttk.Entry(filters_card, textvariable=self.year_from_var, width=10)
        self.year_to_entry = ttk.Entry(filters_card, textvariable=self.year_to_var, width=10)
        self.year_from_entry.grid(row=4, column=0, sticky="ew", pady=(0, 10), padx=(0, 6))
        self.year_to_entry.grid(row=4, column=1, sticky="ew", pady=(0, 10))

        ttk.Label(filters_card, text="Minimum Rating", style="SidebarBody.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.rating_entry = ttk.Entry(filters_card, textvariable=self.rating_var)
        self.rating_entry.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(filters_card, text="Title or Keyword", style="SidebarBody.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.keyword_entry = ttk.Entry(filters_card, textvariable=self.keyword_var)
        self.keyword_entry.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        self.apply_button = ttk.Button(
            filters_card,
            text="Apply Filters",
            command=lambda: self.request_analysis("apply_button"),
            style="Accent.TButton",
        )
        self.apply_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.reset_button = ttk.Button(filters_card, text="Reset Filters", command=self.reset_filters, style="Soft.TButton")
        self.reset_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.refresh_charts_button = ttk.Button(filters_card, text="Refresh Charts", command=self.refresh_charts, style="Danger.TButton")
        self.refresh_charts_button.grid(row=11, column=0, columnspan=2, sticky="ew")
        self.chart_refresh_buttons.append(self.refresh_charts_button)

        filters_card.columnconfigure(0, weight=1)
        filters_card.columnconfigure(1, weight=1)

        note_card = ttk.Frame(parent, style="SidebarCard.TFrame", padding=16)
        note_card.pack(fill="x")
        ttk.Label(note_card, text="Workflow", style="SidebarTitle.TLabel").pack(anchor="w")
        ttk.Label(
            note_card,
            text="1. Set filters\n2. Apply analysis\n3. Refresh charts when you want updated visuals",
            style="SidebarBody.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(8, 10))
        ttk.Label(note_card, textvariable=self.results_info_var, style="SidebarValue.TLabel", wraplength=260, justify="left").pack(anchor="w")
        ttk.Label(note_card, textvariable=self.pending_state_var, style="SidebarBody.TLabel", wraplength=260, justify="left").pack(anchor="w", pady=(10, 0))
        ttk.Label(note_card, textvariable=self.last_action_var, style="SidebarBody.TLabel", wraplength=260, justify="left").pack(anchor="w", pady=(10, 0))

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, style="Card.TFrame", padding=14)
        shell.pack(fill="both", expand=True)

        header = ttk.Frame(shell, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 10))
        header_left = ttk.Frame(header, style="Card.TFrame")
        header_left.pack(side="left")
        ttk.Label(header_left, text="Results Explorer", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(header_left, textvariable=self.results_info_var, style="SectionBody.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(header_left, text="Click a column header to sort. Default order is rating, revenue, then title.", style="SectionBody.TLabel").pack(anchor="w", pady=(2, 0))

        pagination_bar = ttk.Frame(header, style="Card.TFrame")
        pagination_bar.pack(side="right")
        self.prev_button = ttk.Button(pagination_bar, text="Previous", command=lambda: self._change_page(-1), style="TabAction.TButton")
        self.prev_button.pack(side="left", padx=4)
        ttk.Label(pagination_bar, textvariable=self.page_var, style="Pill.TLabel").pack(side="left", padx=6)
        self.next_button = ttk.Button(pagination_bar, text="Next", command=lambda: self._change_page(1), style="TabAction.TButton")
        self.next_button.pack(side="left", padx=4)

        banner = ttk.Frame(shell, style="CardAlt.TFrame", padding=12)
        banner.pack(fill="x", pady=(0, 10))
        ttk.Label(banner, textvariable=self.results_banner_var, style="AltPanelTitle.TLabel", wraplength=1080, justify="left").pack(anchor="w")
        ttk.Label(banner, textvariable=self.applied_filters_var, style="AltPanelBody.TLabel", wraplength=1080, justify="left").pack(anchor="w", pady=(4, 0))
        self.analysis_progress = ttk.Progressbar(banner, mode="indeterminate")
        self.analysis_progress.pack(fill="x", pady=(10, 0))

        table_shell = ttk.Frame(shell, style="Card.TFrame", padding=4)
        table_shell.pack(fill="both", expand=True)
        table_frame = ttk.Frame(table_shell, style="Card.TFrame")
        table_frame.pack(fill="both", expand=True)

        self.results_tree = ttk.Treeview(
            table_frame,
            columns=("id", "title", "genre", "release_date", "year", "rating", "runtime", "budget", "revenue"),
            show="headings",
        )
        for column, label, width, anchor, stretch in (
            ("id", "ID", 90, "center", False),
            ("title", "Title", 360, "w", True),
            ("genre", "Genre", 120, "center", False),
            ("release_date", "Release Date", 120, "center", False),
            ("year", "Year", 90, "center", False),
            ("rating", "Rating", 90, "center", False),
            ("runtime", "Runtime", 95, "center", False),
            ("budget", "Budget", 150, "e", False),
            ("revenue", "Revenue", 160, "e", False),
        ):
            self.results_tree.heading(column, text=label, command=lambda selected=column: self._sort_results_by(selected))
            self.results_tree.column(column, width=width, anchor=anchor, stretch=stretch)

        tree_y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        tree_x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)

        self.results_tree.grid(row=0, column=0, sticky="nsew")
        tree_y_scroll.grid(row=0, column=1, sticky="ns")
        tree_x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.results_tree.tag_configure("even", background="#FFFDFC")
        self.results_tree.tag_configure("odd", background="#F5EEE4")

    def _build_insights_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, style="Card.TFrame", padding=10)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Narrative Insights", style="SectionTitle.TLabel").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(shell, text="The app combines summary findings with the background advanced analysis results.", style="SectionBody.TLabel").pack(anchor="w", padx=6, pady=(4, 10))
        frame = ttk.Frame(shell, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        self.insights_text = tk.Text(
            frame,
            wrap="word",
            font=("Helvetica", 11),
            bg="#FFFDFC",
            fg=self.palette["text_dark"],
            padx=16,
            pady=16,
            relief="flat",
            insertbackground=self.palette["text_dark"],
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=scroll.set)
        self.insights_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, style="Card.TFrame", padding=10)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Cleaning Report", style="SectionTitle.TLabel").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(shell, text="Track the raw source, removed rows, date coverage, and remaining missing values after cleaning.", style="SectionBody.TLabel").pack(anchor="w", padx=6, pady=(4, 10))
        frame = ttk.Frame(shell, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        self.quality_text = tk.Text(
            frame,
            wrap="word",
            font=("Helvetica", 11),
            bg="#FFFDFC",
            fg=self.palette["text_dark"],
            padx=16,
            pady=16,
            relief="flat",
            insertbackground=self.palette["text_dark"],
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.quality_text.yview)
        self.quality_text.configure(yscrollcommand=scroll.set)
        self.quality_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_charts_tab(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent, style="Card.TFrame", padding=12)
        top_bar.pack(fill="x", pady=(0, 8))
        top_left = ttk.Frame(top_bar, style="Card.TFrame")
        top_left.pack(side="left")
        ttk.Label(top_left, text="Chart Studio", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(top_left, textvariable=self.chart_status_var, style="SectionBody.TLabel", wraplength=700, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(
            top_left,
            text="Preview auto-fits the available width and keeps vertical scrolling for detail.",
            style="SectionBody.TLabel",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        tab_refresh = ttk.Button(top_bar, text="Refresh Charts", command=self.refresh_charts, style="Danger.TButton")
        tab_refresh.pack(side="right")
        self.chart_refresh_buttons.append(tab_refresh)

        body = ttk.Panedwindow(parent, orient="horizontal")
        body.pack(fill="both", expand=True)

        left_panel = ttk.Frame(body, style="CardAlt.TFrame", padding=14)
        right_panel = ttk.Frame(body, style="Card.TFrame", padding=14)
        body.add(left_panel, weight=1)
        body.add(right_panel, weight=4)

        ttk.Label(left_panel, text="Chart Gallery", style="AltPanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            left_panel,
            text="Choose a chart to preview. The preview updates after charts are generated for the current selection.",
            style="AltPanelBody.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(6, 12))

        for key, title, _description in self.CHART_OPTIONS:
            radio = ttk.Radiobutton(
                left_panel,
                text=title,
                value=key,
                variable=self.chart_choice_var,
                command=self._on_chart_selection_changed,
                style="Alt.TRadiobutton",
            )
            radio.pack(anchor="w", pady=2)

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(left_panel, text="Selected Chart", style="AltPanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            left_panel,
            textvariable=self.chart_description_var,
            style="AltPanelBody.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(6, 12))
        ttk.Label(
            left_panel,
            text="Tip: refresh charts after applying filters so the preview matches the current movie selection.",
            style="AltPanelBody.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(right_panel, textvariable=self.chart_title_var, style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(right_panel, textvariable=self.chart_info_var, style="PanelBody.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=(4, 2))
        ttk.Label(right_panel, textvariable=self.chart_file_var, style="PanelBody.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=(0, 10))

        preview_shell = ttk.Frame(right_panel, style="Inset.TFrame", padding=10)
        preview_shell.pack(fill="both", expand=True)
        preview_shell.rowconfigure(0, weight=1)
        preview_shell.columnconfigure(0, weight=1)
        self.chart_canvas = tk.Canvas(preview_shell, bg="#FCFAF6", highlightthickness=0)
        self.chart_vertical_scroll = ttk.Scrollbar(preview_shell, orient="vertical", command=self.chart_canvas.yview)
        self.chart_canvas.configure(
            yscrollcommand=self.chart_vertical_scroll.set,
        )
        self.chart_canvas.grid(row=0, column=0, sticky="nsew")
        self.chart_vertical_scroll.grid(row=0, column=1, sticky="ns")

    def _build_summary_card(
        self,
        parent: ttk.Frame,
        title: str,
        variable: tk.StringVar,
        subtitle: str,
        column: int,
        accent: str,
    ) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        accent_bar = tk.Frame(card, bg=accent, height=5)
        accent_bar.pack(fill="x", pady=(0, 12))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=variable, style="CardValue.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(card, text=subtitle, style="CardBody.TLabel").pack(anchor="w")
        parent.columnconfigure(column, weight=1)

    def _bind_events(self) -> None:
        self.root.bind("<Return>", lambda _event: self.request_analysis("return_key"))
        self.root.bind("<KP_Enter>", lambda _event: self.request_analysis("kp_enter"))
        self.chart_canvas.bind("<Configure>", self._on_chart_canvas_configure)
        self.chart_canvas.bind("<MouseWheel>", self._scroll_chart_vertical)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _bind_filter_traces(self) -> None:
        for variable in (self.genre_var, self.year_from_var, self.year_to_var, self.rating_var, self.keyword_var):
            variable.trace_add("write", self._on_filter_input_changed)

    def _on_filter_input_changed(self, *_args: object) -> None:
        self._schedule_filter_feedback_refresh()

    def _schedule_filter_feedback_refresh(self) -> None:
        if self.filter_feedback_job is not None:
            self.root.after_cancel(self.filter_feedback_job)
        self.filter_feedback_job = self.root.after(self.FILTER_FEEDBACK_DEBOUNCE_MS, self._run_scheduled_filter_feedback)

    def _run_scheduled_filter_feedback(self) -> None:
        self.filter_feedback_job = None
        self._refresh_filter_feedback()

    def _populate_genres(self) -> None:
        genres = ["All"]
        if "primary_genre" in self.dataset.columns:
            genres.extend(sorted(self.dataset["primary_genre"].dropna().astype(str).unique().tolist()))
        self.genre_var.set("All")
        self.genre_selector.set_menu("All", *genres)

    def _current_control_snapshot_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._collect_filter_snapshot().items()))

    def _describe_snapshot(self, snapshot_key: tuple[tuple[str, str], ...]) -> str:
        snapshot = dict(snapshot_key)
        parts: list[str] = []
        if snapshot.get("genre") and snapshot["genre"] != "All":
            parts.append(f"Genre: {snapshot['genre']}")
        year_from = snapshot.get("year_from", "")
        year_to = snapshot.get("year_to", "")
        if year_from or year_to:
            if year_from and year_to:
                parts.append(f"Years: {year_from}-{year_to}")
            elif year_from:
                parts.append(f"Year >= {year_from}")
            else:
                parts.append(f"Year <= {year_to}")
        if snapshot.get("min_rating"):
            parts.append(f"Rating >= {snapshot['min_rating']}")
        if snapshot.get("title_keyword"):
            parts.append(f"Search: {snapshot['title_keyword']}")
        return " | ".join(parts) if parts else "All titles"

    def _refresh_filter_feedback(self) -> None:
        if self.filter_feedback_job is not None:
            self.root.after_cancel(self.filter_feedback_job)
            self.filter_feedback_job = None
        current_snapshot = self._current_control_snapshot_key()
        self.filter_dirty = current_snapshot != self.applied_control_snapshot
        current_description = self._describe_snapshot(current_snapshot)
        applied_description = self._describe_snapshot(self.applied_control_snapshot)

        if self.analysis_in_progress:
            self.pending_state_var.set(f"Applying: {current_description}")
        elif self.filter_dirty:
            self.pending_state_var.set(f"Pending changes: {current_description}")
        else:
            self.pending_state_var.set("Filters are synced with the current results. You can still refresh the current selection.")

        self.applied_filters_var.set(f"Applied filters: {applied_description}")
        self._update_action_states()

    def _load_existing_charts(self) -> None:
        existing: dict[str, str] = {}
        for key, filename in self.CHART_FILENAMES.items():
            candidate = self.output_dir / filename
            if candidate.exists():
                existing[key] = str(candidate)
        self.chart_paths = existing
        self._ensure_global_chart_paths()
        self.charts_match_current_filters = self._manifest_matches_current_selection()
        self._clear_chart_preview_cache()
        self._update_chart_status_message()
        self._refresh_chart_panel()

    def _ensure_global_chart_paths(self) -> None:
        global_chart_key = "forecast_backtest_yearly_comparison"
        if global_chart_key in self.chart_paths and Path(self.chart_paths[global_chart_key]).exists():
            return

        chart_paths = create_project_visuals(self.dataset, output_dir=self.output_dir, include_global_forecast=True)
        if global_chart_key in chart_paths:
            self.chart_paths[global_chart_key] = chart_paths[global_chart_key]

    def _parse_filters(self) -> dict[str, object]:
        filters: dict[str, object] = {}
        genre = self.genre_var.get().strip()
        if genre and genre != "All":
            filters["genre"] = genre
        if self.year_from_var.get().strip():
            filters["year_from"] = int(self.year_from_var.get().strip())
        if self.year_to_var.get().strip():
            filters["year_to"] = int(self.year_to_var.get().strip())
        if self.rating_var.get().strip():
            filters["min_rating"] = float(self.rating_var.get().strip())
        if self.keyword_var.get().strip():
            filters["title_keyword"] = self.keyword_var.get().strip()
        return filters

    @staticmethod
    def _default_filter_snapshot() -> tuple[tuple[str, str], ...]:
        return (
            ("genre", "All"),
            ("min_rating", ""),
            ("title_keyword", ""),
            ("year_from", ""),
            ("year_to", ""),
        )

    def _record_user_feedback(self, status_message: str, *, banner_message: str | None = None) -> None:
        if banner_message is not None:
            self.results_banner_var.set(banner_message)
        self._set_status(status_message)
        self.last_action_var.set(f"{status_message} ({datetime.now().strftime('%H:%M:%S')}).")

    def request_analysis(self, source: str = "unknown") -> None:
        filter_snapshot = self._collect_filter_snapshot()
        current_snapshot_key = tuple(sorted(filter_snapshot.items()))

        if self.analysis_in_progress:
            self._record_user_feedback(
                "Analysis already running. Waiting for the current request to finish.",
                banner_message="A filter update is already in progress. Please wait for the current results to finish refreshing.",
            )
            self._log_operation("analysis_request_ignored_busy", source=source)
            return

        try:
            filters = self._parse_filters()
        except ValueError:
            self._log_operation("analysis_request_parse_error", source=source)
            messagebox.showerror("Input Error", "Year must be an integer and rating must be numeric.")
            return

        self._log_operation(
            "analysis_requested",
            source=source,
            widget_state=str(self.apply_button.state()),
            current_filters=filter_snapshot,
        )
        refresh_current = not self.filter_dirty and source != "reset_filters"
        self._start_analysis(filters, source=source, refresh_current=refresh_current)

    def run_analysis(self) -> None:
        self.request_analysis("legacy_run_analysis")

    def reset_filters(self) -> None:
        self._log_operation("reset_filters_requested")
        if self.analysis_in_progress:
            self._record_user_feedback(
                "Analysis already running. Wait for it to finish before resetting filters.",
                banner_message="Filters cannot be reset while a result update is still running.",
            )
            self._log_operation("reset_filters_ignored_busy")
            return

        default_snapshot = self._default_filter_snapshot()
        current_snapshot = self._current_control_snapshot_key()
        if current_snapshot == default_snapshot and self.applied_control_snapshot == default_snapshot:
            self._record_user_feedback(
                "Filters are already cleared.",
                banner_message="Nothing changed: the current results already show the full dataset.",
            )
            self._log_operation("reset_filters_noop")
            return

        self.genre_var.set("All")
        self.year_from_var.set("")
        self.year_to_var.set("")
        self.rating_var.set("")
        self.keyword_var.set("")
        if self.applied_control_snapshot == default_snapshot:
            self._refresh_filter_feedback()
            self._record_user_feedback(
                "Pending filter changes cleared.",
                banner_message="Pending filter edits were cleared. The current results were already up to date.",
            )
            self._log_operation("reset_filters_cleared_pending")
            return

        self.request_analysis("reset_filters")

    def refresh_charts(self) -> None:
        self._log_operation("chart_refresh_requested", rows=int(len(self.current_results)))
        if self.chart_generation_in_progress:
            self._record_user_feedback(
                "Chart refresh already running.",
                banner_message="Charts are already being regenerated for the current selection.",
            )
            self._log_operation("chart_refresh_ignored_busy")
            return

        if self.analysis_in_progress:
            self._record_user_feedback(
                "Wait for the current filter update before refreshing charts.",
                banner_message="Charts can be refreshed as soon as the current filter update finishes.",
            )
            self._log_operation("chart_refresh_ignored_analysis_busy")
            return

        if self.current_results.empty:
            self.notebook.select(self.results_tab)
            self._record_user_feedback(
                "No rows match the current filters, so charts cannot be generated.",
                banner_message="Charts are unavailable because the current movie selection is empty.",
            )
            self._log_operation("chart_refresh_empty_selection")
            return

        self.chart_request_token += 1
        request_token = self.chart_request_token
        df_snapshot = self.current_results
        self.chart_generation_in_progress = True
        self._set_status("Generating charts for the current selection...")
        self._update_chart_status_message()
        self._update_action_states()
        self.notebook.select(self.charts_tab)

        worker = threading.Thread(
            target=self._chart_generation_worker,
            args=(request_token, df_snapshot),
            daemon=True,
        )
        worker.start()

    def _start_analysis(self, filters: dict[str, object], source: str = "unknown", refresh_current: bool = False) -> None:
        self.analysis_request_id += 1
        request_id = self.analysis_request_id
        request_snapshot_key = self._current_control_snapshot_key()
        self.analysis_request_snapshots[request_id] = request_snapshot_key
        cache_key = self._filters_cache_key(filters)
        self.current_filter_key = cache_key
        self.analysis_in_progress = True
        self.chart_request_token += 1
        self.chart_generation_in_progress = False
        self.charts_match_current_filters = False
        self.notebook.select(self.results_tab)
        banner_message = "Refreshing the current result set..." if refresh_current else "Applying filters and refreshing the result set..."
        status_message = "Refreshing the current results..." if refresh_current else "Applying filters and preparing results..."
        self.results_banner_var.set(banner_message)
        self.applied_filters_var.set(f"Requested filters: {self._describe_snapshot(request_snapshot_key)}")
        self._set_status(status_message)
        self._update_chart_status_message()
        self.analysis_progress.start(10)
        self._refresh_filter_feedback()
        self.root.update_idletasks()
        self.last_action_var.set(f"Filter request received at {datetime.now().strftime('%H:%M:%S')}.")
        self._log_operation(
            "analysis_started",
            request_id=request_id,
            source=source,
            filters=filters,
            filter_key=list(cache_key),
            cache_hit=cache_key in self.analysis_cache,
            refresh_current=refresh_current,
        )

        started_at = perf_counter()
        if cache_key in self.analysis_cache:
            analysis_result = self.analysis_cache[cache_key]
            self._log_operation(
                "analysis_worker_completed",
                request_id=request_id,
                duration_ms=0.0,
                rows=int(len(analysis_result["filtered_df"])),
                execution="sync_cache",
            )
            self._apply_analysis_result(request_id, analysis_result, cache_key)
            return

        try:
            analysis_result = analyze(self.dataset, filters)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self._handle_background_error("Analysis failed", str(exc))
            return
        self._log_operation(
            "analysis_worker_completed",
            request_id=request_id,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            rows=int(len(analysis_result["filtered_df"])),
            execution="sync",
        )
        self._apply_analysis_result(request_id, analysis_result, cache_key)

    def _apply_analysis_result(
        self,
        request_id: int,
        analysis_result: dict[str, object],
        cache_key: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        if request_id != self.analysis_request_id:
            self._log_operation("analysis_result_discarded", request_id=request_id, active_request_id=self.analysis_request_id)
            return

        started_at = perf_counter()
        previous_rows = int(len(self.current_results))
        previous_identity = set(self.current_result_identity)
        request_snapshot_key = self.analysis_request_snapshots.pop(request_id, self.applied_control_snapshot)
        if cache_key is not None:
            self.analysis_cache[cache_key] = analysis_result

        self.current_analysis = analysis_result
        self.current_results = analysis_result["filtered_df"]
        self.current_result_identity = self._build_result_identity(self.current_results)
        self._reset_sort_cache()
        self.current_advanced = {}
        self.current_page = 0
        self.analysis_in_progress = False
        self.applied_control_snapshot = request_snapshot_key
        self.charts_match_current_filters = False
        self._refresh_view(include_advanced=False)
        self._update_chart_status_message()
        self.analysis_progress.stop()
        self._refresh_filter_feedback()
        added_count, removed_count = self._selection_change_counts(previous_identity, self.current_result_identity)

        if self.current_results.empty:
            self.results_banner_var.set("No rows matched the requested filters.")
            self.applied_filters_var.set(f"Applied filters: {self._describe_snapshot(request_snapshot_key)}")
            self._set_status("No matching rows for the current filters.")
            self.last_action_var.set(f"No matching results. Updated at {datetime.now().strftime('%H:%M:%S')}.")
            self._log_operation(
                "analysis_applied",
                request_id=request_id,
                rows=0,
                added_count=added_count,
                removed_count=removed_count,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
            return

        row_delta = int(len(self.current_results)) - previous_rows
        if added_count == 0 and removed_count == 0:
            delta_prefix = "selection unchanged"
        elif previous_rows != int(len(self.current_results)):
            delta_prefix = f"{row_delta:+,} rows vs previous selection; {added_count:,} added, {removed_count:,} removed"
        else:
            delta_prefix = f"same row count; {added_count:,} added, {removed_count:,} removed"
        self.results_banner_var.set(
            f"Results updated: {len(self.current_results):,} rows ({delta_prefix})."
        )
        self.applied_filters_var.set(f"Applied filters: {self._describe_snapshot(request_snapshot_key)}")
        self._log_operation(
            "analysis_applied",
            request_id=request_id,
            rows=int(len(self.current_results)),
            added_count=added_count,
            removed_count=removed_count,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )
        self.last_action_var.set(
            f"Base results updated: {len(self.current_results):,} rows at {datetime.now().strftime('%H:%M:%S')}."
        )
        self._set_status(f"Results updated: {len(self.current_results):,} rows ready.")
        self._start_advanced_analysis(self.current_results, request_id)

    def _start_advanced_analysis(self, df: pd.DataFrame, request_id: int) -> None:
        if df.empty:
            self.current_advanced = {}
            self._render_insights(include_advanced=True)
            return

        cache_key = self.current_filter_key
        if cache_key in self.advanced_cache:
            self._apply_advanced_result(request_id, self.advanced_cache[cache_key])
            return

        worker = threading.Thread(target=self._advanced_worker, args=(request_id, df, cache_key), daemon=True)
        worker.start()

    def _advanced_worker(self, request_id: int, df: pd.DataFrame, cache_key: tuple[tuple[str, object], ...]) -> None:
        started_at = perf_counter()
        try:
            advanced_result = get_comprehensive_analysis(df)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.root.after(0, self._handle_background_error, "Advanced analysis failed", str(exc))
            return
        self._log_operation(
            "advanced_worker_completed",
            request_id=request_id,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )
        self.root.after(0, self._apply_advanced_result, request_id, advanced_result, cache_key)

    def _apply_advanced_result(
        self,
        request_id: int,
        advanced_result: dict[str, object],
        cache_key: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        if request_id != self.analysis_request_id:
            self._log_operation("advanced_result_discarded", request_id=request_id, active_request_id=self.analysis_request_id)
            return

        if cache_key is not None:
            self.advanced_cache[cache_key] = advanced_result

        self.current_advanced = advanced_result
        self._render_insights(include_advanced=True)
        self._log_operation("advanced_applied", request_id=request_id)

    def _chart_generation_worker(self, request_token: int, df: pd.DataFrame) -> None:
        try:
            chart_paths = create_project_visuals(df, output_dir=self.output_dir, include_global_forecast=False)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.root.after(0, self._finish_chart_error, request_token, str(exc))
            return
        self.root.after(0, self._finish_chart_generation, request_token, chart_paths)

    def _finish_chart_generation(self, request_token: int, chart_paths: dict[str, str]) -> None:
        if request_token != self.chart_request_token:
            return

        self.chart_generation_in_progress = False
        if not chart_paths:
            self.charts_match_current_filters = False
            self._update_chart_status_message()
            self._update_action_states()
            self._record_user_feedback(
                "No chart data available for the current selection.",
                banner_message="The current selection does not contain enough data to draw charts.",
            )
            return

        self.chart_paths = chart_paths
        self._clear_chart_preview_cache()
        self.charts_match_current_filters = True
        self._write_chart_manifest()
        if self.chart_choice_var.get() not in self.chart_paths:
            self.chart_choice_var.set(next(iter(self.chart_paths)))

        self._update_chart_status_message()
        self._refresh_chart_panel()
        self._update_action_states()
        self._record_user_feedback(
            "Charts refreshed and displayed in the Charts tab.",
            banner_message="Chart files were rebuilt for the current selection and the preview has been updated.",
        )
        self.notebook.select(self.charts_tab)

    def _finish_chart_error(self, request_token: int, message: str) -> None:
        if request_token != self.chart_request_token:
            return
        self.chart_generation_in_progress = False
        self._update_chart_status_message()
        self._update_action_states()
        self._record_user_feedback(
            "Chart generation failed. Check the error dialog for details.",
            banner_message="Chart generation failed before the preview could be updated.",
        )
        messagebox.showerror("Chart Generation Failed", message)

    def _handle_background_error(self, title: str, message: str) -> None:
        self.analysis_in_progress = False
        self.chart_generation_in_progress = False
        self.analysis_progress.stop()
        self._refresh_filter_feedback()
        self._set_status(f"{title}. Check the error dialog for details.")
        messagebox.showerror(title, message)

    def _refresh_view(self, include_advanced: bool) -> None:
        started_at = perf_counter()
        overview = self.current_analysis["overview"]
        self.summary_vars["movie_count"].set(f"{overview['movie_count']:,}")
        self.summary_vars["genre_count"].set(f"{overview['genre_count']:,}")
        self.summary_vars["average_rating"].set(f"{overview['average_rating']:.2f}")
        self.summary_vars["total_revenue"].set(f"${overview['total_revenue']:,.0f}")
        self._update_hero_summary()
        self._rebuild_sorted_results()
        self._populate_results_table()
        self._render_insights(include_advanced=include_advanced)
        self._refresh_chart_panel()
        self._log_operation(
            "view_refreshed",
            include_advanced=include_advanced,
            rows=int(len(self.current_results)),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )

    def _update_hero_summary(self) -> None:
        source_name = Path(str(self.quality_report["source_file"])).name
        cleaned_date_range = self.quality_report["cleaned_date_range"]
        dataset_cutoff = self.quality_report.get("dataset_cutoff_date")
        self.dataset_scope_var.set(
            f"{self.quality_report['clean_rows']:,} cleaned titles | cleaned range {cleaned_date_range['start']} to "
            f"{cleaned_date_range['end']} | cutoff {dataset_cutoff} | source {source_name}"
        )

        if self.current_results.empty:
            self.selection_scope_var.set("0 titles in focus | widen the filters to restore a working selection")
            return

        years = self.current_results["year"].dropna() if "year" in self.current_results.columns else pd.Series(dtype=float)
        year_span = (
            f"{int(years.min())}-{int(years.max())}"
            if not years.empty
            else "Year coverage unavailable"
        )
        overview = self.current_analysis["overview"]
        self.selection_scope_var.set(
            f"{len(self.current_results):,} titles in focus | {overview['genre_count']:,} genres | avg rating {overview['average_rating']:.2f} | {year_span}"
        )

    def _rebuild_sorted_results(self) -> None:
        if self.current_results.empty:
            self.current_sorted_results = self.current_results.copy()
            self.current_display_rows = []
            self.total_pages = 1
            self.current_page = 0
            self._update_results_headings()
            return

        cache_key = tuple(self.sort_state)
        cached = self.sorted_results_cache.get(cache_key)
        if cached is None:
            cached = self._sort_dataframe(self.current_results).reset_index(drop=True)
            self.sorted_results_cache[cache_key] = cached
        self.current_sorted_results = cached
        self.current_display_rows = self._build_display_rows(self.current_sorted_results)
        self.total_pages = max(1, math.ceil(len(self.current_sorted_results) / self.PAGE_SIZE))
        self.current_page = min(self.current_page, self.total_pages - 1)
        self._update_results_headings()

    def _sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sort_columns = [column for column, _ascending in self.sort_state if column in df.columns]
        if not sort_columns:
            return df.reset_index(drop=True)

        ascending = [ascending for column, ascending in self.sort_state if column in df.columns]
        return df.sort_values(sort_columns, ascending=ascending, na_position="last")

    def _sort_results_by(self, tree_column: str) -> None:
        dataset_column = self.RESULT_COLUMN_MAP.get(tree_column)
        if not dataset_column:
            return

        existing_index = next((index for index, (column, _ascending) in enumerate(self.sort_state) if column == dataset_column), None)
        if existing_index is None:
            ascending = dataset_column in {"title", "primary_genre", "release_date"}
        else:
            ascending = not self.sort_state[existing_index][1]

        remaining = [(column, direction) for column, direction in self.sort_state if column != dataset_column]
        self.sort_state = [(dataset_column, ascending), *remaining]
        self.current_page = 0
        self._rebuild_sorted_results()
        self._populate_results_table()

        direction_label = "ascending" if ascending else "descending"
        readable_label = tree_column.replace("_", " ").title()
        self._record_user_feedback(f"Sorted by {readable_label} ({direction_label}).")
        self._log_operation("results_sorted", column=tree_column, dataset_column=dataset_column, ascending=ascending)

    def _update_results_headings(self) -> None:
        primary_sort = self.sort_state[0] if self.sort_state else None
        for tree_column, base_label in (
            ("id", "ID"),
            ("title", "Title"),
            ("genre", "Genre"),
            ("release_date", "Release Date"),
            ("year", "Year"),
            ("rating", "Rating"),
            ("runtime", "Runtime"),
            ("budget", "Budget"),
            ("revenue", "Revenue"),
        ):
            dataset_column = self.RESULT_COLUMN_MAP[tree_column]
            if primary_sort and primary_sort[0] == dataset_column:
                arrow = "↑" if primary_sort[1] else "↓"
                label = f"{base_label} {arrow}"
            else:
                label = base_label
            self.results_tree.heading(tree_column, text=label, command=lambda selected=tree_column: self._sort_results_by(selected))

    def _build_display_rows(self, df: pd.DataFrame) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
        if df.empty:
            return []

        display_columns = [
            "id",
            "title",
            "primary_genre",
            "release_date",
            "year",
            "vote_average",
            "runtime",
            "budget",
            "revenue",
        ]
        working = df.reindex(columns=display_columns)
        rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        for id_value, title, genre, release_date, year, rating, runtime, budget, revenue in working.itertuples(index=False, name=None):
            rows.append(
                (
                    self._format_number(id_value),
                    "" if pd.isna(title) else str(title),
                    "" if pd.isna(genre) else str(genre),
                    self._format_date(release_date),
                    self._format_number(year),
                    self._format_float(rating),
                    self._format_number(runtime),
                    self._format_currency(budget),
                    self._format_currency(revenue),
                )
            )
        return rows

    def _ensure_results_row_items(self) -> None:
        while len(self.results_row_items) < self.PAGE_SIZE:
            index = len(self.results_row_items)
            item_id = self.results_tree.insert(
                "",
                "end",
                values=("", "", "", "", "", "", "", "", ""),
                tags=("even" if index % 2 == 0 else "odd",),
            )
            self.results_row_items.append(item_id)

    def _populate_results_table(self) -> None:
        started_at = perf_counter()
        self._ensure_results_row_items()

        total_rows = len(self.current_sorted_results)
        if total_rows == 0:
            if self.results_row_items:
                self.results_tree.detach(*self.results_row_items)
            self.results_info_var.set("No matching movies")
            self.page_var.set("Page 0 / 0")
            self.prev_button.state(["disabled"])
            self.next_button.state(["disabled"])
            self._log_operation("results_table_populated", rows=0, duration_ms=round((perf_counter() - started_at) * 1000, 2))
            return

        start = self.current_page * self.PAGE_SIZE
        end = min(start + self.PAGE_SIZE, total_rows)
        page_rows = self.current_display_rows[start:end]

        for display_index, values in enumerate(page_rows):
            item_id = self.results_row_items[display_index]
            if item_id not in self.results_tree.get_children(""):
                self.results_tree.reattach(item_id, "", display_index)
            self.results_tree.item(item_id, values=values, tags=("even" if display_index % 2 == 0 else "odd",))
            self.results_tree.move(item_id, "", display_index)

        hidden_items = self.results_row_items[len(page_rows):]
        if hidden_items:
            self.results_tree.detach(*hidden_items)

        self.results_info_var.set(f"Showing {start + 1:,}-{end:,} of {total_rows:,} movies")
        self.page_var.set(f"Page {self.current_page + 1} / {self.total_pages}")
        if self.current_page <= 0:
            self.prev_button.state(["disabled"])
        else:
            self.prev_button.state(["!disabled"])
        if self.current_page >= self.total_pages - 1:
            self.next_button.state(["disabled"])
        else:
            self.next_button.state(["!disabled"])
        self._log_operation(
            "results_table_populated",
            visible_rows=int(len(page_rows)),
            total_rows=int(total_rows),
            page=int(self.current_page + 1),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )

    def _reset_sort_cache(self) -> None:
        self.sorted_results_cache = {}

    def _change_page(self, step: int) -> None:
        if self.current_sorted_results.empty:
            self._record_user_feedback("No pages are available for the current result set.")
            self._log_operation("page_change_empty", step=step)
            return
        new_page = self.current_page + step
        if new_page < 0 or new_page >= self.total_pages:
            edge_message = "Already on the first page." if step < 0 else "Already on the last page."
            self._record_user_feedback(edge_message)
            self._log_operation("page_change_boundary", step=step, page=int(self.current_page + 1), total_pages=int(self.total_pages))
            return
        self.current_page = new_page
        self._populate_results_table()
        self._record_user_feedback(f"Showing page {self.current_page + 1} of {self.total_pages}.")
        self._log_operation("page_changed", step=step, page=int(self.current_page + 1), total_pages=int(self.total_pages))

    def _render_insights(self, include_advanced: bool) -> None:
        lines = [
            "Current Selection",
            f"Rows after filtering: {self.current_analysis['meta']['filtered_rows']:,}",
            f"Genres covered: {self.current_analysis['meta']['genre_categories']}",
            f"Years covered: {self.current_analysis['meta']['years_covered']}",
            "",
            "Key Insights",
        ]
        for insight in self.current_analysis["insights"]:
            lines.append(f"- {insight}")

        lines.extend(["", "Advanced Analysis"])
        if not include_advanced:
            lines.append("- Calculating advanced insights...")
        else:
            advanced_lines_added = False
            production_trend = self.current_advanced.get("production_trend", {})
            if production_trend and "error" not in production_trend:
                lines.append(
                    f"- Production trend ({production_trend['period']}): {production_trend['trend_direction']} with peak output in {production_trend['peak_year']}."
                )
                advanced_lines_added = True

            budget_revenue = self.current_advanced.get("budget_revenue_correlation", {})
            if budget_revenue and "error" not in budget_revenue:
                lines.append(
                    f"- Budget vs revenue correlation: {budget_revenue['correlation']} ({budget_revenue['correlation_strength']})."
                )
                advanced_lines_added = True

            decade_comparison = self.current_advanced.get("decade_comparison", {})
            if decade_comparison and "error" not in decade_comparison:
                lines.append(
                    f"- Most productive decade: {decade_comparison['most_productive_decade']}s; highest rated decade: {decade_comparison['highest_rated_decade']}s."
                )
                advanced_lines_added = True

            if not advanced_lines_added:
                lines.append("- No advanced analysis available for the current selection.")

        self._replace_text(self.insights_text, "\n".join(lines))

    def _render_quality_report(self) -> None:
        report = self.quality_report
        data_origin = report.get("data_origin", "full_cleaning")
        source_label = "cleaned dataset cache" if data_origin == "cleaned_cache" else "cleaning pipeline"
        lines = [
            "Dataset Cleaning and Validation",
            f"Source file: {report['source_file']}",
            f"Load mode: {source_label}",
            f"Rows loaded: {report['source_rows']:,}",
            f"Rows after cleaning: {report['clean_rows']:,}",
            f"Duplicates removed: {report['duplicates_removed']:,}",
            f"Source ID duplicates removed: {report.get('source_duplicates_removed', 0):,}",
            f"Title/year duplicates removed: {report.get('title_duplicates_removed', 0):,}",
            f"Title collisions disambiguated: {report.get('title_collisions_disambiguated', 0):,}",
            f"Rows removed for missing title: {report['rows_removed_missing_title']:,}",
            f"Rows removed for invalid release date: {report['rows_removed_invalid_release_date']:,}",
            f"Rows removed after dataset cutoff ({report['dataset_cutoff_date']}): {report['rows_removed_after_dataset_cutoff']:,}",
            f"Missing text fields filled: {report.get('text_fields_filled', 0):,}",
            f"Raw date range: {report['source_date_range']['start']} to {report['source_date_range']['end']}",
            f"Cleaned date range: {report['cleaned_date_range']['start']} to {report['cleaned_date_range']['end']}",
            "",
            "Missing Values After Cleaning",
        ]

        for column, value in report["missing_values_after_cleaning"].items():
            lines.append(f"- {column}: {value}")

        lines.extend(["", "Numeric Summary"])
        for column, stats in report["numeric_summary"].items():
            lines.append(f"- {column}: min {stats['min']}, median {stats['median']}, max {stats['max']}")

        self._replace_text(self.quality_text, "\n".join(lines))

    def _update_chart_status_message(self) -> None:
        if self.chart_choice_var.get() in self.GLOBAL_CHART_KEYS:
            if self.chart_paths.get(self.chart_choice_var.get()) and Path(self.chart_paths[self.chart_choice_var.get()]).exists():
                self.chart_status_var.set("This yearly forecast chart is fixed to the full cleaned dataset and does not require refreshing.")
            else:
                self.chart_status_var.set("Preparing the fixed yearly forecast chart for the full cleaned dataset.")
            return

        if self.chart_generation_in_progress:
            self.chart_status_var.set(f"Generating charts for {len(self.current_results):,} movies...")
            return

        if self.current_results.empty:
            self.chart_status_var.set("No charts available because the current filters returned zero movies.")
            return

        if not self.chart_paths:
            self.chart_status_var.set("No charts generated yet. Click Refresh Charts to build them for the current selection.")
            return

        if self.charts_match_current_filters:
            self.chart_status_var.set(f"Charts are up to date for the current selection ({len(self.current_results):,} movies).")
            return

        self.chart_status_var.set("Previewing the last generated charts. Refresh Charts to sync them with the current filters.")

    def _on_chart_selection_changed(self) -> None:
        self._refresh_chart_panel()

    def _refresh_chart_panel(self) -> None:
        self._update_chart_metadata()
        if self._is_charts_tab_active():
            self._schedule_chart_render()

    def _current_chart_signature(self) -> dict[str, object]:
        return {
            "filter_key": list(self.current_filter_key),
            "row_count": int(len(self.current_results)),
        }

    def _manifest_matches_current_selection(self) -> bool:
        if not self.chart_manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.chart_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        signature = self._current_chart_signature()
        manifest_filter_key = tuple(tuple(item) for item in manifest.get("filter_key", []))
        return (
            manifest_filter_key == tuple(signature["filter_key"])
            and int(manifest.get("row_count", -1)) == signature["row_count"]
        )

    def _write_chart_manifest(self) -> None:
        manifest = self._current_chart_signature()
        manifest["chart_keys"] = sorted(self.chart_paths.keys())
        manifest["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chart_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.chart_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _on_tab_changed(self, _event: tk.Event) -> None:
        if self._is_charts_tab_active():
            self._refresh_chart_panel()
            return

    def _is_charts_tab_active(self) -> bool:
        return self.notebook.select() == str(self.charts_tab)

    @staticmethod
    def _filters_cache_key(filters: dict[str, object]) -> tuple[tuple[str, object], ...]:
        normalized: list[tuple[str, object]] = []
        for key, value in sorted(filters.items()):
            if isinstance(value, float):
                normalized.append((key, round(value, 4)))
            else:
                normalized.append((key, value))
        return tuple(normalized)

    def _on_chart_canvas_configure(self, event: tk.Event) -> None:
        width = int(getattr(event, "width", 0))
        if width <= 10:
            return
        if abs(width - self.chart_canvas_width) < 4 and self.current_chart_photo is not None:
            return
        self.chart_canvas_width = width
        self._schedule_chart_render()

    def _update_chart_metadata(self) -> None:
        key = self.chart_choice_var.get()
        title = next((option_title for option_key, option_title, _ in self.CHART_OPTIONS if option_key == key), key)
        description = next((option_desc for option_key, _, option_desc in self.CHART_OPTIONS if option_key == key), "")
        self.chart_title_var.set(title)
        self.chart_description_var.set(description)

        selection_count = len(self.current_results)
        if key in self.GLOBAL_CHART_KEYS:
            status_note = "fixed to the full cleaned dataset"
        elif self.charts_match_current_filters:
            status_note = "synced with the current filters"
        elif self.chart_paths:
            status_note = "may be from an earlier selection"
        else:
            status_note = "not generated yet"
        self.chart_info_var.set(f"Current selection: {selection_count:,} movies. This chart is {status_note}.")

        path = self.chart_paths.get(key)
        if path and Path(path).exists():
            modified = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.chart_file_var.set(f"Saved file: {path} • Last updated: {modified}")
        else:
            self.chart_file_var.set("This chart is not available yet. Refresh charts to generate it.")

    def _schedule_chart_render(self) -> None:
        if self.chart_render_job is not None:
            self.root.after_cancel(self.chart_render_job)
        self.chart_render_job = self.root.after(40, self._render_selected_chart)

    def _render_selected_chart(self) -> None:
        self.chart_render_job = None
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        if width <= 10 or height <= 10:
            self._schedule_chart_render()
            return

        self.chart_canvas.delete("all")
        self.chart_canvas_image_id = None
        key = self.chart_choice_var.get()
        path = self.chart_paths.get(key)

        if key not in self.GLOBAL_CHART_KEYS and not self.charts_match_current_filters:
            self.current_chart_photo = None
            self._draw_chart_placeholder(
                "These charts are not synced with the current filters.\nClick 'Refresh Charts' to generate the correct preview."
            )
            return

        if not path or not Path(path).exists():
            self.current_chart_photo = None
            self._draw_chart_placeholder(
                "This chart has not been generated yet.\nClick 'Refresh Charts' to create chart previews for the current selection."
            )
            return

        try:
            render_width = max(width - 24, 320)
            cache_key = (key, render_width)
            if cache_key in self.chart_preview_cache:
                self.current_chart_photo, new_size = self.chart_preview_cache[cache_key]
            else:
                self.current_chart_photo, new_size = self._create_chart_preview(path, render_width)
                self.chart_preview_cache[cache_key] = (self.current_chart_photo, new_size)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.current_chart_photo = None
            self._draw_chart_placeholder(f"Unable to load chart preview.\n{exc}")
            return

        x_pos = max((width - new_size[0]) // 2, 12)
        y_pos = 12 if new_size[1] > height else max((height - new_size[1]) // 2, 12)
        self.chart_canvas_image_id = self.chart_canvas.create_image(x_pos, y_pos, image=self.current_chart_photo, anchor="nw")
        self.chart_canvas.configure(scrollregion=(0, 0, width, max(new_size[1] + 24, height)))
        self.chart_canvas.yview_moveto(0)

    def _draw_chart_placeholder(self, message: str) -> None:
        self.chart_canvas_image_id = None
        width = max(self.chart_canvas.winfo_width(), 300)
        height = max(self.chart_canvas.winfo_height(), 220)
        self.chart_canvas.configure(scrollregion=(0, 0, width, height))
        self.chart_canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            fill="#6A5A4A",
            font=("Helvetica", 14),
            width=max(width - 80, 220),
            justify="center",
        )

    def _create_chart_preview(self, path: str, render_width: int) -> tuple[ImageTk.PhotoImage, tuple[int, int]]:
        with Image.open(path) as image:
            scale = min(render_width / image.width, 1.0)
            new_size = (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            )
            if new_size == image.size:
                preview = image.copy()
            else:
                preview = image.resize(new_size, RESAMPLE, reducing_gap=3.0)
        return ImageTk.PhotoImage(preview), new_size

    def _clear_chart_preview_cache(self) -> None:
        self.chart_preview_cache.clear()
        self.current_chart_photo = None

    def _scroll_chart_vertical(self, event: tk.Event) -> str:
        step = -1 if event.delta > 0 else 1
        self.chart_canvas.yview_scroll(step, "units")
        return "break"

    def _update_action_states(self) -> None:
        current_snapshot = dict(self._current_control_snapshot_key())
        has_non_default_controls = any(
            value and not (key == "genre" and value == "All")
            for key, value in current_snapshot.items()
        )
        applied_has_non_default = any(
            value and not (key == "genre" and value == "All")
            for key, value in dict(self.applied_control_snapshot).items()
        )

        apply_disabled = False
        reset_disabled = False
        chart_disabled = False
        self.apply_button.state(["!disabled"])
        self.reset_button.state(["!disabled"])
        if self.analysis_in_progress:
            button_text = "Applying..."
        elif self.filter_dirty:
            button_text = "Apply Pending Filters"
        else:
            button_text = "Refresh Current Results"
        self.apply_button.configure(text=button_text)

        for button in self.chart_refresh_buttons:
            button.state(["!disabled"])
            if self.chart_generation_in_progress:
                button.configure(text="Refreshing Charts...")
            elif self.chart_choice_var.get() in self.GLOBAL_CHART_KEYS:
                button.configure(text="Refresh Filtered Charts")
            elif self.charts_match_current_filters:
                button.configure(text="Rebuild Charts")
            else:
                button.configure(text="Refresh Charts")

        if self.analysis_in_progress:
            reset_text = "Reset Filters"
        elif has_non_default_controls and not applied_has_non_default:
            reset_text = "Clear Pending Changes"
        elif not has_non_default_controls and not applied_has_non_default:
            reset_text = "Filters Cleared"
        else:
            reset_text = "Reset Filters"
        self.reset_button.configure(text=reset_text)
        self.root.configure(cursor="watch" if (self.analysis_in_progress or self.chart_generation_in_progress) else "")
        action_state_snapshot = (apply_disabled, reset_disabled, chart_disabled)
        if action_state_snapshot != self.last_action_state_log_snapshot:
            self.last_action_state_log_snapshot = action_state_snapshot
            self._log_operation(
                "action_states_updated",
                apply_disabled=apply_disabled,
                reset_disabled=reset_disabled,
                chart_disabled=chart_disabled,
            )

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        normalized = message.lower()
        if any(keyword in normalized for keyword in ("preparing", "applying", "calculating", "generating", "refreshing", "running")):
            self.status_badge_var.set("WORKING")
        elif "failed" in normalized or "error" in normalized:
            self.status_badge_var.set("ERROR")
        elif "no matching" in normalized or "no charts" in normalized:
            self.status_badge_var.set("EMPTY")
        else:
            self.status_badge_var.set("READY")
        self._log_operation("status_updated", message=message, badge=self.status_badge_var.get())

    def _collect_filter_snapshot(self) -> dict[str, str]:
        return {
            "genre": self.genre_var.get().strip(),
            "year_from": self.year_from_var.get().strip(),
            "year_to": self.year_to_var.get().strip(),
            "min_rating": self.rating_var.get().strip(),
            "title_keyword": self.keyword_var.get().strip(),
        }

    @staticmethod
    def _selection_change_counts(previous_identity: set[object], current_identity: set[object]) -> tuple[int, int]:
        return len(current_identity - previous_identity), len(previous_identity - current_identity)

    @staticmethod
    def _normalize_identity_value(value: object) -> object:
        if pd.isna(value):
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return value

    def _build_result_identity(self, df: pd.DataFrame) -> set[object]:
        if df.empty:
            return set()

        if "id" in df.columns:
            ids = pd.to_numeric(df["id"], errors="coerce")
            valid_ids = ids.dropna()
            if len(valid_ids) == len(df) and int(valid_ids.nunique(dropna=True)) == len(df):
                return {int(value) for value in valid_ids.tolist()}

        identity: set[object] = set()
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            identity.add(
                (
                    self._normalize_identity_value(row_dict.get("id")),
                    str(row_dict.get("title", "")).strip().lower(),
                    self._normalize_identity_value(row_dict.get("release_date")),
                    self._normalize_identity_value(row_dict.get("year")),
                )
            )
        return identity

    def _log_operation(self, event: str, **details: object) -> None:
        try:
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "event": event,
                "thread": threading.current_thread().name,
                "analysis_request_id": int(self.analysis_request_id),
                "analysis_in_progress": bool(self.analysis_in_progress),
                "chart_generation_in_progress": bool(self.chart_generation_in_progress),
            }
            payload.update(details)
            self.log_queue.put(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except OSError:
            return

    def _log_writer_worker(self) -> None:
        while True:
            line = self.log_queue.get()
            if line is None:
                break

            batch = [line]
            stop_requested = False
            while True:
                try:
                    queued_line = self.log_queue.get_nowait()
                except queue.Empty:
                    break
                if queued_line is None:
                    stop_requested = True
                    break
                batch.append(queued_line)

            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                with self.log_lock:
                    with self.operation_log_path.open("a", encoding="utf-8") as handle:
                        handle.writelines(batch)
            except OSError:
                pass

            if stop_requested:
                break

    def _on_close(self) -> None:
        if self.filter_feedback_job is not None:
            self.root.after_cancel(self.filter_feedback_job)
            self.filter_feedback_job = None
        if self.chart_render_job is not None:
            self.root.after_cancel(self.chart_render_job)
            self.chart_render_job = None
        self.log_queue.put(None)
        if self.log_writer_thread.is_alive():
            self.log_writer_thread.join(timeout=0.5)
        self.root.destroy()

    @staticmethod
    def _replace_text(widget: tk.Text, content: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state="disabled")

    @staticmethod
    def _format_number(value: object) -> str:
        if pd.isna(value):
            return "-"
        return f"{int(value):,}"

    @staticmethod
    def _format_float(value: object) -> str:
        if pd.isna(value):
            return "-"
        return f"{float(value):.2f}"

    @staticmethod
    def _format_currency(value: object) -> str:
        if pd.isna(value):
            return "-"
        return f"${float(value):,.0f}"

    @staticmethod
    def _format_date(value: object) -> str:
        if pd.isna(value):
            return "-"
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)


def launch_app(dataset_path: str | None = None) -> None:
    root = tk.Tk()
    IMDbExplorerApp(root, dataset_path=dataset_path)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
