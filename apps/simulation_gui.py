"""Small Tkinter GUI for pairwise traces and live round-robin tournaments."""

from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.round_robin_gui_model import (
    MatchResult,
    build_rankings,
    iter_round_robin_results,
    list_engine_names,
    list_strategy_names as list_round_robin_strategy_names,
    select_strategy_names,
)
from apps.simulation_gui_model import (
    ScorePoint,
    TraceSummary,
    build_trace_summary,
    list_strategy_names as list_pairwise_strategy_names,
)


DEFAULT_ROUND_ROBIN_SELECTION = (
    "AlwaysCooperate",
    "AlwaysDefect",
    "RandomStrategy",
    "TitForTat",
    "WinStayLoseShift",
    "Pavlov2",
    "Generous3",
)


class SimulationGUI:
    """Tkinter app for pairwise and tournament-level simulation views."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dilemma Dynamics Simulator")
        self.root.geometry("1120x760")
        self.root.minsize(940, 620)

        self.strategy_names = list_pairwise_strategy_names()
        self.summary_data: TraceSummary | None = None

        self.strategy_a = tk.StringVar(value="TitForTat")
        self.strategy_b = tk.StringVar(value="AlwaysDefect")
        self.rounds = tk.IntVar(value=12)
        self.error = tk.DoubleVar(value=0.0)
        self.initial_state = tk.StringVar(value="CC")
        self.seed = tk.StringVar(value="42")
        self.status = tk.StringVar(value="Ready")
        self.summary = tk.StringVar(value="")

        self.round_robin_strategy_names = list_round_robin_strategy_names()
        self.rr_engine = tk.StringVar(value="markov")
        self.rr_rounds = tk.IntVar(value=50)
        self.rr_trials = tk.IntVar(value=500)
        self.rr_error = tk.DoubleVar(value=0.0)
        self.rr_initial_state = tk.StringVar(value="CC")
        self.rr_update_delay_ms = tk.IntVar(value=25)
        self.rr_progress_text = tk.StringVar(value="No tournament running")
        self.rr_results: list[MatchResult] = []
        self.rr_selected_names: tuple[str, ...] = ()
        self.rr_queue: queue.Queue[tuple[str, int, object]] = queue.Queue()
        self.rr_run_id = 0
        self.rr_running = False
        self.rr_stop_event: threading.Event | None = None

        self._build_layout()
        self.run_simulation()
        self._select_all_round_robin()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        pairwise_tab = ttk.Frame(self.notebook)
        round_robin_tab = ttk.Frame(self.notebook)
        self.notebook.add(pairwise_tab, text="Pairwise simulation")
        self.notebook.add(round_robin_tab, text="Round robin")

        self._build_pairwise_tab(pairwise_tab)
        self._build_round_robin_tab(round_robin_tab)

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            padding=(12, 4),
        )
        status_bar.grid(row=1, column=0, sticky="ew")

    def _build_pairwise_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.Frame(parent, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        for col in range(8):
            controls.columnconfigure(col, weight=1)

        ttk.Label(controls, text="Strategy A").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.strategy_a,
            values=self.strategy_names,
            state="readonly",
            width=24,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Strategy B").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.strategy_b,
            values=self.strategy_names,
            state="readonly",
            width=24,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Rounds").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0,
            to=200,
            textvariable=self.rounds,
            width=8,
        ).grid(row=1, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Error").grid(row=0, column=3, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.error,
            width=8,
        ).grid(row=1, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Start").grid(row=0, column=4, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.initial_state,
            values=["CC", "CD", "DC", "DD"],
            state="readonly",
            width=6,
        ).grid(row=1, column=4, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Seed").grid(row=0, column=5, sticky="w")
        ttk.Entry(controls, textvariable=self.seed, width=8).grid(
            row=1,
            column=5,
            sticky="ew",
            padx=(0, 8),
        )

        ttk.Button(controls, text="Run", command=self.run_simulation).grid(
            row=1,
            column=6,
            sticky="ew",
            padx=(0, 8),
        )
        ttk.Button(controls, text="Swap", command=self.swap_strategies).grid(
            row=1,
            column=7,
            sticky="ew",
        )

        body = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        top = ttk.Frame(body)
        bottom = ttk.Frame(body)
        body.add(top, weight=1)
        body.add(bottom, weight=4)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=2)
        top.rowconfigure(0, weight=1)

        summary_frame = ttk.LabelFrame(top, text="Summary", padding=10)
        summary_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(
            summary_frame,
            textvariable=self.summary,
            justify="left",
            anchor="nw",
        ).pack(fill="both", expand=True)

        chart_frame = ttk.LabelFrame(top, text="Cumulative score", padding=8)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        self.chart = tk.Canvas(chart_frame, height=170, bg="white", highlightthickness=1)
        self.chart.grid(row=0, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

        table_frame = ttk.LabelFrame(bottom, text="Round trace", padding=8)
        table_frame.pack(fill="both", expand=True)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = (
            "round",
            "memory",
            "a_move",
            "b_move",
            "outcome",
            "payoff",
            "cumulative",
        )
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        headings = {
            "round": "Round",
            "memory": "Memory",
            "a_move": "A move",
            "b_move": "B move",
            "outcome": "Outcome",
            "payoff": "Payoff",
            "cumulative": "Cumulative",
        }
        widths = {
            "round": 70,
            "memory": 160,
            "a_move": 100,
            "b_move": 100,
            "outcome": 80,
            "payoff": 90,
            "cumulative": 120,
        }
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_round_robin_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left = ttk.Frame(parent, padding=12)
        left.grid(row=0, column=0, sticky="ns")

        right = ttk.Frame(parent, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(left, text="Strategies").grid(row=0, column=0, columnspan=2, sticky="w")
        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(4, 8))
        left.rowconfigure(1, weight=1)

        self.rr_strategy_list = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=18,
            width=30,
        )
        strategy_scroll = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.rr_strategy_list.yview,
        )
        self.rr_strategy_list.configure(yscrollcommand=strategy_scroll.set)
        self.rr_strategy_list.grid(row=0, column=0, sticky="nsew")
        strategy_scroll.grid(row=0, column=1, sticky="ns")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        for name in self.round_robin_strategy_names:
            self.rr_strategy_list.insert(tk.END, name)

        ttk.Button(left, text="Core set", command=self._set_default_round_robin_selection).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 8),
        )
        ttk.Button(left, text="Select all", command=self._select_all_round_robin).grid(
            row=2,
            column=1,
            sticky="ew",
            pady=(0, 8),
        )

        ttk.Label(left, text="Engine").grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self.rr_engine,
            values=list_engine_names(),
            state="readonly",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Label(left, text="Rounds").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(left, from_=0, to=500, textvariable=self.rr_rounds, width=8).grid(
            row=6,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(2, 8),
        )

        ttk.Label(left, text="Trials").grid(row=5, column=1, sticky="w")
        ttk.Spinbox(left, from_=1, to=100000, textvariable=self.rr_trials, width=8).grid(
            row=6,
            column=1,
            sticky="ew",
            pady=(2, 8),
        )

        ttk.Label(left, text="Error").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(
            left,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.rr_error,
            width=8,
        ).grid(row=8, column=0, sticky="ew", padx=(0, 4), pady=(2, 8))

        ttk.Label(left, text="Start").grid(row=7, column=1, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self.rr_initial_state,
            values=["CC", "CD", "DC", "DD"],
            state="readonly",
            width=6,
        ).grid(row=8, column=1, sticky="ew", pady=(2, 8))

        ttk.Label(left, text="Update delay ms").grid(
            row=9,
            column=0,
            columnspan=2,
            sticky="w",
        )
        ttk.Spinbox(
            left,
            from_=0,
            to=1000,
            textvariable=self.rr_update_delay_ms,
            width=8,
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        self.rr_run_button = ttk.Button(
            left,
            text="Run round robin",
            command=self.start_round_robin,
        )
        self.rr_run_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        self.rr_stop_button = ttk.Button(
            left,
            text="Stop",
            command=self.stop_round_robin,
            state="disabled",
        )
        self.rr_stop_button.grid(row=12, column=0, columnspan=2, sticky="ew")

        progress_frame = ttk.Frame(right)
        progress_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        self.rr_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.rr_progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(progress_frame, textvariable=self.rr_progress_text).grid(
            row=0,
            column=1,
            sticky="e",
        )

        panes = ttk.PanedWindow(right, orient=tk.VERTICAL)
        panes.grid(row=1, column=0, sticky="nsew")

        ranking_frame = ttk.LabelFrame(panes, text="Live rankings", padding=8)
        match_frame = ttk.LabelFrame(panes, text="Completed matchups", padding=8)
        panes.add(ranking_frame, weight=2)
        panes.add(match_frame, weight=3)

        self.rr_ranking_table = self._make_tree(
            ranking_frame,
            columns=("rank", "strategy", "total", "avg", "played", "wins", "losses", "ties"),
            headings={
                "rank": "Rank",
                "strategy": "Strategy",
                "total": "Total score",
                "avg": "Avg / match",
                "played": "Played",
                "wins": "Wins",
                "losses": "Losses",
                "ties": "Ties",
            },
            widths={
                "rank": 70,
                "strategy": 210,
                "total": 120,
                "avg": 110,
                "played": 80,
                "wins": 70,
                "losses": 70,
                "ties": 70,
            },
        )

        self.rr_match_table = self._make_tree(
            match_frame,
            columns=("match", "strategy_a", "score_a", "strategy_b", "score_b"),
            headings={
                "match": "Match",
                "strategy_a": "Strategy A",
                "score_a": "Score A",
                "strategy_b": "Strategy B",
                "score_b": "Score B",
            },
            widths={
                "match": 100,
                "strategy_a": 220,
                "score_a": 100,
                "strategy_b": 220,
                "score_b": 100,
            },
        )

    def _make_tree(
        self,
        parent: ttk.Frame,
        *,
        columns: tuple[str, ...],
        headings: dict[str, str],
        widths: dict[str, int],
    ) -> ttk.Treeview:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        vertical = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center", stretch=True)
        return tree

    def swap_strategies(self) -> None:
        strategy_a = self.strategy_a.get()
        strategy_b = self.strategy_b.get()
        self.strategy_a.set(strategy_b)
        self.strategy_b.set(strategy_a)
        self.run_simulation()

    def run_simulation(self) -> None:
        try:
            rounds = int(self.rounds.get())
            error = float(self.error.get())
            seed_text = self.seed.get().strip()
            seed = int(seed_text) if seed_text else None
            self.summary_data = build_trace_summary(
                self.strategy_a.get(),
                self.strategy_b.get(),
                rounds=rounds,
                error=error,
                initial_state=self.initial_state.get(),
                seed=seed,
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            self.status.set("Simulation failed")
            return

        self._populate_table()
        self._draw_chart()
        self._update_summary()
        rendered_rounds = len(self.summary_data.trace) if self.summary_data else 0
        self.status.set(f"Rendered {rendered_rounds} rounds")

    def _populate_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

        if self.summary_data is None:
            return

        for row in self.summary_data.table_rows:
            self.table.insert(
                "",
                "end",
                values=(
                    row.round_index,
                    row.memory,
                    row.strategy_a_move,
                    row.strategy_b_move,
                    row.outcome,
                    f"{row.payoff_a:.0f}-{row.payoff_b:.0f}",
                    f"{row.cumulative_score_a:.0f}-{row.cumulative_score_b:.0f}",
                ),
            )

    def _update_summary(self) -> None:
        strategy_a = self.strategy_a.get()
        strategy_b = self.strategy_b.get()
        if self.summary_data is None or not self.summary_data.trace:
            self.summary.set(f"{strategy_a} vs {strategy_b}\nNo rounds simulated.")
            return

        trace = self.summary_data.trace
        cooperation_a = sum(row.actual_move_1 == "C" for row in trace)
        cooperation_b = sum(row.actual_move_2 == "C" for row in trace)
        timeline = " ".join(self.summary_data.timeline)
        self.summary.set(
            f"{strategy_a} vs {strategy_b}\n"
            f"Final score: {self.summary_data.final_score_a:.0f} to "
            f"{self.summary_data.final_score_b:.0f}\n"
            f"Cooperation count: {cooperation_a} to {cooperation_b}\n"
            f"Timeline: {timeline}"
        )

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 320)
        height = max(self.chart.winfo_height(), 150)
        pad = 24

        self.chart.create_line(pad, height - pad, width - pad, height - pad, fill="#999")
        self.chart.create_line(pad, pad, pad, height - pad, fill="#999")
        self.chart.create_text(pad + 4, pad, text="score", anchor="w", fill="#555")
        self.chart.create_text(width - pad, height - 8, text="round", anchor="e", fill="#555")

        if self.summary_data is None or not self.summary_data.trace:
            self.chart.create_text(width / 2, height / 2, text="No rounds", fill="#555")
            return

        points = self.summary_data.score_points
        scores_a = [point.strategy_a_score for point in points]
        scores_b = [point.strategy_b_score for point in points]
        max_score = max(max(scores_a), max(scores_b), 1.0)
        rounds = max(point.round_index for point in points)

        def canvas_point(point: ScorePoint, score: float) -> tuple[float, float]:
            x = pad + (width - 2 * pad) * point.round_index / max(rounds, 1)
            y = height - pad - (height - 2 * pad) * score / max_score
            return x, y

        points_a = [
            coord
            for point in points
            for coord in canvas_point(point, point.strategy_a_score)
        ]
        points_b = [
            coord
            for point in points
            for coord in canvas_point(point, point.strategy_b_score)
        ]
        self.chart.create_line(*points_a, fill="#1f77b4", width=2)
        self.chart.create_line(*points_b, fill="#d62728", width=2)
        self.chart.create_text(
            width - pad,
            pad + 12,
            text=self.strategy_a.get(),
            anchor="e",
            fill="#1f77b4",
        )
        self.chart.create_text(
            width - pad,
            pad + 30,
            text=self.strategy_b.get(),
            anchor="e",
            fill="#d62728",
        )

    def _set_default_round_robin_selection(self) -> None:
        self.rr_strategy_list.selection_clear(0, tk.END)
        available = set(self.round_robin_strategy_names)
        for index, name in enumerate(self.round_robin_strategy_names):
            if name in DEFAULT_ROUND_ROBIN_SELECTION and name in available:
                self.rr_strategy_list.selection_set(index)

    def _select_all_round_robin(self) -> None:
        self.rr_strategy_list.selection_set(0, tk.END)

    def _selected_round_robin_names(self) -> tuple[str, ...]:
        indexes = self.rr_strategy_list.curselection()
        return tuple(self.round_robin_strategy_names[index] for index in indexes)

    def start_round_robin(self) -> None:
        if self.rr_running:
            return

        try:
            selected = select_strategy_names(self._selected_round_robin_names())
            rounds = int(self.rr_rounds.get())
            trials = int(self.rr_trials.get())
            error = float(self.rr_error.get())
            delay_seconds = max(0, int(self.rr_update_delay_ms.get())) / 1000
            iterator = iter_round_robin_results(
                selected,
                engine_type=self.rr_engine.get(),
                rounds=rounds,
                trials=trials,
                error=error,
                initial_state=self.rr_initial_state.get(),
            )
        except Exception as exc:
            messagebox.showerror("Round robin error", str(exc))
            self.status.set("Round robin configuration failed")
            return

        self.rr_run_id += 1
        run_id = self.rr_run_id
        stop_event = threading.Event()
        self.rr_stop_event = stop_event
        self.rr_running = True
        self.rr_selected_names = selected
        self.rr_results = []
        self.rr_run_button.configure(state="disabled")
        self.rr_stop_button.configure(state="normal")
        self._clear_tree(self.rr_match_table)
        self._refresh_round_robin_rankings()
        total_matches = len(selected) * (len(selected) - 1) // 2
        self.rr_progress.configure(maximum=total_matches, value=0)
        self.rr_progress_text.set(f"0 / {total_matches} matches")
        self.status.set("Round robin running")

        worker = threading.Thread(
            target=self._round_robin_worker,
            args=(run_id, stop_event, iterator, delay_seconds),
            daemon=True,
        )
        worker.start()
        self.root.after(50, self._poll_round_robin_queue)

    def stop_round_robin(self) -> None:
        if self.rr_stop_event is not None:
            self.rr_stop_event.set()
        self.status.set("Stopping round robin")

    def _round_robin_worker(
        self,
        run_id: int,
        stop_event: threading.Event,
        iterator,
        delay_seconds: float,
    ) -> None:
        try:
            for result in iterator:
                if stop_event.is_set():
                    self.rr_queue.put(("stopped", run_id, None))
                    return
                self.rr_queue.put(("match", run_id, result))
                if delay_seconds:
                    time.sleep(delay_seconds)
            self.rr_queue.put(("done", run_id, None))
        except Exception as exc:
            self.rr_queue.put(("error", run_id, exc))

    def _poll_round_robin_queue(self) -> None:
        while True:
            try:
                kind, run_id, payload = self.rr_queue.get_nowait()
            except queue.Empty:
                break

            if run_id != self.rr_run_id:
                continue

            if kind == "match":
                self._record_round_robin_match(payload)
            elif kind == "done":
                self._finish_round_robin("Round robin complete")
            elif kind == "stopped":
                self._finish_round_robin("Round robin stopped")
            elif kind == "error":
                self._finish_round_robin("Round robin failed")
                messagebox.showerror("Round robin error", str(payload))

        if self.rr_running:
            self.root.after(50, self._poll_round_robin_queue)

    def _record_round_robin_match(self, result: MatchResult) -> None:
        self.rr_results.append(result)
        self.rr_match_table.insert(
            "",
            "end",
            values=(
                f"{result.match_index} / {result.total_matches}",
                result.strategy_a,
                f"{result.score_a:.2f}",
                result.strategy_b,
                f"{result.score_b:.2f}",
            ),
        )
        self.rr_match_table.yview_moveto(1.0)
        self._refresh_round_robin_rankings()
        self.rr_progress.configure(value=result.match_index)
        self.rr_progress_text.set(f"{result.match_index} / {result.total_matches} matches")
        leader = self.rr_results and build_rankings(self.rr_selected_names, self.rr_results)[0]
        if leader:
            self.status.set(
                f"Live leader: {leader.strategy} with {leader.total_score:.2f} points"
            )

    def _refresh_round_robin_rankings(self) -> None:
        self._clear_tree(self.rr_ranking_table)
        if not self.rr_selected_names:
            return
        rankings = build_rankings(self.rr_selected_names, self.rr_results)
        for row in rankings:
            self.rr_ranking_table.insert(
                "",
                "end",
                values=(
                    row.rank,
                    row.strategy,
                    f"{row.total_score:.2f}",
                    f"{row.average_score_per_match:.2f}",
                    row.matches_played,
                    row.wins,
                    row.losses,
                    row.ties,
                ),
            )

    def _finish_round_robin(self, message: str) -> None:
        self.rr_running = False
        self.rr_run_button.configure(state="normal")
        self.rr_stop_button.configure(state="disabled")
        self.status.set(message)

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)


def main() -> int:
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
