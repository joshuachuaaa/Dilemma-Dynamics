"""Small Tkinter GUI for pairwise simulation traces."""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.simulation_gui_model import (
    ScorePoint,
    TraceSummary,
    build_trace_summary,
    list_strategy_names,
)


class SimulationGUI:
    """Tkinter app for inspecting one simulated matchup."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dilemma Dynamics Simulator")
        self.root.geometry("980x680")
        self.root.minsize(820, 560)

        self.strategy_names = list_strategy_names()
        self.summary_data: TraceSummary | None = None

        self.strategy_a = tk.StringVar(value="TitForTat")
        self.strategy_b = tk.StringVar(value="AlwaysDefect")
        self.rounds = tk.IntVar(value=12)
        self.error = tk.DoubleVar(value=0.0)
        self.initial_state = tk.StringVar(value="CC")
        self.seed = tk.StringVar(value="42")
        self.status = tk.StringVar(value="Ready")
        self.summary = tk.StringVar(value="")

        self._build_layout()
        self.run_simulation()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        for col in range(10):
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
            row=1, column=5, sticky="ew", padx=(0, 8)
        )

        ttk.Button(controls, text="Run", command=self.run_simulation).grid(
            row=1, column=6, sticky="ew", padx=(0, 8)
        )
        ttk.Button(controls, text="Swap", command=self.swap_strategies).grid(
            row=1, column=7, sticky="ew", padx=(0, 8)
        )

        body = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
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

        status_bar = ttk.Label(self.root, textvariable=self.status, anchor="w", padding=(12, 4))
        status_bar.grid(row=2, column=0, sticky="ew")

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
        self.chart.create_text(width - pad, pad + 12, text=self.strategy_a.get(), anchor="e", fill="#1f77b4")
        self.chart.create_text(width - pad, pad + 30, text=self.strategy_b.get(), anchor="e", fill="#d62728")


def main() -> int:
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
