"""
Desktop UI for pasting email text and viewing Naïve Bayes classification.

Uses Tkinter (stdlib) — no extra packages. Trains on the SpamAssassin corpus.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from naive_bayes_spam_classifier import (
    MultinomialNaiveBayesClassifier,
    load_spamassassin,
)


def train_demo_classifier() -> MultinomialNaiveBayesClassifier:
    """Fit strictly on the SpamAssassin corpus."""
    train_docs, train_labels = load_spamassassin()

    clf = MultinomialNaiveBayesClassifier()
    clf.fit(train_docs, train_labels)
    return clf


class SpamClassifierApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Spam classifier — Naïve Bayes (scratch)")
        self.root.minsize(560, 520)
        self.root.geometry("720x580")

        self.clf = train_demo_classifier()

        self._build_widgets()

    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 8}

        header = ttk.Label(
            self.root,
            text="Paste email body (HTML or plain text). Result appears below.",
            wraplength=680,
        )
        header.pack(anchor="w", **pad)

        self.input_box = scrolledtext.ScrolledText(
            self.root,
            height=16,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            undo=True,
        )
        self.input_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        btn_row = ttk.Frame(self.root)
        btn_row.pack(fill=tk.X, padx=12, pady=(0, 4))

        classify_btn = ttk.Button(btn_row, text="Classify", command=self._on_classify)
        classify_btn.pack(side=tk.LEFT)

        clear_btn = ttk.Button(btn_row, text="Clear", command=self._on_clear)
        clear_btn.pack(side=tk.LEFT, padx=(8, 0))

        sep = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=12, pady=8)

        out_label = ttk.Label(self.root, text="Result", font=("Segoe UI", 10, "bold"))
        out_label.pack(anchor="w", padx=12)

        self.result_var = tk.StringVar(value="Prediction will appear here after you click Classify.")

        # Colored verdict line
        verdict_frame = ttk.Frame(self.root)
        verdict_frame.pack(fill=tk.X, padx=12, pady=(4, 0))
        self.verdict_label = ttk.Label(
            verdict_frame,
            textvariable=self.result_var,
            font=("Segoe UI", 13, "bold"),
        )
        self.verdict_label.pack(anchor="w")

        self.detail_box = scrolledtext.ScrolledText(
            self.root,
            height=6,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self.detail_box.pack(fill=tk.BOTH, expand=False, padx=12, pady=(4, 12))

        self.root.bind("<Control-Return>", lambda e: self._on_classify())

    def _set_verdict_style(self, prediction: str) -> None:
        pred = prediction.lower().strip()
        if pred == "spam":
            self.verdict_label.configure(foreground="#b00020")
        elif pred == "ham":
            self.verdict_label.configure(foreground="#156b2c")
        else:
            self.verdict_label.configure(foreground="#333333")

    def _on_clear(self) -> None:
        self.input_box.delete("1.0", tk.END)
        self.result_var.set("Prediction will appear here after you click Classify.")
        self.verdict_label.configure(foreground="#333333")
        self._set_detail_text("")

    def _set_detail_text(self, text: str) -> None:
        self.detail_box.configure(state=tk.NORMAL)
        self.detail_box.delete("1.0", tk.END)
        self.detail_box.insert(tk.END, text)
        self.detail_box.configure(state=tk.DISABLED)

    def _on_classify(self) -> None:
        raw = self.input_box.get("1.0", tk.END).strip()
        if not raw:
            self.result_var.set("(empty input)")
            self.verdict_label.configure(foreground="#666666")
            self._set_detail_text("Paste some text above, then click Classify.")
            return

        prediction = self.clf.predict(raw)
        scores = self.clf.predict_log_scores(raw)

        self.result_var.set(f"Prediction: {prediction.upper()}")
        self._set_verdict_style(prediction)

        lines = [
            "Log-scores ∝ log P(class) + Σ count(w)·log P(w|class)  (higher is better)",
            "",
        ]
        for cls in sorted(scores.keys()):
            lines.append(f"  {cls:>4}: {scores[cls]:.4f}")
        lines.append("")
        lines.append(f"Argmax → {prediction}")
        self._set_detail_text("\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SpamClassifierApp().run()


if __name__ == "__main__":
    main()
