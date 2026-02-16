import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
import io
import importlib

# ─── CONFIGURE THIS ──────────────────────────────────────────────────────────
SCRIPT_MODULE = "converter" 
# ─────────────────────────────────────────────────────────────────────────────


# ── Colours & fonts ───────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL     = "#16213e"
ACCENT    = "#e94560"
ACCENT2   = "#0f3460"
TEXT      = "#e0e0e0"
TEXT_DIM  = "#888899"
SUCCESS   = "#4ecca3"
WARNING   = "#f5a623"
ERROR_COL = "#ff6b6b"
FONT_UI   = ("SF Pro Display", 13) if sys.platform == "darwin" else ("Segoe UI", 11)
FONT_MONO = ("SF Mono", 11)        if sys.platform == "darwin" else ("Consolas", 10)


class StreamRedirector(io.StringIO):
    """Captures stdout and feeds it to the log widget in real time."""
    def __init__(self, log_widget, tag="info"):
        super().__init__()
        self.log_widget = log_widget
        self.tag = tag

    def write(self, text):
        if text.strip():
            self.log_widget.after(0, self._append, text)

    def _append(self, text):
        tag = "info"
        if any(k in text for k in ("✅", "🎉", "📊")):
            tag = "success"
        elif "❌" in text or "Error" in text or "error" in text:
            tag = "error"
        elif any(k in text for k in ("🔍", "📁", "📄")):
            tag = "dim"
        elif "⚠" in text:
            tag = "warning"

        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, text + "\n", tag)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")

    def flush(self):
        pass


class PublisherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Obsidian Publisher")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(600, 480)

        self._running = False
        self._mod = None

        self._build_ui()
        self._center_window(700, 560)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self, bg=ACCENT2, pady=14)
        header.pack(fill="x")

        tk.Label(header, text="◈  Obsidian Publisher", bg=ACCENT2, fg=TEXT,
                 font=(FONT_UI[0], 16, "bold")).pack(side="left", padx=20)

        self.status_dot = tk.Label(header, text="●", bg=ACCENT2, fg=SUCCESS,
                                   font=(FONT_UI[0], 14))
        self.status_dot.pack(side="right", padx=6)
        self.status_label = tk.Label(header, text="Ready", bg=ACCENT2, fg=TEXT_DIM,
                                     font=FONT_UI)
        self.status_label.pack(side="right")

        # ── Button row ──
        btn_row = tk.Frame(self, bg=BG, pady=16)
        btn_row.pack(fill="x", padx=20)

        self.btn_convert = self._make_button(
            btn_row, "⇄  Convert MD → HTML",
            cmd=self._run_convert, color=ACCENT2
        )
        self.btn_convert.pack(side="left", padx=(0, 10))

        self.btn_push = self._make_button(
            btn_row, "↑  Push to GitHub",
            cmd=self._run_push, color=ACCENT2
        )
        self.btn_push.pack(side="left", padx=(0, 10))

        self.btn_all = self._make_button(
            btn_row, "✦  Convert + Push",
            cmd=self._run_all, color=ACCENT
        )
        self.btn_all.pack(side="left")

        self.btn_clear = self._make_button(
            btn_row, "✕  Clear Log",
            cmd=self._clear_log, color="#333355"
        )
        self.btn_clear.pack(side="right")

        # ── Branch selector ──
        branch_row = tk.Frame(self, bg=BG, pady=0)
        branch_row.pack(fill="x", padx=20, pady=(0, 12))

        tk.Label(branch_row, text="Branch:", bg=BG, fg=TEXT_DIM,
                 font=FONT_UI).pack(side="left", padx=(0, 10))

        self.branch_var = tk.StringVar()
        self.branch_menu = tk.OptionMenu(branch_row, self.branch_var, "")
        self.branch_menu.configure(
            bg=PANEL, fg=TEXT, activebackground=ACCENT2,
            activeforeground=TEXT, relief="flat", font=FONT_UI,
            highlightthickness=0, bd=0, padx=10, pady=6,
            indicatoron=True, cursor="hand2"
        )
        self.branch_menu["menu"].configure(
            bg=PANEL, fg=TEXT, activebackground=ACCENT,
            activeforeground="white", font=FONT_UI, relief="flat"
        )
        self.branch_menu.pack(side="left")

        self.btn_refresh = self._make_button(
            branch_row, "⟳  Refresh",
            cmd=self._load_branches, color="#333355"
        )
        self.btn_refresh.pack(side="left", padx=(10, 0))

        self.current_branch_label = tk.Label(
            branch_row, text="", bg=BG, fg=TEXT_DIM, font=(FONT_UI[0], 10)
        )
        self.current_branch_label.pack(side="left", padx=(16, 0))

        # Load branches on startup
        self.after(100, self._load_branches)

        # ── Log area ──
        log_frame = tk.Frame(self, bg=PANEL, padx=2, pady=2)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        log_inner = tk.Frame(log_frame, bg=PANEL)
        log_inner.pack(fill="both", expand=True)

        tk.Label(log_inner, text="  Output Log", bg=PANEL, fg=TEXT_DIM,
                 font=(FONT_UI[0], 10), anchor="w").pack(fill="x")

        self.log = scrolledtext.ScrolledText(
            log_inner, bg="#0d0d1a", fg=TEXT,
            font=FONT_MONO, relief="flat",
            state="disabled", wrap="word",
            insertbackground=ACCENT,
            selectbackground=ACCENT2,
            pady=8, padx=10
        )
        self.log.pack(fill="both", expand=True)

        # colour tags for the log
        self.log.tag_config("success", foreground=SUCCESS)
        self.log.tag_config("error",   foreground=ERROR_COL)
        self.log.tag_config("warning", foreground=WARNING)
        self.log.tag_config("dim",     foreground=TEXT_DIM)
        self.log.tag_config("info",    foreground=TEXT)

        self._log("Publisher ready. Choose an action above.\n", "dim")

    def _make_button(self, parent, text, cmd, color):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=TEXT, activebackground=ACCENT,
            activeforeground="white", relief="flat",
            font=FONT_UI, padx=16, pady=8,
            cursor="hand2", bd=0
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=ACCENT))
        btn.bind("<Leave>", lambda e: btn.configure(bg=color))
        btn._default_color = color
        return btn

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _center_window(self, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _log(self, msg, tag="info"):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n", tag)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _set_status(self, text, color=TEXT_DIM):
        self.status_label.configure(text=text, fg=color)
        self.status_dot.configure(fg=color)

    def _set_busy(self, busy: bool):
        self._running = busy
        state = "disabled" if busy else "normal"
        for btn in (self.btn_convert, self.btn_push, self.btn_all):
            btn.configure(state=state)

    def _load_branches(self):
        """Populate the branch dropdown from git and mark the current branch."""
        import subprocess
        try:
            # All local branches
            result = subprocess.run(
                ["git", "branch"], capture_output=True, text=True
            )
            branches = []
            current = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("* "):
                    current = line[2:].strip()
                    branches.append(current)
                elif line:
                    branches.append(line)

            if not branches:
                self._log("⚠ No git branches found. Is this a git repo?\n", "warning")
                return

            # Rebuild the OptionMenu
            menu = self.branch_menu["menu"]
            menu.delete(0, "end")
            for b in branches:
                menu.add_command(
                    label=b,
                    command=lambda v=b: self.branch_var.set(v)
                )

            # Default to current branch
            if current:
                self.branch_var.set(current)
                self.current_branch_label.configure(
                    text=f"(currently on: {current})", fg=SUCCESS
                )
            else:
                self.branch_var.set(branches[0])

            self._log(f"✅ Branches loaded. Current: {current}\n", "success")

        except FileNotFoundError:
            self._log("❌ git not found. Make sure git is installed and in your PATH.\n", "error")
        except Exception as exc:
            self._log(f"❌ Could not load branches: {exc}\n", "error")

    def _load_module(self):
        """Import (or reload) the publisher script so we always get fresh globals."""
        if self._mod is None:
            self._mod = importlib.import_module(SCRIPT_MODULE)
        else:
            importlib.reload(self._mod)
        return self._mod

    def _run_in_thread(self, task_fn):
        if self._running:
            return
        self._set_busy(True)
        self._set_status("Running…", WARNING)
        thread = threading.Thread(target=self._worker, args=(task_fn,), daemon=True)
        thread.start()

    def _worker(self, task_fn):
        redirector = StreamRedirector(self.log)
        old_stdout = sys.stdout
        sys.stdout = redirector
        try:
            task_fn()
            self.after(0, self._set_status, "Done ✓", SUCCESS)
        except Exception as exc:
            self.after(0, self._log, f"\n❌ Error: {exc}\n", "error")
            self.after(0, self._set_status, "Failed ✗", ERROR_COL)
        finally:
            sys.stdout = old_stdout
            self.after(0, self._set_busy, False)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _run_convert(self):
        def task():
            m = self._load_module()
            self._log("\n── Convert MD → HTML ──────────────────\n", "dim")
            m.build_image_map()
            m.copy_images()
            m.generate_index_pages()
            m.generate_material_pages()
            print("\n🎉 Conversion complete!")
        self._run_in_thread(task)

    def _run_push(self):
        branch = self.branch_var.get()
        if not branch:
            self._log("⚠ No branch selected. Use the dropdown to pick one.\n", "warning")
            return
        def task():
            import subprocess
            self._log(f"\n── Push to GitHub → {branch} ───────────────\n", "dim")
            for cmd in (
                ["git", "add", "-A"],
                ["git", "commit", "-m", "publish: auto update from GUI"],
                ["git", "push", "origin", branch],
            ):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout.strip():
                    print(result.stdout.strip())
                if result.stderr.strip():
                    print(result.stderr.strip())
            print(f"\n🎉 Push to '{branch}' complete!")
        self._run_in_thread(task)

    def _run_all(self):
        branch = self.branch_var.get()
        if not branch:
            self._log("⚠ No branch selected. Use the dropdown to pick one.\n", "warning")
            return
        def task():
            m = self._load_module()
            self._log("\n── Convert + Push ─────────────────────\n", "dim")
            m.build_image_map()
            m.copy_images()
            m.generate_index_pages()
            m.generate_material_pages()
            print("\n🎉 Conversion complete!")

            import subprocess
            self._log(f"\n── Pushing to GitHub → {branch} ────────\n", "dim")
            for cmd in (
                ["git", "add", "-A"],
                ["git", "commit", "-m", "publish: auto update from GUI"],
                ["git", "push", "origin", branch],
            ):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout.strip():
                    print(result.stdout.strip())
                if result.stderr.strip():
                    print(result.stderr.strip())
            print(f"\n🎉 All done — converted and pushed to '{branch}'!")
        self._run_in_thread(task)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")
        self._set_status("Ready", TEXT_DIM)
        self._log("Log cleared.\n", "dim")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PublisherApp()
    app.mainloop()
