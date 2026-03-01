"""
Obsidian Publisher GUI
──────────────────────
Drop this file next to converter.py (and your git repo root).
Run:  python publisher_gui.py
Or trigger via Obsidian hotkey pointing to this script.

Fixes vs previous versions
───────────────────────────
• All subprocess calls use cwd=REPO_ROOT — git always runs in the right folder
• REPO_ROOT is auto-detected as the directory containing this script
• All UI updates from worker threads go through self.after() — fully thread-safe
• Hover colours work cross-platform (fg is explicitly restored on Leave)
• Fonts are resolved at runtime from what's available on the OS
• converter.py is imported with sys.path patched so it always resolves
• Push skips commit noise when there's nothing new to commit
• "New Branch" dialog works correctly and reloads the dropdown after
"""

import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
import threading
import sys
import io
import importlib
import subprocess
import pathlib

# ── Repo root: the directory that contains THIS script ────────────────────────
REPO_ROOT = str(pathlib.Path(__file__).resolve().parent)

# ── Module to import for conversion ──────────────────────────────────────────
SCRIPT_MODULE = "converter"

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#001233"   # deep navy
PANEL     = "#002366"   # royal blue panel
PANEL2    = "#000d26"   # log background
ACCENT    = "#FFD700"   # gold
ACCENT_FG = "#001233"   # text on gold buttons
ACCENT2   = "#1a3a8a"   # mid-blue (default button bg)
TEXT      = "#f0f4ff"
TEXT_DIM  = "#7a8ab0"
SUCCESS   = "#FFD700"
WARNING   = "#F5C518"
ERROR_COL = "#ff6060"

# ── Font preference lists (cross-platform) ────────────────────────────────────
_UI_PREF   = ["SF Pro Display", "Segoe UI", "Helvetica Neue", "Arial"]
_MONO_PREF = ["SF Mono", "Consolas", "Menlo", "Courier New", "Courier"]

NEW_BRANCH_OPTION = "+ New Branch…"


# ── stdout redirector ─────────────────────────────────────────────────────────
class StreamRedirector(io.StringIO):
    """Captures stdout from worker thread and feeds it to the log via after()."""
    def __init__(self, app):
        super().__init__()
        self.app = app

    def write(self, text):
        if text and text.strip():
            self.app.after(0, self.app._log_raw, text.rstrip("\n"))

    def flush(self):
        pass


# ── Main application ──────────────────────────────────────────────────────────
class PublisherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Obsidian Publisher")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(640, 500)

        self._running = False
        self._mod     = None

        self._resolve_fonts()
        self._build_ui()
        self._center_window(740, 580)

        # Patch sys.path so converter.py is always importable
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)

        self.after(120, self._load_branches)

    # ── Font resolution ───────────────────────────────────────────────────────

    def _resolve_fonts(self):
        import tkinter.font as tkf
        families = set(tkf.families())

        def pick(prefs, size, *styles):
            for f in prefs:
                if f in families:
                    return (f, size) + styles
            return (prefs[-1], size) + styles

        self.FONT_UI   = pick(_UI_PREF,   11)
        self.FONT_UI_B = pick(_UI_PREF,   11, "bold")
        self.FONT_HEAD = pick(_UI_PREF,   15, "bold")
        self.FONT_MONO = pick(_MONO_PREF, 10)
        self.FONT_TINY = pick(_UI_PREF,    9)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=PANEL, pady=13)
        header.pack(fill="x")

        tk.Label(
            header, text="◈  Obsidian Publisher",
            bg=PANEL, fg=ACCENT, font=self.FONT_HEAD
        ).pack(side="left", padx=20)

        self.status_dot = tk.Label(
            header, text="●", bg=PANEL, fg=SUCCESS,
            font=(self.FONT_HEAD[0], 13)
        )
        self.status_dot.pack(side="right", padx=(0, 18))

        self.status_label = tk.Label(
            header, text="Ready", bg=PANEL, fg=TEXT_DIM, font=self.FONT_UI
        )
        self.status_label.pack(side="right")

        # ── Repo path hint ──────────────────────────────────────────────────
        repo_row = tk.Frame(self, bg=BG)
        repo_row.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(
            repo_row,
            text=f"Repo: {REPO_ROOT}",
            bg=BG, fg=TEXT_DIM, font=self.FONT_TINY, anchor="w"
        ).pack(side="left")

        # ── Button row ──────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG, pady=14)
        btn_row.pack(fill="x", padx=20)

        self.btn_convert = self._btn(btn_row, "⇄  Convert MD → HTML", self._run_convert)
        self.btn_convert.pack(side="left", padx=(0, 8))

        self.btn_push = self._btn(btn_row, "↑  Push to GitHub", self._run_push)
        self.btn_push.pack(side="left", padx=(0, 8))

        self.btn_all = self._btn(btn_row, "✦  Convert + Push", self._run_all, primary=True)
        self.btn_all.pack(side="left")

        self.btn_clear = self._btn(btn_row, "✕  Clear Log", self._clear_log, subtle=True)
        self.btn_clear.pack(side="right")

        # ── Branch row ──────────────────────────────────────────────────────
        branch_row = tk.Frame(self, bg=BG, pady=0)
        branch_row.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(
            branch_row, text="Branch:", bg=BG, fg=TEXT_DIM, font=self.FONT_UI
        ).pack(side="left", padx=(0, 8))

        self.branch_var = tk.StringVar()
        self.branch_var.trace("w", self._on_branch_selected)

        self.branch_menu = tk.OptionMenu(branch_row, self.branch_var, "loading…")
        self._style_option_menu(self.branch_menu)
        self.branch_menu.pack(side="left")

        self.btn_refresh = self._btn(branch_row, "⟳", self._load_branches, subtle=True)
        self.btn_refresh.pack(side="left", padx=(8, 0))

        self.current_branch_label = tk.Label(
            branch_row, text="", bg=BG, fg=TEXT_DIM, font=self.FONT_TINY
        )
        self.current_branch_label.pack(side="left", padx=(14, 0))

        # ── Log area ────────────────────────────────────────────────────────
        log_outer = tk.Frame(self, bg=PANEL, padx=1, pady=1)
        log_outer.pack(fill="both", expand=True, padx=20, pady=(0, 18))

        log_inner = tk.Frame(log_outer, bg=PANEL2)
        log_inner.pack(fill="both", expand=True)

        log_header = tk.Frame(log_inner, bg=PANEL, pady=4)
        log_header.pack(fill="x")
        tk.Label(
            log_header, text="  Output Log", bg=PANEL, fg=TEXT_DIM,
            font=self.FONT_TINY, anchor="w"
        ).pack(side="left")

        self.log = scrolledtext.ScrolledText(
            log_inner,
            bg=PANEL2, fg=TEXT,
            font=self.FONT_MONO,
            relief="flat",
            state="disabled",
            wrap="word",
            insertbackground=ACCENT,
            selectbackground=ACCENT2,
            pady=10, padx=12,
            borderwidth=0
        )
        self.log.pack(fill="both", expand=True)

        self.log.tag_config("success", foreground=SUCCESS)
        self.log.tag_config("error",   foreground=ERROR_COL)
        self.log.tag_config("warning", foreground=WARNING)
        self.log.tag_config("dim",     foreground=TEXT_DIM)
        self.log.tag_config("info",    foreground=TEXT)

        self._log("Publisher ready — choose an action above.", "dim")
        self._log(f"Working directory: {REPO_ROOT}", "dim")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, primary=False, subtle=False):
        """Create a styled button with correct cross-platform hover behaviour."""
        if primary:
            bg, fg, h_bg, h_fg = ACCENT, ACCENT_FG, "#ffe44d", ACCENT_FG
        elif subtle:
            bg, fg, h_bg, h_fg = "#0a1840", TEXT_DIM, ACCENT2, TEXT
        else:
            bg, fg, h_bg, h_fg = ACCENT2, ACCENT, ACCENT, ACCENT_FG

        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg,
            activebackground=h_bg, activeforeground=h_fg,
            relief="flat", font=self.FONT_UI,
            padx=14, pady=7,
            cursor="hand2", bd=0,
            highlightthickness=0
        )
        # Store colours on the widget so lambdas don't capture stale refs
        btn._bg, btn._fg, btn._h_bg, btn._h_fg = bg, fg, h_bg, h_fg
        btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=b._h_bg, fg=b._h_fg))
        btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=b._bg,   fg=b._fg))
        return btn

    def _style_option_menu(self, w):
        w.configure(
            bg=ACCENT2, fg=ACCENT,
            activebackground=ACCENT, activeforeground=ACCENT_FG,
            relief="flat", font=self.FONT_UI,
            highlightthickness=0, bd=0,
            padx=10, pady=6,
            cursor="hand2",
            indicatoron=True
        )
        w["menu"].configure(
            bg=PANEL, fg=TEXT,
            activebackground=ACCENT, activeforeground=ACCENT_FG,
            font=self.FONT_UI, relief="flat",
            borderwidth=0
        )

    def _center_window(self, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg, tag="info"):
        """Main-thread only. Append a line to the log widget."""
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n", tag)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _log_raw(self, text):
        """Called via self.after() from worker threads — runs on main thread."""
        tag = "info"
        low = text.lower()
        if any(k in text for k in ("✅", "🎉", "📊", "Done", "complete", "──")):
            tag = "success" if "──" not in text else "dim"
        elif "❌" in text or "error" in low or "fatal" in low:
            tag = "error"
        elif "⚠" in text or "warning" in low:
            tag = "warning"
        elif any(k in text for k in ("🔍", "📁", "📄")):
            tag = "dim"
        self._log(text, tag)

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, text, color=TEXT_DIM):
        self.status_label.configure(text=text, fg=color)
        self.status_dot.configure(fg=color)

    def _set_busy(self, busy):
        self._running = busy
        state = "disabled" if busy else "normal"
        for btn in (self.btn_convert, self.btn_push, self.btn_all):
            btn.configure(state=state)

    # ── Branch management ────────────────────────────────────────────────────

    def _on_branch_selected(self, *_):
        if self.branch_var.get() == NEW_BRANCH_OPTION:
            self.after(50, self._create_new_branch)

    def _git(self, *args):
        """Run a git command rooted at REPO_ROOT."""
        return subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True,
            cwd=REPO_ROOT
        )

    def _load_branches(self):
        try:
            result = self._git("branch")
            branches, current = [], None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("* "):
                    current = stripped[2:].strip()
                    branches.append(current)
                elif stripped:
                    branches.append(stripped)

            if not branches:
                self._log("⚠ No git branches found — is this a git repo?", "warning")
                return

            self._populate_branch_menu(branches, current)
            self._log(f"✅ Branches loaded. Current: {current}", "success")

        except FileNotFoundError:
            self._log("❌ git not found — install git and add it to your PATH.", "error")
        except Exception as exc:
            self._log(f"❌ Could not load branches: {exc}", "error")

    def _populate_branch_menu(self, branches, current=None):
        menu = self.branch_menu["menu"]
        menu.delete(0, "end")
        for b in branches:
            menu.add_command(label=b, command=lambda v=b: self.branch_var.set(v))
        menu.add_separator()
        menu.add_command(
            label=NEW_BRANCH_OPTION,
            command=lambda: self.branch_var.set(NEW_BRANCH_OPTION)
        )
        # Temporarily remove all write traces so setting the var
        # doesn't re-trigger _on_branch_selected (which opens the dialog).
        # trace_info() returns tuples of varying length across Tk versions,
        # so we just grab the last element (the callback id) safely.
        for info in self.branch_var.trace_info():
            if info[0] == "write":
                self.branch_var.trace_remove("write", info[-1])

        chosen = current or (branches[0] if branches else "")
        self.branch_var.set(chosen)

        # Re-attach trace
        self.branch_var.trace("w", self._on_branch_selected)

        if current:
            self.current_branch_label.configure(text=f"(on: {current})", fg=SUCCESS)

    def _create_new_branch(self):
        name = simpledialog.askstring(
            "New Branch",
            "Branch name (use hyphens, no spaces):",
            parent=self
        )
        if not name or not name.strip():
            self._load_branches()
            return
        name = name.strip().replace(" ", "-")
        result = self._git("checkout", "-b", name)
        if result.returncode == 0:
            self._log(f"✅ Created and switched to '{name}'", "success")
            self._load_branches()
        else:
            err = result.stderr.strip()
            self._log(f"❌ Could not create branch: {err}", "error")
            messagebox.showerror("Branch Error", err, parent=self)
            self._load_branches()

    # ── Threading ────────────────────────────────────────────────────────────

    def _run_in_thread(self, task_fn):
        if self._running:
            return
        self._set_busy(True)
        self._set_status("Running…", WARNING)
        threading.Thread(target=self._worker, args=(task_fn,), daemon=True).start()

    def _worker(self, task_fn):
        redirector = StreamRedirector(self)
        old_stdout  = sys.stdout
        sys.stdout  = redirector
        try:
            task_fn()
            self.after(0, self._set_status, "Done ✓", SUCCESS)
        except Exception as exc:
            self.after(0, self._log_raw, f"❌ Error: {exc}")
            self.after(0, self._set_status, "Failed ✗", ERROR_COL)
        finally:
            sys.stdout = old_stdout
            self.after(0, self._set_busy, False)

    # ── Module loader ────────────────────────────────────────────────────────

    def _load_module(self):
        if self._mod is None:
            self._mod = importlib.import_module(SCRIPT_MODULE)
        else:
            importlib.reload(self._mod)
        return self._mod

    # ── Git push helper (runs inside worker thread) ──────────────────────────

    def _git_push_steps(self, branch):
        r = self._git("add", "-A")
        if r.stdout.strip(): print(r.stdout.strip())

        r = self._git("commit", "-m", "publish: auto update from GUI")
        out = (r.stdout + r.stderr).strip()
        if out: print(out)
        # exit code 1 = "nothing to commit" — that's fine
        if r.returncode not in (0, 1):
            print(f"⚠ git commit exited with code {r.returncode}")

        r = self._git("push", "origin", branch)
        if r.stdout.strip(): print(r.stdout.strip())
        if r.stderr.strip(): print(r.stderr.strip())
        if r.returncode != 0:
            raise RuntimeError(f"git push failed (exit {r.returncode})")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _run_convert(self):
        def task():
            print("── Convert MD → HTML ──────────────────")
            m = self._load_module()
            m.build_image_map()
            m.copy_images()
            m.generate_index_pages()
            m.generate_material_pages()
            print("🎉 Conversion complete!")
        self._run_in_thread(task)

    def _run_push(self):
        branch = self.branch_var.get()
        if not branch or branch == NEW_BRANCH_OPTION:
            self._log("⚠ No branch selected.", "warning")
            return
        def task():
            print(f"── Push to GitHub → {branch} ───────────────")
            self._git_push_steps(branch)
            print(f"🎉 Push to '{branch}' complete!")
        self._run_in_thread(task)

    def _run_all(self):
        branch = self.branch_var.get()
        if not branch or branch == NEW_BRANCH_OPTION:
            self._log("⚠ No branch selected.", "warning")
            return
        def task():
            print("── Convert + Push ─────────────────────")
            m = self._load_module()
            m.build_image_map()
            m.copy_images()
            m.generate_index_pages()
            m.generate_material_pages()
            print("🎉 Conversion complete!")
            print(f"── Pushing to GitHub → {branch} ────────")
            self._git_push_steps(branch)
            print(f"🎉 All done — converted and pushed to '{branch}'!")
        self._run_in_thread(task)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")
        self._set_status("Ready", TEXT_DIM)
        self._log("Log cleared.", "dim")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PublisherApp()
    app.mainloop()
