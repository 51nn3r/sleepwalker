# Sleepwalker

Dreamer with request-actions: an actor (a small LLM) solves tasks by delegating sub-tasks recursively; a world model
over a discrete text window, a value model (VM) trained pairwise from an LLM judge, Shapley credit assignment
(VM game → reward-model labels; critic game → actor credit), actor learning in imagination. Thesis prototype.

Files
- `sleepwalker.py` — the whole pipeline (also the notebook source; cells are marked with `# %%`).
- `build_notebook.py` — builds `sleepwalker.ipynb` (fast preset) and `sleepwalker_pro.ipynb` (pro) for Colab;
  `OPENROUTER_API_KEY=... python3 build_notebook.py sleepwalker.py [--preset pro]` embeds the key into the notebook
  (notebooks are git-ignored for that reason).
- `run_server.sh` — run on your own GPU server; see the header for the one-GPU and two-GPU recipes.
- `docs/` — model description and the judge design (Russian).

Local self-check without GPU, models or network: `DRY_RUN=1 python3 sleepwalker.py` (stubs, ~5 min).

Outputs: `sleepwalker_runs/<run>/logs/` (log, config, results, trajectories, judge verdicts, metrics, `status.json`)
and `sleepwalker_runs/<run>/weights/`. `README.txt` inside a run folder explains every file.
