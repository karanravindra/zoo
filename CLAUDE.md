# zoo

Personal ML experiments. Shared library code lives in `src/zoo/`; each experiment is a project under `projects/<name>/`.

## Running code

- Never run training, embedding, download, or other pipeline scripts unless the user says it's okay. That okay lasts for the rest of the session. Without it, write the code and hand it off with the exact command for the user to run.
- Cheap checks don't need permission: imports, `--help`, a tiny CPU forward pass.
- Use uv for everything: `uv run ...` and `uv add ...`. Never use pip.

## Layout

- Code reused across projects goes in `src/zoo/` (`zoo.data` for datasets, `zoo.utils` for device/seeding etc.).
- Each project lives in `projects/<name>/` with a `README.md` and is otherwise self-contained. Keep the README in sync when scripts, outputs, or configs change.

## Script config

- Every runnable script builds its settings from a dataclass and parses them with [tyro](https://brentyi.github.io/tyro/), so the dataclass fields define the CLI flags. Don't use argparse.
- Pattern: `main(tyro.cli(Config))`. Defaults live in the dataclass, and a field's trailing comment is its `--help` text.
- Training configs are saved into checkpoints, so they should stay plain, serializable dataclasses.

## Run outputs

- Each run writes to `runs/<project>/<timestamp>/` (gitignored): `config.json` (config plus git commit and device), `metrics.csv`, `best.pt` / `last.pt` checkpoints, and `results.json`.
- Save plots as PNG files. Never call `plt.show()`.

## Git

- Use atomic conventional commits, with the project as the scope when there is one: `feat(pixel_art): ...`, `docs(digitnet): ...`, `build: ...`.
- Commit only when asked.
- Commits are SSH-signed. Don't use history-rewriting tools that strip signatures (e.g. `git filter-branch`); if you rewrite history, re-sign it.
