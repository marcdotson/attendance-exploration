# Attendance Exploration


The school district has asked us to explore the current state of
attendance. Questions include:

- Is there a hinge point (or multiple) for academic achievement with
  attendance? Does it vary by grade or other demographic subgroup?
- For students that transition from poor to good attendance, are there
  correlates we can identify like schools, grades, classes taken,
  teachers, etc. that could be possible reasons for the change?
- Which demographic subgroups are most impacted with poor attendance and
  does poor attendance impact those groups equally in terms of academic
  achievement (gpa, test scores, etc.)?
- Are there long term impacts of consistently poor attendance (Chronic
  Absenteeism) on things like credit, graduation, AP/CE enrollment, etc.

## Project Organization

- `/code` Scripts with prefixes (e.g., `01_import-data.py`,
  `02_clean-data.py`) and functions in `/code/src`.
- `/data` Simulated and real data, the latter not pushed.
- `/figures` PNG images and plots.
- `/output` Output from model runs, not pushed.
- `/private` A catch-all folder for miscellaneous files, not pushed.
- `/.quarto` Hidden Quarto project library, not pushed.
- `/.venv` Hidden Python project library, not pushed.
- `.gitignore` Hidden Git instructions file.
- `.python-version` Hidden Python version file.
- `LICENSE` MIT License for “as is” permission.
- `README.md` GitHub-flavored markdown README rendered from
  `README.qmd`.
- `README.qmd` Quarto markdown README to edit and render into
  `README.md`.
- `_quarto.yml` Quarto project configuration file.
- `pyproject.toml` Python project environment configuration file.
- `uv.lock` Python project environment lockfile.

## Project Environment

After cloning this repository, go to the project’s terminal in Positron
and run `uv run` to create the `/.venv` project library and install the
specified Python and library versions.

For more details on using Python, Positron, GitHub, Quarto, etc. see the
recommended [data stack](https://github.com/marcdotson/data-stack).
