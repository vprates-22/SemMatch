**SemMatch**

SemMatch is a toolbox for evaluating image matching pipelines. It provides modular components for data generation, analysis, metric computation, visualization, and report generation (HTML/PDF/Flask server). The project aims to make it easy to run reproducible experiments and compare metrics across datasets such as HPatches, MegaDepth and ScanNet.

**Key features**
- **Modularity**: clear separation of `data_generators`, `analyzers`, and `metrics` for easy composition.
- **Pipeline orchestration**: build repeatable experiments using `PipelineOrchestrator` and `AnalysisPlan`.
- **Reports & visualization**: generate HTML/PDF reports and serve interactive visualizations through a Flask server.
- **Supported datasets**: includes adapters and utilities for HPatches, MegaDepth, and ScanNet (see `data/`).

**Repository layout**
- `semmatch/`: core code (configs, datasets, evaluation, report, statistics, utils).
- `data/`: sample datasets and pair lists used for evaluation (`hpatches/`, `megaDepth/`, `scannet/`).
- `tests/`: unit tests (run with `pytest`).
- `requirements.txt`: Python dependencies to reproduce the environment.

**Quick start (install)**

Clone the repository and create a virtual environment (macOS / zsh):

```bash
git clone https://github.com/<your-username>/SemMatch.git
cd SemMatch
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Note: some packages (e.g., WeasyPrint, pycairo) require system libraries on macOS. Install them via Homebrew if needed, for example:

```bash
brew install cairo pango gdk-pixbuf libffi
```

**Usage examples**

1) Run tests

```bash
pytest -q
```

2) Programmatic example using `PipelineOrchestrator`

```python
from semmatch.statistics.orchestrator import AnalysisPlan, PipelineOrchestrator

# Replace these placeholders with concrete classes from the codebase
# Example: MyAnalyzer could be a class from semmatch.statistics.analyzers
# and MyMetricA/MyMetricB subclasses of semmatch.statistics.metrics.base.BaseMetric

plan = [
    AnalysisPlan(title="Example Analysis",
                 analysis=MyAnalyzer,
                 metrics=[MyMetricA, MyMetricB])
]

config = {"dataset": "hpatches", "output_dir": "results/"}

orchestrator = PipelineOrchestrator(plan=plan, config=config)

# For each image pair (RawDataInput) do:
# orchestrator.execute(raw_data)

# Finalize and retrieve results:
orchestrator.summarize()
results = orchestrator.get_results()
print(results)
```

3) Start the report server (visualization)

```bash
python -c "from semmatch.report.dynamic.server import run_report; run_report()"
```

This starts a Flask app on `http://localhost:5000` by default and exposes result and visualization routes.

**How to structure an experiment**
- Define one or more `AnalysisPlan` entries for each analysis stage.
- Implement `DataGenerator` classes to produce auxiliary data (depth maps, descriptors, matches).
- Implement `DataAnalyzer` to convert generated data into `AnalysisResult` objects.
- Implement `BaseMetric` subclasses to aggregate and compute per-pair metrics.
- Use `PipelineOrchestrator` to iterate over image pairs, update metrics, and collect results.

**High-level architecture**
- `semmatch.datasets` — dataset adapters and utilities (HPatches, MegaDepth, ScanNet).
- `semmatch.statistics` — data generators, analyzers, metrics and pipeline orchestration.
- `semmatch.evaluation` — evaluation logic and validators.
- `semmatch.report` — report generation (HTML/PDF) and Flask-based visualization server.
- `semmatch.utils` — helper utilities (I/O, image processing, geometry, validation).

**Contributing**
- Open issues for bugs or feature requests.
- For code contributions: create a feature branch, include tests, and submit a descriptive PR.
- Keep changes small and consistent with the existing code style.

**Practical notes**
- Check `semmatch/statistics/orchestrator.py` for an example of how `AnalysisPlan` and `PipelineOrchestrator` are used.
- Reproduce the Python environment with `requirements.txt`; install OS-level dependencies for PDF/HTML generation if required.

**References**
- Datasets and resources: HPatches, MegaDepth, ScanNet.

---
