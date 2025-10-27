from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from semmatch.report.static.modes import ReportMode
from semmatch.report.static.base_static_report import BaseStaticReport


class HTMLReport(BaseStaticReport):
    def __init__(self, orchestrator, mode=ReportMode.SHOW_SUMMARY_ONLY):
        super().__init__(orchestrator, mode)

        env = Environment(loader=FileSystemLoader('.'))
        self.template = env.get_template('template/index.html')

    def generate_report(self):
        tables = []
        summary = self.generate_summary_table()

        tables.append({
            "title": "Summary",
            "headers": summary.columns.tolist(),
            "rows": summary.values.tolist()
        })

        if self.mode == ReportMode.SHOW_ALL_RESULTS:
            pair_metrics = self.generate_pairs_table()
            tables.append({
                "title": "Pair Metrics",
                "headers": pair_metrics.columns.tolist(),
                "rows": pair_metrics.values.tolist()
            })

        time_str = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        html = self.template.render(
            title="Validation Report",
            subtitle=time_str,
            tables=tables,
            # images=images
        )


        file_name = f"report {time_str}.html"
        with open(file_name, "w") as f:
            f.write(html)

        return file_name