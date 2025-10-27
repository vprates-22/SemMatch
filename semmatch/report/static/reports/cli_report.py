from semmatch.report.static.modes import ReportMode
from semmatch.report.static.base_static_report import BaseStaticReport


class CLIReport(BaseStaticReport):
    def __init__(self, orchestrator, mode=ReportMode.SHOW_SUMMARY_ONLY):
        super().__init__(orchestrator, mode)

    def generate_report(self):
        print(self.generate_summary_table())

        if self.mode == ReportMode.SHOW_ALL_RESULTS:
            print(self.generate_pairs_table())
