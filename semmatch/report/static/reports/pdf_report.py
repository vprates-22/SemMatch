from datetime import datetime

from xhtml2pdf import pisa

from semmatch.report.static.modes import ReportMode
from semmatch.report.static.base_static_report import BaseStaticReport
from semmatch.report.static.reports.html_report import HTMLReport


class PDFReport(BaseStaticReport):
    def __init__(self, orchestrator, mode=ReportMode.SHOW_SUMMARY_ONLY):
        super().__init__(orchestrator, mode)

        self.html_report = HTMLReport(orchestrator, mode)

    def generate_report(self):
        html_source = self.html_report.generate_report(False)

        time_str = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        with open(f"report {time_str}.pdf", "w+b") as result_file:
            pisa.CreatePDF(
                html_source,
                dest=result_file,
            )
