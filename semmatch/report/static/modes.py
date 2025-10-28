from enum import Enum, auto


class ReportMode(Enum):
    SHOW_ALL_RESULTS = 0
    SHOW_RELEVANT_RESULTS = auto()
    SHOW_SUMMARY_ONLY = auto()
    SHOW_ERRORS_ONLY = auto()


SHOW_ALL_RESULTS = "show_all_results"
SHOW_RELEVANT_RESULTS = "show_relevant_results"
SHOW_SUMMARY_ONLY = "show_summary_only"
# SHOW_TOP_N_RESULTS = "show_top_n_results"
# SHOW_ERRORS_ONLY = "show_errors_only"
# SHOW_WARNINGS_ONLY = "show_warnings_only"
# SHOW_SUCCESS_ONLY = "show_success_only"
SHOW_COMPARISON = "show_comparison"
