from pathlib import Path

BASE_PATH = Path(__file__).parent.parent

DATA_PATH = BASE_PATH / 'data/'
OUTPUT_PATH = BASE_PATH / 'output/'

RESULTS_PATH = OUTPUT_PATH / 'reports/'
MATCHES_PATH = OUTPUT_PATH / 'matches/'
VISUALIZATIONS_PATH = OUTPUT_PATH / 'visualizations/'