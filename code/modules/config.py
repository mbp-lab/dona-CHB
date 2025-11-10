# config.py
from pathlib import Path

# Where to save numbers from the analysis
data_reports_path = Path('../../reports/numbers_updated.dat')
data_reports_path.parent.mkdir(parents=True, exist_ok=True)
if not data_reports_path.exists():
    data_reports_path.touch()

# Where to save figures from the analysis
figpath = Path('../../reports')

# Default colors
pre_color = 'gray'
post_color = 'slateblue'
default_color = 'blue'
