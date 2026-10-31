"""Backward-compatible launcher. The app now lives in the filter_tuner package:

    python -m filter_tuner [project.json]
"""

import sys

from filter_tuner.app import main

if __name__ == "__main__":
    sys.exit(main())
