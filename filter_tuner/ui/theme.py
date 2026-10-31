"""Shared colors and pens so every view draws zeros/poles the same way."""

from pyqtgraph.Qt import QtCore

ZERO_COLOR = (90, 160, 255)
POLE_COLOR = (255, 90, 90)
SELECT_COLOR = (255, 220, 80)
QUANT_COLOR = (255, 160, 40)
MASK_COLOR = (255, 80, 80, 200)
CURVE_COLOR = (230, 230, 230)
STEP_COLOR = (120, 200, 140)

DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine

GOOD_STYLE = "font-weight: bold; color: #3c3;"
BAD_STYLE = "font-weight: bold; color: #e44;"
