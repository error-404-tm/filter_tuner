"""Dockable side panels. Each one talks only to the Session."""

from .design_panel import DesignPanel
from .export_panel import ExportPanel
from .filter_settings import FilterSettingsPanel
from .mask_panel import MaskPanel
from .math_panel import MathPanel
from .quant_panel import QuantPanel
from .root_editor import RootEditor
from .signal_panel import SignalPanel

__all__ = [
    "DesignPanel", "ExportPanel", "FilterSettingsPanel", "MaskPanel",
    "MathPanel", "QuantPanel", "RootEditor", "SignalPanel",
]
