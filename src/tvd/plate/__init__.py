"""Optional license plate recognition (ANPR) with privacy controls.

Off by default. When enabled, plate text is OCR'd but NOT stored unless
`store_plate_text: true` is explicitly set in configs/extensions.yaml.
"""

from tvd.plate.anpr import PlateRead, PlateReader

__all__ = ["PlateRead", "PlateReader"]
