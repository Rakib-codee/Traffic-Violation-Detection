"""Evidence building: event clips, annotated frames, metadata."""

from tvd.evidence.blur import blur_plate_band, blur_regions
from tvd.evidence.builder import EvidenceBuilder, EvidenceItem
from tvd.evidence.db import EventStore

__all__ = ["EvidenceBuilder", "EvidenceItem", "EventStore", "blur_plate_band", "blur_regions"]
