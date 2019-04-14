"""
Metrics used for evaluation.

For more competition-related metrics (detection F1, object dice, etc),
please refer to: https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/
"""

from segmentation_models.metrics import iou_score
from segmentation_models.metrics import dice_score
