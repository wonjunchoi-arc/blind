"""
BlindInsight AI - Company Analysis and Career Guidance System

A sophisticated AI system that analyzes anonymous employee insights 
to provide actionable career guidance and company intelligence.
"""

__version__ = "1.0.0"
__author__ = "BlindInsight Team"
__email__ = "team@blindinsight.ai"

from .models import *
from .agents import *
from .services import *

__all__ = [
    "BlindInsightSystem",
    "AnalysisRequest", 
    "AnalysisResponse",
    "CompanyReport",
    "CareerGuidance"
]