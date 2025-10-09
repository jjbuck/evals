"""
SessionConverter - Base class for converting telemetry data to Session format
"""
from abc import ABC, abstractmethod
from typing import List, Any

from ..types.trace import Session


class SessionConverter(ABC):
    """Base class for converting telemetry data to Session format for evaluation."""
    
    @abstractmethod
    def convert(self, spans: List[Any], session_id: str) -> Session:
        """
        Convert spans to Session format.
        
        Args:
            spans: List of span objects
            session_id: Session identifier
            
        Returns:
            Session object ready for evaluation
        """
        pass
