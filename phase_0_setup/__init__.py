# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\phase_0_setup\__init__.py

"""
Legal Assistant AI - Phase 0 Setup Package
Environment setup and validation tools for the Legal Assistant AI project
"""

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"

# Import main classes for easy access
from .environment_setup import EnvironmentSetup
from .validate_setup import SetupValidator

# Re-export for convenience
__all__ = [
    'EnvironmentSetup',
    'SetupValidator'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'Phase 0 Setup Tools',
    'description': 'Environment setup and validation for Legal Assistant AI',
    'version': __version__,
    'components': [
        'EnvironmentSetup - Automated environment configuration',
        'SetupValidator - Comprehensive environment validation'
    ]
}