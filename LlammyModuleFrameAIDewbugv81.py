# core/llammy_debug.py - AI-Powered Debug System with Persistent Memory
# Llammy Framework v8.5 - The Self-Learning Debug Brain

import os
import json
import time
import traceback
import re
import http.client
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import bpy

class AIDebugSystem:
    """AI-Powered Debug System with Persistent Memory
    
    This system:
    1. Uses your existing models (technical/creative) when available
    2. Falls back to direct Ollama call for debug analysis
    3. Maintains persistent memory of all fixes applied
    4. Learns from successful/failed fix attempts
    5. Builds a knowledge base of error patterns and solutions
    6. Provides intelligent fix suggestions based on historical data
    7. Feeds learning back to the main RAG system
    """
    
    def __init__(self, addon_directory: str = None):
        self.addon_directory = addon_directory or self._get_addon_directory()
        self.initialized = False
        
        # Model integration
        self.model_manager = None  # Will be set by framework
        self.use_framework_models = True  # Prefer framework models
        self.fallback_to_ollama = True  # Fallback option
        
        # Persistent memory storage
        self.debug_memory_path = os.path.join(self.addon_directory, "data", "debug_memory.json")
        self.error_patterns_path = os.path.join(self.addon_directory, "data", "error_patterns.json")
        self.fix_history_path = os.path.join(self.addon_directory, "data", "fix_history.json")
        
        # In-memory caches
        self.debug_memory = {}
        self.error_patterns = {}
        self.fix_history = []
        self.active_fixes = {}  # Currently applying fixes
        
        # Configuration
        self.config = {
            "max_fix_attempts": 3,
            "ollama_timeout": 30,
            "debug_model": "qwen2.5:7b",  # Fallback debug model
            "memory_retention_days": 90,
            "confidence_threshold": 0.7,
            "max_memory_entries": 1000,
            "prefer_framework_models": True,
            "enable_ollama_fallback": True
        }
        
        # AI Analysis Templates
        self.analysis_templates = {
            "error_analysis": """You are an expert Blender Python debugger with deep knowledge of the Blender API.

ERROR ANALYSIS REQUEST:
Error Type: {error_type}
Error Message: {error_message}
Code Context: {code_context}
User Intent: {user_intent}
Blender Version: {blender_version}

HISTORICAL CONTEXT:
{historical_context}

Please analyze this error and provide:
1. ROOT_CAUSE: What exactly caused this error?
2. FIX_STRATEGY: Specific steps to fix it
3. PREVENTION: How to avoid this in the future
4. CONFIDENCE: How confident you are (0.0-1.0)
5. BLENDER_SPECIFIC: Any Blender API specific considerations

Format your response as JSON:
{{
    "root_cause": "detailed explanation",
    "fix_strategy": "specific fix steps",
    "prevention": "prevention advice",
    "confidence": 0.0,
    "blender_specific": "API considerations",
    "suggested_code": "corrected code if applicable"
}}""",
            
            "fix_validation": """You are a Blender Python code validator.

VALIDATION REQUEST:
Original Code: {original_code}
Fixed Code: {fixed_code}
Error Context: {error_context}
Fix