# core/llammy_rag.py - Smart Learning RAG System with CSV Integration
# Llammy Framework v8.5 - The Brain That Learns and Adapts

import os
import csv
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import bpy

# RAG system imports
try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
    from llama_index.core.schema import Document
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

class LlammyRAG:
    """Smart Learning RAG System - The Brain of Llammy Framework
    
    This system:
    1. Reads your CSV learning data and extracts success patterns
    2. Maintains persistent memory across sessions
    3. Learns from every successful code generation
    4. Adapts to Blender API changes (4.4.1 → 5.0)
    5. Feeds intelligence back to your frontend model
    6. Builds contextual knowledge for better code generation
    """
    
    def __init__(self, addon_directory: str = None):
        self.addon_directory = addon_directory or self._get_addon_directory()
        self.initialized = False
        self.vector_index = None
        self.learning_data = []
        self.success_patterns = {}
        self.api_knowledge = {}
        self.user_preferences = {}
        self.blender_version = self._detect_blender_version()
        
        # File paths
        self.csv_path = os.path.join(self.addon_directory, "llammy_memory.csv")
        self.patterns_path = os.path.join(self.addon_directory, "data", "success_patterns.json")
        self.rag_data_dir = os.path.join(self.addon_directory, "data", "rag_data")
        self.vector_index_dir = os.path.join(self.rag_data_dir, "vector_index")
        self.api_cache_path = os.path.join(self.addon_directory, "data", "api_knowledge.json")
        
        # Learning configuration
        self.learning_config = {
            "pattern_confidence_threshold": 0.7,
            "min_success_samples": 3,
            "context_window_days": 30,
            "max_context_items": 10,
            "feedback_learning_rate": 0.1
        }
        
        # Blender API tracking for version changes
        self.api_evolution = {
            "4.4.1": {
                "material_nodes": "material.node_tree.nodes",
                "primitive_ops": "bpy.ops.mesh.primitive_",
                "context_active": "bpy.context.active_object"
            },
            "5.0": {
                # Will auto-detect and update when 5.0 patterns emerge
                "new_patterns": [],
                "deprecated_patterns": []
            }
        }
        
        print(f"🧠 LlammyRAG initialized for Blender {self.blender_version}")
    
    def _get_addon_directory(self) -> str:
        """Get the addon directory path"""
        return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    def _detect_blender_version(self) -> str:
        """Detect current Blender version"""
        try:
            return f"{bpy.app.version[0]}.{bpy.app.version[1]}.{bpy.app.version[2]}"
        except:
            return "4.4.1"  # Default fallback
    
    def initialize_enhanced(self) -> Tuple[bool, str]:
        """Initialize the enhanced RAG system with learning capabilities"""
        try:
            # Create directory structure
            os.makedirs(self.rag_data_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.patterns_path), exist_ok=True)
            
            # Load existing learning data
            self._load_csv_learning_data()
            self._load_success_patterns()
            self._load_api_knowledge()
            
            # Initialize vector store if LlamaIndex available
            if LLAMAINDEX_AVAILABLE:
                self._initialize_vector_store()
            
            # Analyze patterns and build intelligence
            self._analyze_success_patterns()
            self._build_contextual_knowledge()
            
            self.initialized = True
            
            stats = self._get_learning_stats()
            message = f"RAG Enhanced! {stats['total_entries']} entries, {stats['success_rate']:.1f}% success rate"
            
            print(f"✅ {message}")
            return True, message
            
        except Exception as e:
            error_msg = f"RAG initialization failed: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _load_csv_learning_data(self):
        """Load and process CSV learning data"""
        if not os.path.exists(self.csv_path):
            print("⚠️ No CSV learning data found - starting fresh")
            return
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.learning_data = list(reader)
            
            print(f"📊 Loaded {len(self.learning_data)} learning entries")
            
            # Process recent data for immediate insights
            self._process_recent_data()
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
    
    def _process_recent_data(self):
        """Process recent learning data for immediate insights"""
        if not self.learning_data:
            return
        
        # Get recent entries (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=self.learning_config["context_window_days"])
        
        recent_entries = []
        for entry in self.learning_data:
            try:
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if entry_date > cutoff_date:
                    recent_entries.append(entry)
            except:
                continue
        
        # Extract patterns from recent successes
        successful_recent = [e for e in recent_entries if e.get('success', '').lower() == 'true']
        
        for entry in successful_recent:
            self._extract_success_pattern(entry)
        
        print(f"🔍 Processed {len(successful_recent)} recent successful entries")
    
    def _extract_success_pattern(self, entry: Dict[str, Any]):
        """Extract success patterns from a learning entry"""
        user_input = entry.get('user_input', '')
        code = entry.get('code', '')
        model_info = entry.get('model_info', '')
        
        # Extract key patterns
        pattern_key = self._generate_pattern_key(user_input)
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = {
                "count": 0,
                "successful_approaches": [],
                "common_apis": [],
                "user_preferences": {},
                "model_performance": {}
            }
        
        pattern = self.success_patterns[pattern_key]
        pattern["count"] += 1
        
        # Extract API usage
        apis_used = self._extract_api_calls(code)
        pattern["common_apis"].extend(apis_used)
        
        # Track model performance
        if model_info:
            model_key = model_info.split('|')[0] if '|' in model_info else model_info
            if model_key not in pattern["model_performance"]:
                pattern["model_performance"][model_key] = 0
            pattern["model_performance"][model_key] += 1
        
        # Store successful approach
        if len(code) > 50:  # Only store substantial code
            pattern["successful_approaches"].append({
                "code_snippet": code[:500],  # First 500 chars
                "timestamp": entry.get('timestamp', ''),
                "context": user_input[:200]