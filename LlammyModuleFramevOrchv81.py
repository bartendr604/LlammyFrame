# core/llammy_models.py - Universal AI Model Manager
# Llammy Framework v8.5 - The AI Brain Connector

import http.client
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import bpy

class ModelManager:
    """Universal AI Model Manager - The Gateway to All AI Models
    
    This system:
    1. Manages connections to multiple AI backends (Ollama, Claude, OpenAI, Gemini)
    2. Handles model discovery and caching
    3. Provides intelligent model selection and fallback
    4. Tracks model performance and reliability
    5. Optimizes API calls for cost and speed
    6. Maintains connection health monitoring
    """
    
    def __init__(self):
        self.initialized = False
        self.cached_models = {}
        self.model_performance = {}
        self.connection_status = {}
        self.api_keys = {}
        self.last_refresh = {}
        
        # Configuration
        self.config = {
            "cache_duration_minutes": 30,
            "connection_timeout": 10,
            "max_retries": 2,
            "enable_performance_tracking": True,
            "enable_auto_fallback": True,
            "refresh_interval_seconds": 300,  # 5 minutes
            "rate_limit_buffer": 1.0  # 1 second between calls
        }
        
        # Backend configurations
        self.backends = {
            "ollama": {
                "name": "Ollama (Local)",
                "description": "Local AI models via Ollama",
                "host": "localhost",
                "port": 11434,
                "requires_api_key": False,
                "supports_streaming": True,
                "cost_per_call": 0.0,
                "typical_speed": "fast"
            },
            "claude": {
                "name": "Claude (Anthropic)",
                "description": "Anthropic's Claude models",
                "host": "api.anthropic.com",
                "port": 443,
                "requires_api_key": True,
                "supports_streaming": False,
                "cost_per_call": 0.01,  # Estimated
                "typical_speed": "medium"
            },
            "openai": {
                "name": "OpenAI GPT",
                "description": "OpenAI's GPT models",
                "host": "api.openai.com",
                "port": 443,
                "requires_api_key": True,
                "supports_streaming": True,
                "cost_per_call": 0.02,  # Estimated
                "typical_speed": "fast"
            },
            "gemini": {
                "name": "Google Gemini",
                "description": "Google's Gemini models",
                "host": "generativelanguage.googleapis.com",
                "port": 443,
                "requires_api_key": True,
                "supports_streaming": False,
                "cost_per_call": 0.005,  # Estimated
                "typical_speed": "medium"
            }
        }
        
        # Model categorization
        self.model_categories = {
            "creative": ["claude", "gpt-4", "gemini-pro"],
            "technical": ["qwen", "codellama", "deepseek", "gpt-3.5-turbo"],
            "debug": ["qwen2.5:7b", "codellama:7b", "deepseek-coder"],
            "general": ["llama3", "mistral", "phi3"]
        }
        
        # Performance tracking
        self.performance_metrics = {
            "response_times": {},
            "success_rates": {},
            "error_counts": {},
            "last_used": {}
        }
        
        print("🤖 ModelManager initialized - Ready to connect to AI backends!")
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize the model manager with backend discovery"""
        try:
            # Test all backend connections
            self._test_all_connections()
            
            # Refresh model lists
            self._refresh_all_models()
            
            # Load performance history
            self._load_performance_history()
            
            self.initialized = True
            
            # Generate status message
            active_backends = [name for name, status in self.connection_status.items() if status.get("connected", False)]
            total_models = sum(len(models) for models in self.cached_models.values())
            
            message = f"Initialized! {len(active_backends)} backends, {total_models} models available"
            print(f"✅ {message}")
            
            return True, message
            
        except Exception as e:
            error_msg = f"Model manager initialization failed: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _test_all_connections(self):
        """Test connections to all configured backends"""
        for backend_name, backend_config in self.backends.items():
            try:
                success, message = self._test_backend_connection(backend_name)
                self.connection_status[backend_name] = {
                    "connected": success,
                    "message": message,
                    "last_tested": datetime.now().isoformat()
                }
                
                if success:
                    print(f"✅ {backend_name}: {message}")
                else:
                    print(f"❌ {backend_name}: {message}")
                    
            except Exception as e:
                self.connection_status[backend_name] = {
                    "connected": False,
                    "message": f"Connection test failed: {str(e)}",
                    "last_tested": datetime.now().isoformat()
                }
    
    def _test_backend_connection(self, backend_name: str) -> Tuple[bool, str]:
        """Test connection to a specific backend"""
        backend_config = self.backends[backend_name]
        
        if backend_name == "ollama":
            return self._test_ollama_connection()
        elif backend_name == "claude":
            return self._test_claude_connection()
        elif backend_name == "openai":
            return self._test_openai_connection()
        elif backend_name == "gemini":
            return self._test_gemini_connection()
        else:
            return False, f"Unknown backend: {backend_name}"
    
    def _test_ollama_connection(self) -> Tuple[bool, str]:
        """Test Ollama connection"""
        try:
            conn = http.client.HTTPConnection("localhost", 11434, timeout=self.config["connection_timeout"])
            conn.request("GET", "/api/tags")
            response = conn.getresponse()
            result = response.read().decode()
            conn.close()
            
            if response.status == 200:
                data = json.loads(result)
                model_count = len(data.get("models", []))
                return True, f"Connected - {model_count} models available"
            else:
                return False, f"HTTP {response.status} error"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def _test_claude_connection(self) -> Tuple[bool, str]:
        """Test Claude connection"""
        api_key = self.api_keys.get("claude", "")
        if not api_key:
            return False, "No API key configured"
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-5-haiku-20241022',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'test'}]
            }
            
            request = urllib.request.Request(
                'https://api.anthropic.com/v1/messages',
                data=json.dumps(data).encode('utf-8'),
                headers=headers
            )
            
            with urllib.request.urlopen(request, timeout=self.config["connection_timeout"]) as response:
                if response.status == 200:
                    return True, "Connected and authenticated"
                else:
                    return False, f"HTTP {response.status}"
                    
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid API key"
            elif e.code == 403:
                return False, "Access denied"
            else:
                return False, f"HTTP {e.code} error"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def _test_openai_connection(self) -> Tuple[bool, str]:
        """Test OpenAI connection"""
        api_key = self.api_keys.get("openai", "")
        if not api_key:
            return False, "No API key configured"
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': 'test'}],
                'max_tokens': 10
            }
            
            request = urllib.request.Request(
                'https://api.openai.com/v1/chat/completions',
                data=json.dumps(data).encode('utf-8'),
                headers=headers
            )
            
            with urllib.request.urlopen(request, timeout=self.config["connection_timeout"]) as response:
                if response.status == 200:
                    return True, "Connected and authenticated"
                else:
                    return False, f"HTTP {response.status}"
                    
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid API key"
            else:
                return False, f"HTTP {e.code} error"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def _test_gemini_connection(self) -> Tuple[bool, str]:
        """Test Gemini connection"""
        api_key = self.api_keys.get("gemini", "")
        if not api_key:
            return False, "No API key configured"
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            
            data = {
                'contents': [{'parts': [{'text': 'test'}]}]
            }
            
            request = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(request, timeout=self.config["connection_timeout"]) as response:
                if response.status == 200:
                    return True, "Connected and authenticated"
                else:
                    return False, f"HTTP {response.status}"
                    
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return False, "Invalid API key"
            else:
                return False, f"HTTP {e.code} error"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def _refresh_all_models(self):
        """Refresh model lists from all connected backends"""
        for backend_name in self.backends.keys():
            if self.connection_status.get(backend_name, {}).get("connected", False):
                try:
                    models = self._get_models_for_backend(backend_name)
                    self.cached_models[backend_name] = models
                    self.last_refresh[backend_name] = datetime.now()
                    print(f"🔄 Refreshed {len(models)} models from {backend_name}")
                except Exception as e:
                    print(f"⚠️ Failed to refresh models from {backend_name}: {e}")
    
    def _get_models_for_backend(self, backend_name: str) -> List[Tuple[str, str, str]]:
        """Get available models for a specific backend"""
        if backend_name == "ollama":
            return self._get_ollama_models()
        elif backend_name == "claude":
            return self._get_claude_models()
        elif backend_name == "openai":
            return self._get_openai_models()
        elif backend_name == "gemini":
            return self._get_gemini_models()
        else:
            return []
    
    def _get_ollama_models(self) -> List[Tuple[str, str, str]]:
        """Get Ollama models"""
        try:
            conn = http.client.HTTPConnection("localhost", 11434, timeout=10)
            conn.request("GET", "/api/tags")
            response = conn.getresponse()
            result = response.read().decode()
            conn.close()
            
            if response.status != 200:
                return []
            
            data = json.loads(result)
            models = []
            
            for model in data.get("models", []):
                model_name = model.get("name", "")
                display_name = model_name.replace(":latest", "")
                size_info = model.get("size", 0)
                size_gb = size_info / (1024**3) if size_info else 0
                
                # Categorize model
                model_lower = model_name.lower()
                if any(keyword in model_lower for keyword in ['qwen', 'codellama', 'deepseek']):
                    category = "[TECHNICAL]"
                elif any(keyword in model_lower for keyword in ['llama3', 'mistral', 'phi3']):
                    category = "[GENERAL]"
                else:
                    category = "[GENERAL]"
                
                description = f"{category} {display_name}"
                if size_gb > 0:
                    description += f" ({size_gb:.1f}GB)"
                
                models.append((model_name, display_name, description))
            
            return models
            
        except Exception as e:
            print(f"❌ Error fetching Ollama models: {e}")
            return []
    
    def _get_claude_models(self) -> List[Tuple[str, str, str]]:
        """Get Claude models"""
        return [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "[CREATIVE] Latest Claude - Best for creative tasks"),
            ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "[TECHNICAL] Fast Claude - Good for technical tasks"),
            ("claude-3-opus-20240229", "Claude 3 Opus", "[CREATIVE] Most capable - Premium creative model")
        ]
    
    def _get_openai_models(self) -> List[Tuple[str, str, str]]:
        """Get OpenAI models"""
        return [
            (