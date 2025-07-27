# core/llammy_models_realtime.py - Real-Time Model Discovery
# Llammy Framework v8.5 - Live Model Detection

import http.client
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import bpy
import re

class RealTimeModelManager:
    """Real-Time Model Manager - Shows Only What You Actually Have
    
    No bullshit hardcoded lists. This system:
    1. LIVE queries Ollama for installed models
    2. LIVE tests API keys for cloud models  
    3. Categorizes based on actual model names
    4. Shows real memory/disk requirements
    5. Updates availability in real-time
    """
    
    def __init__(self):
        self.initialized = False
        self.live_models = {}  # What's actually available RIGHT NOW
        self.model_specs = {}  # Real specs (size, memory, etc.)
        self.connection_health = {}
        self.api_keys = {}
        self.last_scan = None
        self.scan_interval = 30  # seconds
        
        # Dynamic categorization patterns
        self.category_patterns = {
            "creative": {
                "keywords": ["claude", "llama3", "llama-3", "vicuna"],
                "models": []  # Populated dynamically
            },
            "technical": {
                "keywords": ["qwen", "codellama", "code-llama", "deepseek", "coder"],
                "models": []
            },
            "debug": {
                "keywords": ["7b", "small", "debug", "mini"],
                "models": []
            },
            "general": {
                "keywords": ["llama", "phi", "gemma", "mistral"],
                "models": []
            }
        }
        
        print("🔍 Real-Time Model Manager initialized - Live discovery mode!")
    
    def scan_available_models(self, force_refresh: bool = False) -> Tuple[bool, str]:
        """Scan for actually available models RIGHT NOW"""
        
        # Check if we need to refresh
        if not force_refresh and self.last_scan:
            time_since_scan = (datetime.now() - self.last_scan).total_seconds()
            if time_since_scan < self.scan_interval:
                return True, f"Using cached scan from {int(time_since_scan)}s ago"
        
        print("🔍 Scanning for live models...")
        
        self.live_models = {}
        self.model_specs = {}
        
        # 1. Scan Ollama (local models)
        ollama_success = self._scan_ollama_live()
        
        # 2. Test Claude API
        claude_success = self._test_claude_live()
        
        # 3. Test Gemini API  
        gemini_success = self._test_gemini_live()
        
        # 4. Categorize discovered models
        self._categorize_discovered_models()
        
        # 5. Update health status
        self._update_health_status()
        
        self.last_scan = datetime.now()
        
        # Generate report
        total_models = sum(len(models) for models in self.live_models.values())
        active_backends = len([k for k, v in self.live_models.items() if v])
        
        message = f"Live scan complete: {total_models} models from {active_backends} backends"
        print(f"✅ {message}")
        
        return True, message
    
    def _scan_ollama_live(self) -> bool:
        """Scan Ollama for actually installed models"""
        try:
            conn = http.client.HTTPConnection("localhost", 11434, timeout=5)
            conn.request("GET", "/api/tags")
            response = conn.getresponse()
            result = response.read().decode()
            conn.close()
            
            if response.status != 200:
                self.live_models["ollama"] = []
                return False
            
            data = json.loads(result)
            ollama_models = []
            
            for model in data.get("models", []):
                model_name = model.get("name", "")
                size_bytes = model.get("size", 0)
                modified_at = model.get("modified_at", "")
                
                # Extract real specs
                size_gb = size_bytes / (1024**3) if size_bytes else 0
                
                # Get model family and size from name
                family, size_param = self._parse_model_name(model_name)
                
                # Create model entry
                model_entry = {
                    "id": model_name,
                    "name": model_name.replace(":latest", ""),
                    "family": family,
                    "size_gb": size_gb,
                    "size_param": size_param,
                    "backend": "ollama",
                    "available": True,
                    "last_modified": modified_at,
                    "memory_required": self._estimate_memory_requirement(size_gb),
                    "description": self._generate_model_description(model_name, size_gb, size_param)
                }
                
                ollama_models.append(model_entry)
                
                # Store specs separately
                self.model_specs[model_name] = {
                    "size_gb": size_gb,
                    "memory_gb": self._estimate_memory_requirement(size_gb),
                    "family": family,
                    "parameters": size_param
                }
            
            self.live_models["ollama"] = ollama_models
            print(f"📦 Ollama: Found {len(ollama_models)} installed models")
            return True
            
        except Exception as e:
            print(f"❌ Ollama scan failed: {e}")
            self.live_models["ollama"] = []
            return False
    
    def _parse_model_name(self, model_name: str) -> Tuple[str, str]:
        """Parse model name to extract family and parameter size"""
        name_lower = model_name.lower()
        
        # Extract family
        if "qwen" in name_lower:
            family = "qwen"
        elif "llama" in name_lower:
            family = "llama"
        elif "codellama" in name_lower or "code-llama" in name_lower:
            family = "codellama"
        elif "deepseek" in name_lower:
            family = "deepseek"
        elif "phi" in name_lower:
            family = "phi"
        elif "gemma" in name_lower:
            family = "gemma"
        elif "mistral" in name_lower:
            family = "mistral"
        else:
            family = "unknown"
        
        # Extract parameter size
        size_match = re.search(r'(\d+\.?\d*)b', name_lower)
        if size_match:
            size_param = size_match.group(1) + "B"
        else:
            size_param = "unknown"
        
        return family, size_param
    
    def _estimate_memory_requirement(self, size_gb: float) -> float:
        """Estimate RAM requirement for model"""
        # Rule of thumb: model needs 1.2-1.5x its size in RAM
        return size_gb * 1.3
    
    def _generate_model_description(self, model_name: str, size_gb: float, size_param: str) -> str:
        """Generate dynamic description based on actual model"""
        name_lower = model_name.lower()
        
        # Determine category
        if any(keyword in name_lower for keyword in ["qwen", "codellama", "deepseek", "coder"]):
            category = "[TECHNICAL]"
        elif any(keyword in name_lower for keyword in ["llama3", "llama-3"]):
            category = "[CREATIVE]"
        elif "7b" in name_lower or "small" in name_lower:
            category = "[DEBUG]"
        else:
            category = "[GENERAL]"
        
        description = f"{category} {model_name.replace(':latest', '')}"
        
        if size_gb > 0:
            description += f" ({size_gb:.1f}GB, ~{self._estimate_memory_requirement(size_gb):.1f}GB RAM)"
        
        if size_param != "unknown":
            description += f" [{size_param} params]"
        
        return description
    
    def _test_claude_live(self) -> bool:
        """Test Claude API and get available models"""
        api_key = self.api_keys.get("claude", "")
        if not api_key:
            self.live_models["claude"] = []
            return False
        
        try:
            # Test with smallest model first
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-5-haiku-20241022',
                'max_tokens': 5,
                'messages': [{'role': 'user', 'content': 'hi'}]
            }
            
            request = urllib.request.Request(
                'https://api.anthropic.com/v1/messages',
                data=json.dumps(data).encode('utf-8'),
                headers=headers
            )
            
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    # API key works, list available models
                    claude_models = [
                        {
                            "id": "claude-3-5-sonnet-20241022",
                            "name": "Claude 3.5 Sonnet",
                            "family": "claude",
                            "backend": "claude",
                            "available": True,
                            "cost_per_1k": 0.003,  # Real pricing
                            "description": "[CREATIVE] Claude 3.5 Sonnet - Latest & greatest",
                            "capabilities": ["reasoning", "creative", "code"]
                        },
                        {
                            "id": "claude-3-5-haiku-20241022", 
                            "name": "Claude 3.5 Haiku",
                            "family": "claude",
                            "backend": "claude",
                            "available": True,
                            "cost_per_1k": 0.0008,  # Real pricing
                            "description": "[TECHNICAL] Claude 3.5 Haiku - Fast & efficient",
                            "capabilities": ["fast", "technical", "code"]
                        }
                    ]
                    
                    self.live_models["claude"] = claude_models
                    print(f"☁️ Claude: {len(claude_models)} models accessible")
                    return True
                else:
                    self.live_models["claude"] = []
                    return False
                    
        except Exception as e:
            print(f"❌ Claude test failed: {e}")
            self.live_models["claude"] = []
            return False
    
    def _test_gemini_live(self) -> bool:
        """Test Gemini API and get available models"""
        api_key = self.api_keys.get("gemini", "")
        if not api_key:
            self.live_models["gemini"] = []
            return False
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            
            data = {'contents': [{'parts': [{'text': 'hi'}]}]}
            
            request = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    gemini_models = [
                        {
                            "id": "gemini-pro",
                            "name": "Gemini Pro", 
                            "family": "gemini",
                            "backend": "gemini",
                            "available": True,
                            "cost_per_1k": 0.0005,
                            "description": "[CREATIVE] Gemini Pro - Google's flagship",
                            "capabilities": ["reasoning", "creative"]
                        }
                    ]
                    
                    self.live_models["gemini"] = gemini_models
                    print(f"🔮 Gemini: {len(gemini_models)} models accessible")
                    return True
                else:
                    self.live_models["gemini"] = []
                    return False
                    
        except Exception as e:
            print(f"❌ Gemini test failed: {e}")
            self.live_models["gemini"] = []
            return False
    
    def _categorize_discovered_models(self):
        """Categorize models based on what was actually discovered"""
        # Reset categories
        for category in self.category_patterns:
            self.category_patterns[category]["models"] = []
        
        # Categorize all discovered models
        for backend_name, models in self.live_models.items():
            for model in models:
                model_name = model["id"].lower()
                model_family = model.get("family", "").lower()
                
                # Determine category based on patterns
                categorized = False
                
                for category, patterns in self.category_patterns.items():
                    keywords = patterns["keywords"]
                    
                    if any(keyword in model_name or keyword in model_family for keyword in keywords):
                        patterns["models"].append(model)
                        categorized = True
                        break
                
                # Default to general if not categorized
                if not categorized:
                    self.category_patterns["general"]["models"].append(model)
    
    def _update_health_status(self):
        """Update connection health based on scan results"""
        self.connection_health = {
            "ollama": {
                "connected": len(self.live_models.get("ollama", [])) > 0,
                "model_count": len(self.live_models.get("ollama", [])),
                "last_scan": datetime.now().isoformat()
            },
            "claude": {
                "connected": len(self.live_models.get("claude", [])) > 0,
                "model_count": len(self.live_models.get("claude", [])),
                "last_scan": datetime.now().isoformat()
            },
            "gemini": {
                "connected": len(self.live_models.get("gemini", [])) > 0,
                "model_count": len(self.live_models.get("gemini", [])),
                "last_scan": datetime.now().isoformat()
            }
        }
    
    def get_available_models(self, category: str = None, backend: str = None) -> List[Dict]:
        """Get models that are available RIGHT NOW"""
        if not self.live_models:
            self.scan_available_models()
        
        available_models = []
        
        if category:
            # Get models from specific category
            if category in self.category_patterns:
                available_models = self.category_patterns[category]["models"].copy()
        else:
            # Get all models
            for backend_models in self.live_models.values():
                available_models.extend(backend_models)
        
        # Filter by backend if specified
        if backend:
            available_models = [m for m in available_models if m.get("backend") == backend]
        
        return available_models
    
    def recommend_model(self, task_type: str, complexity: str = "medium") -> Optional[Dict]:
        """Recommend best available model for task"""
        
        # Map task to category
        task_to_category = {
            "creative": "creative",
            "technical": "technical", 
            "debug": "debug",
            "code": "technical",
            "general": "general"
        }
        
        category = task_to_category.get(task_type, "general")
        available_models = self.get_available_models(category=category)
        
        if not available_models:
            # Fallback to any available
            available_models = self.get_available_models()
        
        if not available_models:
            return None
        
        # Smart selection based on complexity and availability
        if complexity == "simple":
            # Prefer local models for simple tasks
            local_models = [m for m in available_models if m.get("backend") == "ollama"]
            if local_models:
                # Prefer smaller models for simple tasks
                local_models.sort(key=lambda x: x.get("size_gb", 999))
                return local_models[0]
        
        elif complexity == "complex":
            # Prefer cloud models for complex tasks
            cloud_models = [m for m in available_models if m.get("backend") in ["claude", "gemini"]]
            if cloud_models:
                # Prefer Claude for complex tasks
                claude_models = [m for m in cloud_models if m.get("backend") == "claude"]
                if claude_models:
                    return claude_models[0]
                return cloud_models[0]
        
        # Return first available as fallback
        return available_models[0]
    
    def get_model_specs(self, model_id: str) -> Dict:
        """Get real specs for a model"""
        return self.model_specs.get(model_id, {})
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if model is actually available right now"""
        for backend_models in self.live_models.values():
            for model in backend_models:
                if model["id"] == model_id:
                    return model.get("available", False)
        return False
    
    def set_api_key(self, backend: str, api_key: str):
        """Set API key and immediately test it"""
        self.api_keys[backend] = api_key
        
        # Immediately test the key
        if backend == "claude":
            self._test_claude_live()
        elif backend == "gemini":
            self._test_gemini_live()
    
    def force_refresh(self) -> Tuple[bool, str]:
        """Force immediate refresh of all models"""
        return self.scan_available_models(force_refresh=True)
    
    def get_live_status_report(self) -> str:
        """Generate real-time status report"""
        if not self.live_models:
            self.scan_available_models()
        
        report = f"""# Real-Time Model Status
Scanned: {self.last_scan.strftime('%H:%M:%S') if self.last_scan else 'Never'}

## Live Availability:
"""
        
        total_models = 0
        for backend, models in self.live_models.items():
            count = len(models)
            total_models += count
            status = "🟢 ONLINE" if count > 0 else "🔴 OFFLINE"
            report += f"{backend}: {status} ({count} models)\n"
        
        report += f"\nTotal Available: {total_models} models\n"
        
        report += "\n## By Category:\n"
        for category, data in self.category_patterns.items():
            model_count = len(data["models"])
            report += f"{category}: {model_count} models\n"
        
        if self.model_specs:
            report += "\n## Model Specs (Local):\n"
            for model_id, specs in self.model_specs.items():
                size = specs.get("size_gb", 0)
                memory = specs.get("memory_gb", 0)
                family = specs.get("family", "unknown")
                report += f"{model_id}: {family} ({size:.1f}GB disk, {memory:.1f}GB RAM)\n"
        
        return report

