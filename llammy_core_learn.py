# =============================================================================
# LLAMMY MODULE 1: CORE LEARNING ENGINE
# llammy_core_learning.py
# 
# Extracted from Llammy v8.4 Framework - The Foundation Module
# Handles persistent memory, data accumulation, and system status
# =============================================================================

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

def get_addon_directory():
    """Get the addon directory path"""
    return os.path.dirname(os.path.realpath(__file__))

def get_learning_csv_path():
    """Get path to learning data CSV"""
    return os.path.join(get_addon_directory(), "llammy_memory.csv")

def get_metrics_csv_path():
    """Get path to metrics CSV"""
    return os.path.join(get_addon_directory(), "llammy_metrics.csv")

# =============================================================================
# CORE STATUS TRACKING
# =============================================================================

class LlammyStatus:
    """
    Core system status tracking
    Monitors operations, timeouts, and system state
    """
    def __init__(self):
        self.current_operation = "idle"
        self.processing_step = ""
        self.last_update = time.time()
        self.start_time = None
        self.timeout_seconds = 120
        
    def update_operation(self, operation, step=""):
        """Update current operation and step"""
        self.current_operation = operation
        self.processing_step = step
        self.last_update = time.time()
        
        if operation != "idle" and self.start_time is None:
            self.start_time = time.time()
            print(f"🚀 Starting: {operation} - {step}")
        elif operation == "idle":
            self.start_time = None
            print(f"✅ Completed: {operation}")
        
        self.check_timeout()
        
    def check_timeout(self):
        """Check if operation has timed out"""
        if (self.start_time and 
            time.time() - self.start_time > self.timeout_seconds):
            print(f"⏰ TIMEOUT: Auto-forcing idle after {self.timeout_seconds}s")
            self.force_idle()
    
    def force_idle(self):
        """Emergency stop - force system to idle state"""
        self.current_operation = "idle"
        self.processing_step = ""
        self.start_time = None
        self.last_update = time.time()
        print("🛑 FORCED TO IDLE STATE")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status information"""
        return {
            'operation': self.current_operation,
            'step': self.processing_step,
            'last_update': self.last_update,
            'start_time': self.start_time,
            'timeout_seconds': self.timeout_seconds,
            'is_active': self.current_operation != "idle"
        }

# =============================================================================
# METRICS TRACKING WITH DASHBOARD
# =============================================================================

class MetricsTracker:
    """
    Comprehensive metrics tracking for business value and performance analysis
    Tracks success rates, response times, and system health
    """
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.current_stage = "idle"
        
        # Pipeline stages for UI visualization
        self.pipeline_stages = [
            {"name": "Prompt Generation", "status": "pending"},
            {"name": "RAG Context Retrieval", "status": "pending"}, 
            {"name": "Heavy Lifting", "status": "pending"},
            {"name": "Code Generation", "status": "pending"},
            {"name": "Auto-Debug", "status": "pending"}
        ]
        
        # System health monitoring
        self.system_health = {
            "ram_status": "OPTIMAL",
            "temperature": "COOL",
            "pipeline_status": "ACTIVE",
            "api_status": "ONLINE"
        }
        
        self.active_models = {}
        
        # Load historical data on initialization
        self.load_historical_metrics()
    
    def update_metrics(self, success=True, response_time=0.0):
        """Update performance metrics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        if self.total_requests > 1:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
        else:
            self.avg_response_time = response_time
        
        # Save to CSV for accumulation - THIS IS THE GOLD MINE!
        self.save_metrics_to_csv(success, response_time)
    
    def get_success_rate(self):
        """Get current success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def update_stage(self, stage_name, status="active"):
        """Update pipeline stage status"""
        self.current_stage = stage_name
        for stage in self.pipeline_stages:
            if stage["name"] == stage_name:
                stage["status"] = status
            elif stage["status"] == "active" and status == "active":
                stage["status"] = "completed"
    
    def get_ram_usage(self):
        """Get current RAM usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0  # psutil not available
    
    def get_active_model_status(self):
        """Get status of active models"""
        return {
            "total_models": len(self.active_models),
            "ram_usage": self.get_ram_usage(),
            "gpu_available": self.check_gpu_available()
        }
    
    def check_gpu_available(self):
        """Check if GPU is available for diffusion models"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            return False
    
    def save_metrics_to_csv(self, success, response_time):
        """Save metrics to CSV for accumulative data - TRAINING DATASET GOLD!"""
        csv_path = get_metrics_csv_path()
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                fieldnames = [
                    'timestamp', 'success', 'response_time', 'total_requests',
                    'success_rate', 'current_stage', 'ram_usage'
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'success': success,
                    'response_time': response_time,
                    'total_requests': self.total_requests,
                    'success_rate': self.get_success_rate(),
                    'current_stage': self.current_stage,
                    'ram_usage': self.get_ram_usage()
                })
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def load_historical_metrics(self):
        """Load historical metrics from CSV to restore state"""
        csv_path = get_metrics_csv_path()
        if not os.path.exists(csv_path):
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(reader)
                
                if data:
                    # Restore totals from CSV - PERSISTENT MEMORY!
                    last_entry = data[-1]
                    self.total_requests = int(last_entry.get('total_requests', 0))
                    success_rate = float(last_entry.get('success_rate', 0))
                    self.successful_requests = int(self.total_requests * success_rate / 100)
                    self.failed_requests = self.total_requests - self.successful_requests
                    
                    print(f"📊 Restored metrics: {self.total_requests} requests, {success_rate:.1f}% success")
        except Exception as e:
            print(f"⚠️ Error loading historical metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'performance': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.get_success_rate(),
                'avg_response_time': self.avg_response_time
            },
            'pipeline': {
                'current_stage': self.current_stage,
                'stages': self.pipeline_stages.copy()
            },
            'system_health': {
                'ram_usage': self.get_ram_usage(),
                'active_models': len(self.active_models),
                'gpu_available': self.check_gpu_available()
            }
        }
    
    def reset_pipeline_stages(self):
        """Reset all pipeline stages to pending"""
        for stage in self.pipeline_stages:
            stage["status"] = "pending"
        self.current_stage = "idle"

# =============================================================================
# LEARNING SYSTEM - THE GOLD MINE DATA ACCUMULATION
# =============================================================================

class LearningSystem:
    """
    Core learning system that accumulates data for fine-tuning datasets
    This is where the BUSINESS VALUE is generated!
    """
    
    def __init__(self):
        self.learning_csv_path = get_learning_csv_path()
        self.character_knowledge = {}
        self.pattern_library = {}
        self.learning_enabled = True
    
    def save_learning_entry(self, user_input: str, code: str, success: bool, model_info: str = ""):
        """Save learning data to CSV - TRAINING DATASET ACCUMULATION"""
        if not self.learning_enabled:
            return
            
        csv_path = self.learning_csv_path
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'user_input', 'code', 'success', 'model_info']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input[:200],  # Truncate for CSV
                    'code': code[:1000],  # Truncate for CSV
                    'success': success,
                    'model_info': model_info,
                })
                print(f"💎 Learning data saved: {success} | Dataset grows!")
        except Exception as e:
            print(f"❌ Error saving learning: {e}")
    
    def update_character_knowledge(self, character_name: str, interaction_data: Dict[str, Any]):
        """Update character knowledge for persistent learning"""
        if character_name not in self.character_knowledge:
            self.character_knowledge[character_name] = {
                'appearances': 0,
                'successful_interactions': 0,
                'personality_traits': [],
                'typical_actions': [],
                'emotional_states': [],
                'last_updated': None
            }
        
        char_data = self.character_knowledge[character_name]
        char_data['appearances'] += 1
        char_data['last_updated'] = datetime.now().isoformat()
        
        # Update based on interaction success
        if interaction_data.get('success', False):
            char_data['successful_interactions'] += 1
        
        # Update traits if provided
        if 'traits' in interaction_data:
            for trait in interaction_data['traits']:
                if trait not in char_data['personality_traits']:
                    char_data['personality_traits'].append(trait)
        
        # Update actions if provided
        if 'actions' in interaction_data:
            for action in interaction_data['actions']:
                if action not in char_data['typical_actions']:
                    char_data['typical_actions'].append(action)
        
        # Update emotional states if provided
        if 'emotional_state' in interaction_data:
            emotion = interaction_data['emotional_state']
            if emotion not in char_data['emotional_states']:
                char_data['emotional_states'].append(emotion)
        
        print(f"🧠 Updated knowledge for {character_name}: {char_data['appearances']} appearances")
    
    def get_character_insights(self, character_name: str) -> Dict[str, Any]:
        """Get learned insights about a character"""
        if character_name not in self.character_knowledge:
            return {'status': 'unknown_character'}
        
        char_data = self.character_knowledge[character_name]
        success_rate = 0.0
        if char_data['appearances'] > 0:
            success_rate = (char_data['successful_interactions'] / char_data['appearances']) * 100
        
        return {
            'character_name': character_name,
            'appearances': char_data['appearances'],
            'success_rate': success_rate,
            'personality_traits': char_data['personality_traits'],
            'typical_actions': char_data['typical_actions'][-5:],  # Last 5 actions
            'emotional_range': char_data['emotional_states'],
            'last_updated': char_data['last_updated']
        }
    
    def get_dataset_value_metrics(self) -> Dict[str, Any]:
        """Get metrics showing the business value of accumulated dataset"""
        csv_path = self.learning_csv_path
        
        if not os.path.exists(csv_path):
            return {'status': 'no_data', 'value': 0}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(reader)
            
            total_entries = len(data)
            successful_entries = len([d for d in data if d.get('success') == 'True'])
            
            # Calculate business value metrics
            # Quality training data is worth $$ for fine-tuning
            estimated_value_per_entry = 0.50  # Conservative estimate
            total_estimated_value = total_entries * estimated_value_per_entry
            
            return {
                'total_entries': total_entries,
                'successful_entries': successful_entries,
                'success_rate': (successful_entries / total_entries * 100) if total_entries > 0 else 0,
                'estimated_dataset_value_usd': total_estimated_value,
                'characters_learned': len(self.character_knowledge),
                'data_quality': 'high' if successful_entries / total_entries > 0.8 else 'medium' if successful_entries / total_entries > 0.6 else 'low'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing dataset value: {e}")
            return {'status': 'error', 'value': 0}
    
    def export_training_dataset(self, format_type: str = 'json') -> str:
        """Export accumulated data as training dataset"""
        csv_path = self.learning_csv_path
        
        if not os.path.exists(csv_path):
            return "No training data available"
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(reader)
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_entries': len(data),
                    'characters_learned': len(self.character_knowledge),
                    'export_format': format_type
                },
                'character_knowledge': self.character_knowledge,
                'training_entries': data,
                'pattern_library': self.pattern_library
            }
            
            if format_type == 'json':
                import json
                export_path = os.path.join(get_addon_directory(), f"llammy_training_export_{int(time.time())}.json")
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
                return export_path
            
        except Exception as e:
            print(f"❌ Error exporting training dataset: {e}")
            return f"Export failed: {e}"

# =============================================================================
# CORE LEARNING ENGINE - MAIN COORDINATOR
# =============================================================================

class CoreLearningEngine:
    """
    Main coordinator for the core learning system
    Manages status, metrics, and learning data accumulation
    """
    
    def __init__(self):
        self.status = LlammyStatus()
        self.metrics = MetricsTracker()
        self.learning = LearningSystem()
        self.module_name = "core_learning"
        self.version = "1.0"
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the core learning engine"""
        try:
            # Test CSV access
            self.learning.save_learning_entry("init_test", "# test", True, "initialization")
            
            self.initialized = True
            print("✅ Core Learning Engine initialized successfully!")
            print(f"📊 Loaded metrics: {self.metrics.total_requests} total requests")
            print(f"🧠 Character knowledge: {len(self.learning.character_knowledge)} characters")
            
            # Show dataset value
            value_metrics = self.learning.get_dataset_value_metrics()
            if value_metrics.get('total_entries', 0) > 0:
                print(f"💎 Dataset value: {value_metrics['total_entries']} entries worth ~${value_metrics['estimated_dataset_value_usd']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Core Learning Engine initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the core learning engine"""
        self.status.force_idle()
        self.metrics.reset_pipeline_stages()
        print("🛑 Core Learning Engine shutdown")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all core systems"""
        return {
            'module_info': {
                'name': self.module_name,
                'version': self.version,
                'initialized': self.initialized
            },
            'status': self.status.get_status(),
            'metrics': self.metrics.get_metrics_summary(),
            'learning': {
                'characters_known': len(self.learning.character_knowledge),
                'learning_enabled': self.learning.learning_enabled,
                'dataset_value': self.learning.get_dataset_value_metrics()
            },
            'file_paths': {
                'learning_csv': self.learning.learning_csv_path,
                'metrics_csv': get_metrics_csv_path(),
                'addon_directory': get_addon_directory()
            }
        }
    
    def force_reset(self):
        """Emergency reset of all systems"""
        self.status.force_idle()
        self.metrics.reset_pipeline_stages()
        print("🚨 CORE LEARNING ENGINE: EMERGENCY RESET COMPLETE")

# =============================================================================
# MODULE INTERFACE FOR OTHER SYSTEMS
# =============================================================================

# Global instance - other modules will import this
core_learning_engine = CoreLearningEngine()

# Convenience functions for other modules
def get_core_status():
    """Get core system status"""
    return core_learning_engine.status

def get_core_metrics():
    """Get core metrics"""
    return core_learning_engine.metrics

def get_core_learning():
    """Get core learning system"""
    return core_learning_engine.learning

def update_status(operation: str, step: str = ""):
    """Update system status"""
    core_learning_engine.status.update_operation(operation, step)

def update_metrics(success: bool = True, response_time: float = 0.0):
    """Update performance metrics"""
    core_learning_engine.metrics.update_metrics(success, response_time)

def save_learning_data(user_input: str, code: str, success: bool, model_info: str = ""):
    """Save learning data"""
    core_learning_engine.learning.save_learning_entry(user_input, code, success, model_info)

def update_character_knowledge(character_name: str, interaction_data: Dict[str, Any]):
    """Update character knowledge"""
    core_learning_engine.learning.update_character_knowledge(character_name, interaction_data)

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Test the module
    print("🚀 Testing Core Learning Engine Module...")
    
    if core_learning_engine.initialize():
        print("✅ Module test successful!")
        
        # Test basic functionality
        update_status("testing", "core functionality")
        update_metrics(True, 1.5)
        save_learning_data("test input", "test code", True, "test model")
        update_character_knowledge("Tien", {
            'success': True,
            'traits': ['enthusiastic', 'musical'],
            'actions': ['plays_harmonica'],
            'emotional_state': 'joyful'
        })
        
        status = core_learning_engine.get_comprehensive_status()
        print(f"📊 Final status: {status['metrics']['performance']['total_requests']} requests")
        print(f"🧠 Characters known: {status['learning']['characters_known']}")
        
        update_status("idle")
        core_learning_engine.shutdown()
    else:
        print("❌ Module test failed!")

print("💎 MODULE 1: CORE LEARNING ENGINE - READY FOR INTEGRATION!")
print("This module provides:")
print("✅ Persistent memory and data accumulation")
print("✅ Performance metrics tracking") 
print("✅ System status monitoring")
print("✅ Character knowledge building")
print("✅ Training dataset generation (BUSINESS VALUE!)")
print("✅ CSV-based persistent storage")
print("✅ Module interface for other systems")