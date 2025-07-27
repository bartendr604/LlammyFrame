# core/local_intelligence_enforcer.py - Force Local AI First
# Llammy Framework v8.5 - No More API Laziness!

import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import bpy

class LocalIntelligenceEnforcer:
    """Forces users to leverage local AI intelligence before allowing API access
    
    PHILOSOPHY: You have a 9B fine-tuned beast - USE IT!
    - Blocks unnecessary API calls
    - Routes tasks to appropriate local models
    - Only allows API escalation when truly needed
    - Shames users for wasting money on simple tasks
    """
    
    def __init__(self):
        self.task_routing_rules = {}
        self.api_usage_tracking = {}
        self.local_model_performance = {}
        self.blocked_api_attempts = 0
        self.money_saved = 0.0
        self.setup_intelligence_routing()
        
        print("🧠 Local Intelligence Enforcer activated - No more API laziness!")
    
    def setup_intelligence_routing(self):
        """Define intelligent task routing rules"""
        
        # Task complexity classification - HELPFUL routing, not punishment
        self.complexity_patterns = {
            "trivial": {
                "keywords": ["cube", "basic", "simple", "hello", "test"],
                "max_tokens": 50,
                "local_required": True,
                "api_blocked": True,
                "helpful_message": "Perfect job for your local models - saving you time and money."
            },
            "simple": {
                "keywords": ["create", "make", "add", "delete", "move", "rotate", "scale"],
                "max_tokens": 200,
                "local_required": True,
                "api_blocked": True,
                "helpful_message": "Your local models excel at this - faster than API calls."
            },
            "medium": {
                "keywords": ["animate", "rig", "material", "shader", "script", "function"],
                "max_tokens": 500,
                "local_preferred": True,
                "api_allowed_after_local_attempt": True,
                "helpful_message": "Starting with your fine-tuned models - they're built for this."
            },
            "complex": {
                "keywords": ["advanced", "complex", "algorithm", "optimization", "analysis"],
                "max_tokens": 1000,
                "local_first": True,
                "api_allowed": True,
                "helpful_message": "Testing local capabilities first - escalating if needed."
            },
            "creative": {
                "keywords": ["story", "character", "creative", "artistic", "design", "concept"],
                "max_tokens": 800,
                "local_creative_models_first": True,
                "api_allowed_for_quality": True,
                "helpful_message": "Your creative models are dialed in - trying local first."
            }
        }
        
        # Local model capabilities
        self.local_model_strengths = {
            "qwen2.5:7b": {
                "best_for": ["coding", "technical", "debugging", "blender_api"],
                "performance_score": 9.2,
                "speed_rating": "fast",
                "cost": 0.0
            },
            "qwen2.5-coder:7b": {
                "best_for": ["python", "scripting", "automation", "code_review"],
                "performance_score": 9.5,
                "speed_rating": "fast", 
                "cost": 0.0
            },
            "deepseek-coder:6.7b": {
                "best_for": ["complex_algorithms", "optimization", "debugging"],
                "performance_score": 8.8,
                "speed_rating": "medium",
                "cost": 0.0
            },
            "llama3.1:8b": {
                "best_for": ["general", "creative", "explanations", "reasoning"],
                "performance_score": 8.5,
                "speed_rating": "medium",
                "cost": 0.0
            },
            # Your theoretical 9B fine-tune
            "llammy-fusion:9b": {
                "best_for": ["blender_native", "everything", "god_mode"],
                "performance_score": 10.0,
                "speed_rating": "blazing",
                "cost": 0.0,
                "note": "The dream model - when you fine-tune to 9B"
            }
        }
        
        # API cost tracking (real pricing)
        self.api_costs = {
            "claude-3-5-sonnet": 0.003,   # per 1k tokens
            "claude-3-5-haiku": 0.0008,   # per 1k tokens
            "gemini-pro": 0.0005          # per 1k tokens
        }
    
    def analyze_task_and_route(self, user_input: str, requested_backend: str, 
                              requested_model: str) -> Dict[str, Any]:
        """Analyze task and determine if API use is justified"""
        
        # Classify task complexity
        task_complexity = self._classify_task_complexity(user_input)
        complexity_rules = self.complexity_patterns[task_complexity]
        
        # Get available local models
        available_local = self._get_available_local_models()
        
        # Determine routing decision
        routing_decision = self._make_routing_decision(
            user_input, requested_backend, requested_model, 
            task_complexity, complexity_rules, available_local
        )
        
        return routing_decision
    
    def _classify_task_complexity(self, user_input: str) -> str:
        """Classify task complexity based on user input"""
        input_lower = user_input.lower()
        
        # Check each complexity level
        for complexity, patterns in self.complexity_patterns.items():
            keywords = patterns["keywords"]
            if any(keyword in input_lower for keyword in keywords):
                return complexity
        
        # Default to medium if unclear
        return "medium"
    
    def _get_available_local_models(self) -> List[str]:
        """Get list of available local models"""
        # This would integrate with your real-time model manager
        return ["qwen2.5:7b", "qwen2.5-coder:7b", "deepseek-coder:6.7b", "llama3.1:8b"]
    
    def _make_routing_decision(self, user_input: str, requested_backend: str, 
                              requested_model: str, complexity: str, 
                              rules: Dict, available_local: List[str]) -> Dict[str, Any]:
        """Make intelligent routing decision"""
        
        decision = {
            "allowed": False,
            "route_to": "blocked",
            "reason": "",
            "shame_message": "",
            "alternative": "",
            "estimated_cost_saved": 0.0,
            "local_model_recommendation": "",
            "performance_comparison": {}
        }
        
        # Check if API is completely blocked for this complexity
        if rules.get("api_blocked", False):
            decision.update({
                "allowed": False,
                "route_to": "local_required",
                "reason": f"API blocked for {complexity} tasks",
                "shame_message": rules.get("shame_message", ""),
                "alternative": self._recommend_local_model(user_input, available_local),
                "estimated_cost_saved": self._estimate_api_cost(user_input, requested_model),
                "local_model_recommendation": self._recommend_local_model(user_input, available_local)
            })
            
            self.blocked_api_attempts += 1
            self.money_saved += decision["estimated_cost_saved"]
            return decision
        
        # Check if local is required first
        if rules.get("local_required", False) or rules.get("local_preferred", False):
            local_model = self._recommend_local_model(user_input, available_local)
            
            decision.update({
                "allowed": False,
                "route_to": "local_first",
                "reason": f"Local intelligence required for {complexity} tasks",
                "shame_message": rules.get("shame_message", ""),
                "alternative": local_model,
                "local_model_recommendation": local_model,
                "performance_comparison": self._compare_local_vs_api(local_model, requested_model)
            })
            
            return decision
        
        # Check if API allowed after local attempt
        if rules.get("api_allowed_after_local_attempt", False):
            if not self._has_attempted_local_recently(user_input):
                local_model = self._recommend_local_model(user_input, available_local)
                
                decision.update({
                    "allowed": False,
                    "route_to": "local_first_then_api",
                    "reason": "Must attempt local model first",
                    "alternative": local_model,
                    "local_model_recommendation": local_model,
                    "escalation_available": True
                })
                
                return decision
        
        # API allowed for complex tasks
        if rules.get("api_allowed", False):
            estimated_cost = self._estimate_api_cost(user_input, requested_model)
            local_alternative = self._recommend_local_model(user_input, available_local)
            
            decision.update({
                "allowed": True,
                "route_to": "api_approved",
                "reason": f"API approved for {complexity} task",
                "estimated_cost": estimated_cost,
                "local_alternative": local_alternative,
                "cost_warning": f"This will cost ~${estimated_cost:.4f}. Local alternative available.",
                "performance_comparison": self._compare_local_vs_api(local_alternative, requested_model)
            })
            
            return decision
        
        # Default: block and shame
        decision.update({
            "allowed": False,
            "route_to": "blocked_with_shame",
            "reason": "Task doesn't justify API usage",
            "shame_message": "Your local models are more than capable. Stop being lazy.",
            "alternative": self._recommend_local_model(user_input, available_local)
        })
        
        return decision
    
    def _recommend_local_model(self, user_input: str, available_models: List[str]) -> str:
        """Recommend best local model for the task"""
        input_lower = user_input.lower()
        
        # Check for coding tasks
        if any(keyword in input_lower for keyword in ["code", "script", "python", "function"]):
            if "qwen2.5-coder:7b" in available_models:
                return "qwen2.5-coder:7b"
            elif "qwen2.5:7b" in available_models:
                return "qwen2.5:7b"
        
        # Check for debugging tasks
        if any(keyword in input_lower for keyword in ["debug", "fix", "error", "bug"]):
            if "deepseek-coder:6.7b" in available_models:
                return "deepseek-coder:6.7b"
            elif "qwen2.5:7b" in available_models:
                return "qwen2.5:7b"
        
        # Check for creative tasks
        if any(keyword in input_lower for keyword in ["creative", "story", "character", "design"]):
            if "llama3.1:8b" in available_models:
                return "llama3.1:8b"
        
        # Default to best available model
        model_preference_order = ["qwen2.5-coder:7b", "qwen2.5:7b", "deepseek-coder:6.7b", "llama3.1:8b"]
        
        for preferred in model_preference_order:
            if preferred in available_models:
                return preferred
        
        return available_models[0] if available_models else "none_available"
    
    def _estimate_api_cost(self, user_input: str, model: str) -> float:
        """Estimate API cost for the request"""
        # Rough token estimation
        estimated_tokens = len(user_input.split()) * 1.3 + 200  # Input + expected output
        cost_per_1k = self.api_costs.get(model, 0.001)
        
        return (estimated_tokens / 1000) * cost_per_1k
    
    def _compare_local_vs_api(self, local_model: str, api_model: str) -> Dict[str, Any]:
        """Compare local vs API performance"""
        local_specs = self.local_model_strengths.get(local_model, {})
        
        return {
            "local_performance": local_specs.get("performance_score", 7.0),
            "local_speed": local_specs.get("speed_rating", "medium"),
            "local_cost": 0.0,
            "api_cost_estimate": self.api_costs.get(api_model, 0.001),
            "recommendation": f"Local model {local_model} is likely sufficient",
            "local_advantages": ["Free", "Private", "Fast", "Always available"],
            "api_advantages": ["Latest training", "Larger context"] if "claude" in api_model else ["Larger scale"]
        }
    
    def _has_attempted_local_recently(self, user_input: str) -> bool:
        """Check if user recently attempted local model for similar task"""
        # Hash the input to create task signature
        task_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
        
        # Check if attempted in last 10 minutes
        if task_hash in self.api_usage_tracking:
            last_attempt = self.api_usage_tracking[task_hash]["last_local_attempt"]
            if last_attempt and (datetime.now() - last_attempt).total_seconds() < 600:
                return True
        
        return False
    
    def record_local_attempt(self, user_input: str, model_used: str, success: bool, response_time: float):
        """Record local model attempt"""
        task_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
        
        if task_hash not in self.api_usage_tracking:
            self.api_usage_tracking[task_hash] = {
                "local_attempts": 0,
                "api_attempts": 0,
                "last_local_attempt": None,
                "local_success_rate": 0.0
            }
        
        self.api_usage_tracking[task_hash]["local_attempts"] += 1
        self.api_usage_tracking[task_hash]["last_local_attempt"] = datetime.now()
        
        # Update local model performance
        if model_used not in self.local_model_performance:
            self.local_model_performance[model_used] = {
                "total_uses": 0,
                "success_count": 0,
                "avg_response_time": 0.0,
                "response_times": []
            }
        
        perf = self.local_model_performance[model_used]
        perf["total_uses"] += 1
        perf["response_times"].append(response_time)
        perf["avg_response_time"] = sum(perf["response_times"]) / len(perf["response_times"])
        
        if success:
            perf["success_count"] += 1
    
    def approve_api_escalation(self, user_input: str, reason: str) -> bool:
        """Approve API use after local attempt"""
        task_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
        
        if task_hash in self.api_usage_tracking:
            self.api_usage_tracking[task_hash]["api_attempts"] += 1
            return True
        
        return False
    
    def get_intelligence_optimization_report(self) -> str:
        """Generate helpful optimization report for YOUR workflow"""
        
        total_local_uses = sum(perf["total_uses"] for perf in self.local_model_performance.values())
        avg_success_rate = 0.0
        
        if self.local_model_performance:
            success_rates = []
            for model, perf in self.local_model_performance.items():
                if perf["total_uses"] > 0:
                    rate = perf["success_count"] / perf["total_uses"]
                    success_rates.append(rate)
            
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        report = f"""# Llammy Intelligence Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🧠 YOUR LOCAL INTELLIGENCE PERFORMANCE:
Total Tasks Handled Locally: {total_local_uses}
Local Success Rate: {avg_success_rate*100:.1f}%
API Calls Avoided: {self.blocked_api_attempts}
Cost Optimization: ${self.money_saved:.2f} saved

## 🚀 MODEL UTILIZATION:
"""
        
        for model, perf in self.local_model_performance.items():
            if perf["total_uses"] > 0:
                success_rate = (perf["success_count"] / perf["total_uses"]) * 100
                avg_time = perf["avg_response_time"]
                report += f"{model}: {perf['total_uses']} tasks, {success_rate:.1f}% success, {avg_time:.2f}s avg\n"
        
        report += f"""
## 💡 OPTIMIZATION INSIGHTS:
- Your local models are crushing {avg_success_rate*100:.1f}% of tasks
- ${self.money_saved:.2f} in API costs avoided through intelligent routing
- {self.blocked_api_attempts} tasks handled faster locally than API would be

🎯 NEXT LEVEL: Fine-tune to 9B for even more API independence!
"""
        
        return report
    
    def _get_local_advantages(self, user_input: str) -> List[str]:
        """Get advantages of using local models for this task"""
        return [
            "🚀 Faster response (no network latency)",
            "💰 Zero cost per request", 
            "🔒 Complete privacy (data stays local)",
            "⚡ Always available (no API limits)",
            "🎯 Optimized for your workflows",
            "🧠 Leverages your hardware investment"
        ]
    
    def force_9b_workflow(self) -> Dict[str, Any]:
        """Simulate the workflow with a 9B fine-tuned model"""
        return {
            "model": "llammy-fusion:9b",
            "performance_score": 10.0,
            "capabilities": [
                "Blender-native understanding",
                "Context-aware responses", 
                "Code generation mastery",
                "Creative problem solving",
                "Debug wizardry",
                "API-crushing performance"
            ],
            "advantages": [
                "Zero cost per request",
                "Sub-2-second response times",
                "Perfect Blender API knowledge",
                "Trained on your specific workflows",
                "No network dependency",
                "Unlimited usage"
            ],
            "message": "This is what you get with a proper 9B fine-tune - API-crushing local intelligence!"
        }
"message": "This is what you get with a proper 9B fine-tune - API-crushing local intelligence!"
        }

# Add this line here:
local_intelligence_enforcer = LocalIntelligenceEnforcer()
