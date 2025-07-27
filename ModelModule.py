# core/local_intelligence_advisor.py - Smart Model Selection & Cost Analysis
# Llammy Framework v8.5 - Show Don't Force!

import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import bpy
import subprocess
import json

class LocalIntelligenceAdvisor:
    """Intelligent model selection advisor with real-time Ollama integration
    
    PHILOSOPHY: Show them the power of their local setup with data
    - Live Ollama model discovery
    - Real cost comparisons with profit analysis
    - High-quality data value estimation
    - Performance benchmarking
    - Let THEM choose with full transparency
    """
    
    def __init__(self):
        self.task_routing_suggestions = {}
        self.api_usage_tracking = {}
        self.local_model_performance = {}
        self.data_quality_metrics = {}
        self.profit_calculations = {}
        self.ollama_models = []
        self.refresh_ollama_models()
        self.setup_intelligence_analysis()
        
        print("🧠 Local Intelligence Advisor initialized - Your models, your choice!")
    
    def refresh_ollama_models(self):
        """Get real-time list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                self.ollama_models = []
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        model_name = parts[0]
                        model_size = parts[1] if len(parts) > 1 else "unknown"
                        
                        # Get model info
                        model_info = self._get_model_specs(model_name, model_size)
                        self.ollama_models.append(model_info)
                        
                print(f"📋 Found {len(self.ollama_models)} Ollama models ready for action")
            else:
                print("⚠️ Couldn't connect to Ollama - check if it's running")
                self.ollama_models = self._get_fallback_models()
                
        except Exception as e:
            print(f"⚠️ Ollama discovery failed: {e}")
            self.ollama_models = self._get_fallback_models()
    
    def _get_model_specs(self, model_name: str, model_size: str) -> Dict[str, Any]:
        """Get detailed specs for a model"""
        
        # Model capability analysis based on name patterns
        capabilities = []
        strengths = []
        
        name_lower = model_name.lower()
        
        if 'coder' in name_lower or 'code' in name_lower:
            capabilities.extend(['coding', 'debugging', 'scripting'])
            strengths.append('Code Generation')
            
        if 'qwen' in name_lower:
            capabilities.extend(['technical', 'analysis', 'reasoning'])
            strengths.append('Technical Analysis')
            
        if 'llama' in name_lower:
            capabilities.extend(['general', 'creative', 'conversation'])
            strengths.append('General Purpose')
            
        if 'deepseek' in name_lower:
            capabilities.extend(['algorithms', 'optimization', 'complex_logic'])
            strengths.append('Algorithm Design')
            
        if 'mistral' in name_lower:
            capabilities.extend(['reasoning', 'analysis', 'multilingual'])
            strengths.append('Analytical Reasoning')
        
        # Size-based performance estimation
        size_gb = self._extract_size_gb(model_size)
        performance_score = min(10.0, 6.0 + (size_gb / 2.0))  # Rough heuristic
        
        return {
            'name': model_name,
            'size': model_size,
            'size_gb': size_gb,
            'capabilities': capabilities,
            'strengths': strengths,
            'performance_score': performance_score,
            'speed_rating': self._estimate_speed_rating(size_gb),
            'cost_per_use': 0.0,
            'data_quality_potential': self._estimate_data_quality_potential(model_name, size_gb)
        }
    
    def _extract_size_gb(self, size_str: str) -> float:
        """Extract size in GB from size string"""
        if 'GB' in size_str.upper():
            return float(size_str.upper().replace('GB', '').strip())
        elif 'B' in size_str.upper():
            # Convert from raw bytes
            size_bytes = float(size_str.upper().replace('B', '').strip())
            return size_bytes / (1024**3)
        else:
            # Try to extract number and assume GB
            import re
            numbers = re.findall(r'[\d.]+', size_str)
            return float(numbers[0]) if numbers else 7.0
    
    def _estimate_speed_rating(self, size_gb: float) -> str:
        """Estimate speed based on model size"""
        if size_gb < 4:
            return "blazing"
        elif size_gb < 8:
            return "fast"
        elif size_gb < 15:
            return "medium"
        else:
            return "deliberate"
    
    def _estimate_data_quality_potential(self, model_name: str, size_gb: float) -> float:
        """Estimate potential for generating high-quality training data"""
        base_score = min(9.0, 5.0 + (size_gb / 3.0))
        
        # Boost for specialized models
        name_lower = model_name.lower()
        if 'coder' in name_lower:
            base_score += 1.5  # Code data is highly valuable
        if 'qwen' in name_lower and size_gb > 6:
            base_score += 1.0  # Qwen models are solid data generators
        if size_gb > 10:
            base_score += 0.5  # Larger models generally produce better data
            
        return min(10.0, base_score)
    
    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Fallback model list if Ollama discovery fails"""
        return [
            {
                'name': 'qwen2.5-coder:7b',
                'size': '4.2GB',
                'size_gb': 4.2,
                'capabilities': ['coding', 'debugging', 'scripting'],
                'strengths': ['Code Generation', 'Technical Analysis'],
                'performance_score': 9.5,
                'speed_rating': 'fast',
                'cost_per_use': 0.0,
                'data_quality_potential': 9.2
            },
            {
                'name': 'qwen2.5:7b',
                'size': '4.4GB', 
                'size_gb': 4.4,
                'capabilities': ['technical', 'analysis', 'reasoning'],
                'strengths': ['Technical Analysis', 'General Purpose'],
                'performance_score': 9.2,
                'speed_rating': 'fast',
                'cost_per_use': 0.0,
                'data_quality_potential': 8.8
            }
        ]
    
    def setup_intelligence_analysis(self):
        """Setup analysis patterns and cost structures"""
        
        # API cost tracking (current real pricing)
        self.api_costs = {
            "claude-3-5-sonnet": {
                "input_cost": 0.003,    # per 1k tokens
                "output_cost": 0.015,   # per 1k tokens
                "avg_quality_score": 9.5
            },
            "claude-3-5-haiku": {
                "input_cost": 0.0008,
                "output_cost": 0.004,
                "avg_quality_score": 8.5
            },
            "gpt-4": {
                "input_cost": 0.03,
                "output_cost": 0.06,
                "avg_quality_score": 9.2
            },
            "gemini-pro": {
                "input_cost": 0.0005,
                "output_cost": 0.0015,
                "avg_quality_score": 8.0
            }
        }
        
        # High-quality data value estimates (per 1k tokens of training data)
        self.data_value_estimates = {
            "code_snippets": {
                "market_value": 0.05,      # Code training data is valuable
                "quality_multiplier": 2.0,  # High-quality code is 2x more valuable
                "description": "Clean, commented code with examples"
            },
            "technical_explanations": {
                "market_value": 0.03,
                "quality_multiplier": 1.8,
                "description": "Detailed technical documentation and explanations"
            },
            "creative_content": {
                "market_value": 0.02,
                "quality_multiplier": 1.5,
                "description": "Original creative writing and concepts"
            },
            "problem_solving": {
                "market_value": 0.04,
                "quality_multiplier": 2.2,
                "description": "Step-by-step problem-solving examples"
            },
            "blender_specific": {
                "market_value": 0.08,      # Specialized domain data is premium
                "quality_multiplier": 3.0,
                "description": "Blender-specific workflows and scripts"
            }
        }
    
    def analyze_task_and_suggest(self, user_input: str, task_context: str = "") -> Dict[str, Any]:
        """Analyze task and provide intelligent model suggestions with full transparency"""
        
        # Classify the task
        task_type = self._classify_task_type(user_input)
        complexity = self._estimate_complexity(user_input)
        
        # Get suggestions for local models
        local_suggestions = self._rank_local_models_for_task(user_input, task_type)
        
        # Calculate API costs and comparisons
        api_comparisons = self._calculate_api_costs(user_input)
        
        # Estimate data generation value
        data_value_analysis = self._analyze_data_generation_potential(user_input, task_type)
        
        # Generate profit analysis
        profit_analysis = self._calculate_profit_potential(user_input, task_type, local_suggestions)
        
        return {
            "task_analysis": {
                "type": task_type,
                "complexity": complexity,
                "estimated_tokens": self._estimate_token_count(user_input)
            },
            "local_model_options": local_suggestions,
            "api_cost_comparison": api_comparisons,
            "data_value_analysis": data_value_analysis,
            "profit_analysis": profit_analysis,
            "recommendations": self._generate_recommendations(local_suggestions, api_comparisons, profit_analysis),
            "local_advantages": self._get_local_advantages_for_task(task_type),
            "quality_expectations": self._estimate_quality_expectations(task_type, local_suggestions)
        }
    
    def _classify_task_type(self, user_input: str) -> str:
        """Classify what type of task this is"""
        input_lower = user_input.lower()
        
        # Code-related tasks
        if any(keyword in input_lower for keyword in ['code', 'script', 'function', 'class', 'debug', 'fix']):
            return "coding"
        
        # Blender-specific tasks
        if any(keyword in input_lower for keyword in ['blender', 'mesh', 'material', 'shader', 'animation', 'render']):
            return "blender_specific"
        
        # Creative tasks
        if any(keyword in input_lower for keyword in ['create', 'design', 'story', 'character', 'concept']):
            return "creative"
        
        # Analysis tasks
        if any(keyword in input_lower for keyword in ['analyze', 'explain', 'compare', 'evaluate', 'assess']):
            return "analysis"
        
        # Problem-solving
        if any(keyword in input_lower for keyword in ['how to', 'solve', 'help', 'fix', 'optimize']):
            return "problem_solving"
        
        return "general"
    
    def _estimate_complexity(self, user_input: str) -> str:
        """Estimate task complexity"""
        word_count = len(user_input.split())
        input_lower = user_input.lower()
        
        # Simple heuristics
        if word_count < 10 and any(word in input_lower for word in ['simple', 'basic', 'quick']):
            return "simple"
        elif any(word in input_lower for word in ['complex', 'advanced', 'detailed', 'comprehensive']):
            return "complex"
        elif word_count > 50:
            return "complex"
        elif word_count > 20:
            return "medium"
        else:
            return "simple"
    
    def _rank_local_models_for_task(self, user_input: str, task_type: str) -> List[Dict[str, Any]]:
        """Rank available local models for this specific task"""
        ranked_models = []
        
        for model in self.ollama_models:
            score = self._calculate_model_task_fit(model, task_type, user_input)
            
            model_suggestion = {
                **model,
                "task_fit_score": score,
                "estimated_response_time": self._estimate_response_time(model, user_input),
                "expected_quality": self._estimate_output_quality(model, task_type),
                "data_generation_value": self._calculate_data_value_for_model(model, task_type)
            }
            ranked_models.append(model_suggestion)
        
        # Sort by task fit score
        return sorted(ranked_models, key=lambda x: x["task_fit_score"], reverse=True)
    
    def _calculate_model_task_fit(self, model: Dict, task_type: str, user_input: str) -> float:
        """Calculate how well a model fits this task"""
        base_score = model["performance_score"]
        
        # Boost for relevant capabilities
        if task_type in model["capabilities"]:
            base_score += 2.0
        
        # Specific task type bonuses
        name_lower = model["name"].lower()
        
        if task_type == "coding" and "coder" in name_lower:
            base_score += 3.0
        elif task_type == "blender_specific" and model["size_gb"] > 6:
            base_score += 2.0  # Larger models better for specialized tasks
        elif task_type == "analysis" and "qwen" in name_lower:
            base_score += 1.5
        
        return min(10.0, base_score)
    
    def _estimate_response_time(self, model: Dict, user_input: str) -> float:
        """Estimate response time for this model/task combo"""
        base_time = {
            "blazing": 1.5,
            "fast": 3.0,
            "medium": 6.0,
            "deliberate": 12.0
        }.get(model["speed_rating"], 5.0)
        
        # Adjust for input complexity
        token_count = self._estimate_token_count(user_input)
        complexity_multiplier = 1.0 + (token_count / 1000) * 0.5
        
        return base_time * complexity_multiplier
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation"""
        return int(len(text.split()) * 1.3 + 200)  # Input + expected output
    
    def _calculate_api_costs(self, user_input: str) -> Dict[str, Any]:
        """Calculate what this would cost via various APIs"""
        token_count = self._estimate_token_count(user_input)
        input_tokens = int(len(user_input.split()) * 1.3)
        output_tokens = token_count - input_tokens
        
        api_costs = {}
        
        for api_name, pricing in self.api_costs.items():
            input_cost = (input_tokens / 1000) * pricing["input_cost"]
            output_cost = (output_tokens / 1000) * pricing["output_cost"]
            total_cost = input_cost + output_cost
            
            api_costs[api_name] = {
                "total_cost": total_cost,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "quality_score": pricing["avg_quality_score"],
                "cost_per_quality_point": total_cost / pricing["avg_quality_score"]
            }
        
        return api_costs
    
    def _analyze_data_generation_potential(self, user_input: str, task_type: str) -> Dict[str, Any]:
        """Analyze the value of the data that would be generated"""
        
        # Determine data type
        data_type = self._map_task_to_data_type(task_type)
        data_value_info = self.data_value_estimates.get(data_type, self.data_value_estimates["creative_content"])
        
        # Estimate output size
        estimated_output_tokens = self._estimate_token_count(user_input) * 0.7  # Rough output estimation
        
        # Calculate raw value
        raw_value = (estimated_output_tokens / 1000) * data_value_info["market_value"]
        
        # Apply quality multiplier for high-quality generation
        high_quality_value = raw_value * data_value_info["quality_multiplier"]
        
        return {
            "data_type": data_type,
            "description": data_value_info["description"],
            "estimated_output_tokens": estimated_output_tokens,
            "raw_data_value": raw_value,
            "high_quality_value": high_quality_value,
            "value_per_1k_tokens": data_value_info["market_value"],
            "quality_multiplier": data_value_info["quality_multiplier"],
            "potential_profit_margin": high_quality_value * 0.8  # 80% profit margin estimate
        }
    
    def _map_task_to_data_type(self, task_type: str) -> str:
        """Map task type to data value category"""
        mapping = {
            "coding": "code_snippets",
            "blender_specific": "blender_specific", 
            "analysis": "technical_explanations",
            "creative": "creative_content",
            "problem_solving": "problem_solving"
        }
        return mapping.get(task_type, "creative_content")
    
    def _calculate_profit_potential(self, user_input: str, task_type: str, local_suggestions: List[Dict]) -> Dict[str, Any]:
        """Calculate profit potential of using local vs API"""
        
        data_analysis = self._analyze_data_generation_potential(user_input, task_type)
        api_costs = self._calculate_api_costs(user_input)
        
        # Get cheapest API option for comparison
        cheapest_api = min(api_costs.values(), key=lambda x: x["total_cost"])
        cheapest_api_name = [name for name, cost in api_costs.items() if cost["total_cost"] == cheapest_api["total_cost"]][0]
        
        # Best local model
        best_local = local_suggestions[0] if local_suggestions else None
        
        if not best_local:
            return {"error": "No local models available"}
        
        # Calculate profit scenarios
        local_profit = data_analysis["high_quality_value"]  # Full value since local is free
        api_profit = data_analysis["high_quality_value"] - cheapest_api["total_cost"]
        
        return {
            "best_local_model": best_local["name"],
            "local_cost": 0.0,
            "local_profit": local_profit,
            "cheapest_api": cheapest_api_name,
            "api_cost": cheapest_api["total_cost"],
            "api_profit": api_profit,
            "profit_advantage_local": local_profit - api_profit,
            "roi_improvement": ((local_profit / max(api_profit, 0.001)) - 1) * 100 if api_profit > 0 else float('inf'),
            "break_even_analysis": {
                "tasks_to_pay_for_hardware": self._calculate_hardware_break_even(cheapest_api["total_cost"]),
                "monthly_savings_estimate": self._estimate_monthly_savings(cheapest_api["total_cost"])
            }
        }
    
    def _calculate_hardware_break_even(self, api_cost_per_task: float) -> int:
        """Calculate how many tasks needed to pay for hardware investment"""
        # Assume $2000 hardware investment for good local setup
        hardware_cost = 2000.0
        return int(hardware_cost / max(api_cost_per_task, 0.001))
    
    def _estimate_monthly_savings(self, api_cost_per_task: float) -> float:
        """Estimate monthly savings assuming typical usage"""
        # Assume 50 tasks per month for active user
        monthly_tasks = 50
        return monthly_tasks * api_cost_per_task
    
    def _generate_recommendations(self, local_suggestions: List[Dict], api_costs: Dict, profit_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent recommendations"""
        
        if not local_suggestions:
            return {"recommendation": "No local models available - consider setting up Ollama"}
        
        best_local = local_suggestions[0]
        
        # Simple decision tree
        if profit_analysis.get("profit_advantage_local", 0) > 0.01:  # More than 1 cent advantage
            recommendation = "strong_local"
            reason = f"Local model '{best_local['name']}' will save ${profit_analysis['profit_advantage_local']:.3f} per task"
        elif best_local["expected_quality"] >= 8.0:
            recommendation = "local_recommended"
            reason = f"Local model '{best_local['name']}' offers excellent quality at zero cost"
        else:
            recommendation = "consider_both"
            reason = "Local model capable, but API might provide higher quality for critical tasks"
        
        return {
            "recommendation": recommendation,
            "reason": reason,
            "primary_choice": best_local["name"],
            "fallback_choice": local_suggestions[1]["name"] if len(local_suggestions) > 1 else None,
            "api_alternative": min(api_costs.keys(), key=lambda x: api_costs[x]["total_cost"]),
            "confidence_score": min(10.0, best_local["task_fit_score"])
        }
    
    def _get_local_advantages_for_task(self, task_type: str) -> List[str]:
        """Get task-specific advantages of local models"""
        base_advantages = [
            "🚀 Zero latency (no network calls)",
            "💰 Zero cost per request",
            "🔒 Complete privacy & data control",
            "⚡ Always available (no rate limits)",
            "🎯 Can be fine-tuned for your specific needs"
        ]
        
        task_specific = {
            "coding": ["🧠 Specialized code models available", "🔄 Iterative debugging without cost"],
            "blender_specific": ["🎨 Domain-specific knowledge can be trained", "🚀 Tight integration with Blender API"],
            "creative": ["✨ Unlimited creative iterations", "🎭 No content policy restrictions"],
            "analysis": ["📊 Can process large datasets locally", "🔍 No data leaves your system"]
        }
        
        return base_advantages + task_specific.get(task_type, [])
    
    def _estimate_output_quality(self, model: Dict, task_type: str) -> float:
        """Estimate output quality for this model/task combo"""
        base_quality = model["performance_score"]
        
        # Task-specific adjustments
        if task_type in model["capabilities"]:
            base_quality += 1.0
        
        # Model-specific bonuses
        name_lower = model["name"].lower()
        if task_type == "coding" and "coder" in name_lower:
            base_quality += 1.5
        
        return min(10.0, base_quality)
    
    def _estimate_quality_expectations(self, task_type: str, local_suggestions: List[Dict]) -> Dict[str, Any]:
        """Estimate quality expectations vs API alternatives"""
        
        if not local_suggestions:
            return {"error": "No local models to analyze"}
        
        best_local = local_suggestions[0]
        local_quality = self._estimate_output_quality(best_local, task_type)
        
        # Compare to typical API quality
        api_quality_benchmark = {
            "coding": 9.0,
            "blender_specific": 7.5,  # APIs less specialized
            "creative": 8.5,
            "analysis": 9.2,
            "problem_solving": 8.8,
            "general": 8.0
        }.get(task_type, 8.0)
        
        quality_gap = api_quality_benchmark - local_quality
        
        return {
            "local_quality_estimate": local_quality,
            "api_quality_benchmark": api_quality_benchmark,
            "quality_gap": quality_gap,
            "local_quality_rating": self._rate_quality(local_quality),
            "recommendation": "excellent" if quality_gap < 1.0 else "good" if quality_gap < 2.0 else "acceptable"
        }
    
    def _rate_quality(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 9.0:
            return "excellent"
        elif score >= 8.0:
            return "very_good"
        elif score >= 7.0:
            return "good"
        elif score >= 6.0:
            return "acceptable"
        else:
            return "limited"
    
    def _calculate_data_value_for_model(self, model: Dict, task_type: str) -> float:
        """Calculate the value of data this model could generate"""
        data_type = self._map_task_to_data_type(task_type)
        base_value = self.data_value_estimates[data_type]["market_value"]
        
        # Adjust based on model quality potential
        quality_factor = model["data_quality_potential"] / 10.0
        
        return base_value * quality_factor
    
    def get_model_dropdown_options(self) -> List[Dict[str, Any]]:
        """Get formatted options for UI dropdown"""
        options = []
        
        for model in self.ollama_models:
            option = {
                "value": model["name"],
                "label": f"{model['name']} ({model['size']}) - {', '.join(model['strengths'])}",
                "performance": model["performance_score"],
                "speed": model["speed_rating"],
                "size": model["size"],
                "capabilities": model["capabilities"]
            }
            options.append(option)
        
        return sorted(options, key=lambda x: x["performance"], reverse=True)
    
    def record_usage_and_feedback(self, model_used: str, task_type: str, user_satisfaction: float, 
                                 response_time: float, output_quality: float):
        """Record actual usage data for improving recommendations"""
        
        timestamp = datetime.now()
        
        usage_record = {
            "timestamp": timestamp,
            "model": model_used,
            "task_type": task_type,
            "user_satisfaction": user_satisfaction,
            "response_time": response_time,
            "output_quality": output_quality
        }
        
        # Store in tracking
        if model_used not in self.local_model_performance:
            self.local_model_performance[model_used] = {
                "usage_history": [],
                "avg_satisfaction": 0.0,
                "avg_response_time": 0.0,
                "avg_quality": 0.0,
                "total_uses": 0
            }
        
        perf = self.local_model_performance[model_used]
        perf["usage_history"].append(usage_record)
        perf["total_uses"] += 1
        
        # Update averages
        recent_records = perf["usage_history"][-50:]  # Last 50 uses
        perf["avg_satisfaction"] = sum(r["user_satisfaction"] for r in recent_records) / len(recent_records)
        perf["avg_response_time"] = sum(r["response_time"] for r in recent_records) / len(recent_records)
        perf["avg_quality"] = sum(r["output_quality"] for r in recent_records) / len(recent_records)
        
        print(f"📊 Recorded feedback for {model_used}: satisfaction={user_satisfaction}/10")
    
    def get_profit_optimization_report(self) -> str:
        """Generate comprehensive profit and optimization report"""
        
        total_uses = sum(perf.get("total_uses", 0) for perf in self.local_model_performance.values())
        
        if total_uses == 0:
            return "No usage data available yet. Start using your local models to see profit analysis!"
        
        # Calculate savings
        avg_api_cost_per_task = 0.015  # Rough average
        total_savings = total_uses * avg_api_cost_per_task
        
        # Calculate data value generated
        estimated_data_value = total_uses * 0.08  # Average data value per task
        
        report = f"""# 🚀 Llammy Local Intelligence Profit Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 💰 FINANCIAL IMPACT:
Total Local Tasks Completed: {total_uses}
Estimated API Costs Avoided: ${total_savings:.2f}
High-Quality Data Generated Value: ${estimated_data_value:.2f}
Total Economic Impact: ${total_savings + estimated_data_value:.2f}

## 🧠 MODEL PERFORMANCE:
"""
        
        for model_name, perf in self.local_model_performance.items():
            if perf.get("total_uses", 0) > 0:
                satisfaction = perf.get("avg_satisfaction", 0)
                quality = perf.get("avg_quality", 0)
                speed = perf.get("avg_response_time", 0)
                uses = perf.get("total_uses", 0)
                
                report += f"""
### {model_name}:
- Uses: {uses} tasks
- User Satisfaction: {satisfaction:.1f}/10
- Quality Score: {quality:.1f}/10  
- Avg Response Time: {speed:.1f}s
- Value Generated: ${uses * 0.08:.2f}
"""
        
        # Calculate ROI
        hardware_investment = 2000  # Assumed hardware cost
        roi_months = hardware_investment / max((total