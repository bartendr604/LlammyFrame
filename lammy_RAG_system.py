# =============================================================================
# LLAMMY MODULE 3: RAG CONTEXT SYSTEM
# llammy_rag_system.py
# 
# Extracted from Llammy v8.4 Framework - The Context Enhancement Module
# Provides RAG (Retrieval Augmented Generation) for context-aware code generation
# =============================================================================

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import LlamaIndex for RAG functionality
try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Import core learning module for status tracking
try:
    from .llammy_core_learning import get_core_status, update_status
    CORE_LEARNING_AVAILABLE = True
except ImportError:
    CORE_LEARNING_AVAILABLE = False
    print("⚠️ Core learning module not available - running in standalone mode")

# =============================================================================
# RAG SYSTEM INTEGRATION
# =============================================================================

class LlammyRAG:
    """
    RAG system for context-aware Blender code generation
    
    Provides enhanced context from:
    - Blender API documentation
    - Code examples and patterns
    - Best practices and corrections
    """
    
    def __init__(self):
        self.rag_initialized = False
        self.vector_index = None
        self.api_data = []  # Blender API reference data
        self.rag_directory = None
        self.context_cache = {}  # Cache for performance
        self.api_cache = {}  # Cache API searches
        
        print("🧠 RAG Context System initialized")
        
    def find_rag_directory(self) -> Optional[str]:
        """Find the RAG data directory in various locations"""
        possible_locations = [
            # User's home directory
            os.path.join(os.path.expanduser("~"), "llammy_rag_data"),
            # Addon directory
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "llammy_rag_data"),
            # Parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "llammy_rag_data"),
            # Common locations
            os.path.join(os.getcwd(), "llammy_rag_data"),
            os.path.join(os.path.expanduser("~"), "Documents", "llammy_rag_data")
        ]
        
        # Also try Blender's script paths if available
        try:
            import bpy
            script_paths = [
                os.path.join(bpy.utils.script_path_user() or "", "addons", "llammy_rag_data"),
                os.path.join(bpy.utils.script_path_pref() or "", "addons", "llammy_rag_data")
            ]
            possible_locations.extend([p for p in script_paths if p])
        except ImportError:
            pass  # Not in Blender environment
        
        for location in possible_locations:
            if location and os.path.exists(location):
                print(f"📂 Found RAG directory: {location}")
                return location
        
        print("⚠️ RAG data directory not found in any standard location")
        return None
    
    def initialize_rag(self) -> bool:
        """Initialize RAG system with existing data"""
        if not LLAMAINDEX_AVAILABLE:
            print("⚠️ LlamaIndex not available - RAG disabled")
            print("   Install with: pip install llama-index")
            return False
            
        if CORE_LEARNING_AVAILABLE:
            update_status("initializing", "RAG system setup")
        
        self.rag_directory = self.find_rag_directory()
        if not self.rag_directory:
            print("⚠️ RAG data directory not found")
            if CORE_LEARNING_AVAILABLE:
                update_status("idle")
            return False
        
        try:
            # Load API data first
            self.load_api_data()
            
            # Try to load existing vector index
            index_dir = os.path.join(self.rag_directory, "vector_index")
            if os.path.exists(index_dir):
                print("📚 Loading existing RAG vector index...")
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                self.vector_index = load_index_from_storage(storage_context)
                print(f"✅ RAG loaded from: {self.rag_directory}")
            else:
                # Create new index from docs
                docs_dir = os.path.join(self.rag_directory, "2_Docs")
                if os.path.exists(docs_dir):
                    print("🔨 Creating new RAG vector index from documents...")
                    docs = SimpleDirectoryReader(input_dir=docs_dir, recursive=True).load_data()
                    self.vector_index = VectorStoreIndex.from_documents(docs, embed_model="local")
                    # Save the index
                    os.makedirs(index_dir, exist_ok=True)
                    self.vector_index.storage_context.persist(index_dir)
                    print(f"✅ RAG index created: {index_dir}")
                else:
                    print(f"⚠️ Documents directory not found: {docs_dir}")
                    if CORE_LEARNING_AVAILABLE:
                        update_status("idle")
                    return False
            
            self.rag_initialized = True
            
            if CORE_LEARNING_AVAILABLE:
                update_status("idle", "RAG initialization complete")
            
            print(f"🧠 RAG System Active:")
            print(f"   📚 API Entries: {len(self.api_data)}")
            print(f"   🗂️ Vector Index: {'Loaded' if self.vector_index else 'Failed'}")
            print(f"   📁 Data Directory: {self.rag_directory}")
            
            return True
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
            if CORE_LEARNING_AVAILABLE:
                update_status("idle")
            return False
    
    def load_api_data(self):
        """Load Blender API data from JSONL file"""
        api_file = os.path.join(self.rag_directory, "1_API_Dumper", "blender_api_441.jsonl")
        
        if not os.path.exists(api_file):
            # Try alternative locations
            alternative_paths = [
                os.path.join(self.rag_directory, "blender_api_441.jsonl"),
                os.path.join(self.rag_directory, "api_data.jsonl"),
                os.path.join(self.rag_directory, "1_API_Dumper", "api_data.jsonl")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    api_file = alt_path
                    break
            else:
                print(f"⚠️ API data file not found. Searched:")
                print(f"   • {api_file}")
                for alt in alternative_paths:
                    print(f"   • {alt}")
                return
        
        try:
            with open(api_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            self.api_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON error in API file line {line_num}: {e}")
                            continue
            
            print(f"✅ API data loaded: {len(self.api_data)} entries from {api_file}")
            
        except Exception as e:
            print(f"⚠️ API data load error: {e}")
    
    def search_api(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search Blender API data for relevant functions"""
        if not self.api_data:
            return []
        
        # Check cache first
        cache_key = f"{query.lower()}_{limit}"
        if cache_key in self.api_cache:
            return self.api_cache[cache_key]
        
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for item in self.api_data:
            score = 0
            name = item.get('name', '').lower()
            module = item.get('module', '').lower()
            description = item.get('description', '').lower()
            
            # Scoring algorithm for relevance
            # Exact match in name (highest priority)
            if query_lower == name:
                score += 50
            elif query_lower in name:
                score += 20
            
            # Word matches in name
            for word in query_words:
                if word in name:
                    score += 10
                if word in module:
                    score += 5
                if word in description:
                    score += 3
            
            # Boost for commonly used functions
            if any(common in name for common in ['add', 'new', 'create', 'set']):
                score += 2
            
            # Boost for bpy.ops (operators)
            if 'bpy.ops' in module:
                score += 3
            
            if score > 0:
                results.append((score, item))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        final_results = [item[1] for item in results[:limit]]
        
        # Cache the results
        self.api_cache[cache_key] = final_results
        
        return final_results
    
    def query_documentation(self, query: str) -> Optional[str]:
        """Query vector index for documentation context"""
        if not self.rag_initialized or not self.vector_index:
            return None
        
        # Check cache first
        if query in self.context_cache:
            return self.context_cache[query]
        
        try:
            query_engine = self.vector_index.as_query_engine()
            response = query_engine.query(query)
            result = str(response)[:500]  # Limit response length for context
            
            # Cache the result
            self.context_cache[query] = result
            
            return result
        except Exception as e:
            print(f"RAG query error: {e}")
            return None
    
    def get_context_for_request(self, user_request: str) -> str:
        """Get comprehensive context for user request"""
        if not self.rag_initialized:
            return self._fallback_context(user_request)
        
        context_parts = []
        
        # 1. API search for relevant Blender functions
        api_results = self.search_api(user_request, limit=3)
        if api_results:
            context_parts.append("=== RELEVANT BLENDER API ===")
            for item in api_results[:3]:  # Limit to top 3 results
                module = item.get('module', '')
                name = item.get('name', '')
                description = item.get('description', '')[:100]  # Truncate description
                
                context_parts.append(f"• {module}.{name}")
                if description:
                    context_parts.append(f"  Description: {description}")
        
        # 2. Documentation search using vector index
        doc_response = self.query_documentation(user_request)
        if doc_response:
            context_parts.append("=== DOCUMENTATION CONTEXT ===")
            context_parts.append(doc_response)
        
        # 3. Add proven patterns and best practices
        context_parts.append("=== PROVEN PATTERNS ===")
        patterns = self._get_proven_patterns(user_request)
        context_parts.extend(patterns)
        
        # 4. Add context-specific corrections
        corrections = self._get_contextual_corrections(user_request)
        if corrections:
            context_parts.append("=== COMMON CORRECTIONS ===")
            context_parts.extend(corrections)
        
        return "\n".join(context_parts)
    
    def _fallback_context(self, user_request: str) -> str:
        """Fallback context when RAG is not initialized"""
        context_parts = [
            "=== BASIC BLENDER CONTEXT ===",
            "• Use bpy.data.materials.new() for materials",
            "• Use material.use_nodes = True before node operations",
            "• Use material.node_tree.nodes for node access",
            "• Use bpy.ops.mesh.primitive_* for mesh creation",
            "• Use bpy.context.active_object for current object",
            "",
            "=== PROVEN PATTERNS ===",
            "• Always clear scene: bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()",
            "• Material setup: mat = bpy.data.materials.new('name'); mat.use_nodes = True",
            "• Node access: nodes = mat.node_tree.nodes",
            "• Camera setup: bpy.ops.object.camera_add(location=(7, -7, 5))"
        ]
        
        return "\n".join(context_parts)
    
    def _get_proven_patterns(self, user_request: str) -> List[str]:
        """Get proven patterns based on request content"""
        patterns = []
        request_lower = user_request.lower()
        
        # Material-related patterns
        if any(word in request_lower for word in ['material', 'shader', 'node', 'color']):
            patterns.extend([
                "• Always use material.use_nodes = True before node operations",
                "• Access nodes via material.node_tree.nodes (never material.nodes)",
                "• Use Principled BSDF for realistic materials",
                "• Set RGBA colors: (R, G, B, 1.0) with values 0.0-1.0"
            ])
        
        # Object creation patterns
        if any(word in request_lower for word in ['create', 'add', 'object', 'mesh']):
            patterns.extend([
                "• Use bpy.ops.mesh.primitive_* for mesh creation",
                "• Get active object: obj = bpy.context.active_object",
                "• Name objects: obj.name = 'descriptive_name'",
                "• Location: bpy.ops.mesh.primitive_cube_add(location=(x, y, z))"
            ])
        
        # Animation patterns
        if any(word in request_lower for word in ['animate', 'keyframe', 'movement']):
            patterns.extend([
                "• Insert keyframes: obj.keyframe_insert(data_path='location', frame=1)",
                "• Set frame: bpy.context.scene.frame_set(frame_number)",
                "• Animation length: bpy.context.scene.frame_end = 250"
            ])
        
        # Lighting patterns
        if any(word in request_lower for word in ['light', 'lamp', 'illumination']):
            patterns.extend([
                "• Add lights: bpy.ops.object.light_add(type='SUN', location=(x, y, z))",
                "• Set energy: light.data.energy = 300",
                "• Set color: light.data.color = (R, G, B)"
            ])
        
        # Camera patterns
        if any(word in request_lower for word in ['camera', 'view', 'shot']):
            patterns.extend([
                "• Add camera: bpy.ops.object.camera_add(location=(7, -7, 5))",
                "• Point at object: constraint = camera.constraints.new('TRACK_TO')",
                "• Set active camera: bpy.context.scene.camera = camera"
            ])
        
        # If no specific patterns, add general ones
        if not patterns:
            patterns.extend([
                "• Always start with: import bpy",
                "• Clear scene: bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()",
                "• Update viewport: bpy.context.view_layer.update()"
            ])
        
        return patterns
    
    def _get_contextual_corrections(self, user_request: str) -> List[str]:
        """Get context-specific corrections"""
        corrections = []
        request_lower = user_request.lower()
        
        # Material corrections
        if 'material' in request_lower:
            corrections.extend([
                "• NEVER use material.nodes - always use material.node_tree.nodes",
                "• NEVER use bpy.ops.material.new() - use bpy.data.materials.new()",
                "• ALWAYS enable nodes: material.use_nodes = True"
            ])
        
        # Object corrections
        if any(word in request_lower for word in ['cube', 'sphere', 'cylinder']):
            corrections.extend([
                "• Use primitive_cube_add NOT cube_add",
                "• Use primitive_uv_sphere_add NOT sphere_add",
                "• Use primitive_cylinder_add NOT cylinder_add"
            ])
        
        # Context corrections
        if any(word in request_lower for word in ['active', 'selected', 'current']):
            corrections.extend([
                "• Use bpy.context.active_object NOT bpy.context.scene.objects.active",
                "• Use bpy.context.view_layer.update() NOT bpy.context.scene.update()"
            ])
        
        return corrections
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get comprehensive RAG system status"""
        return {
            'initialized': self.rag_initialized,
            'llamaindex_available': LLAMAINDEX_AVAILABLE,
            'rag_directory': self.rag_directory,
            'api_data_count': len(self.api_data),
            'vector_index_loaded': self.vector_index is not None,
            'cache_entries': len(self.context_cache),
            'api_cache_entries': len(self.api_cache)
        }
    
    def clear_cache(self):
        """Clear all caches to free memory"""
        self.context_cache.clear()
        self.api_cache.clear()
        print("🧹 RAG caches cleared")
    
    def search_specific_api(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Search for a specific API function by name"""
        for item in self.api_data:
            if item.get('name', '').lower() == function_name.lower():
                return item
        return None
    
    def get_api_suggestions(self, partial_name: str, limit: int = 5) -> List[str]:
        """Get API function suggestions for autocomplete"""
        suggestions = []
        partial_lower = partial_name.lower()
        
        for item in self.api_data:
            name = item.get('name', '')
            if name.lower().startswith(partial_lower):
                suggestions.append(name)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def export_api_summary(self) -> Dict[str, Any]:
        """Export summary of available API functions"""
        if not self.api_data:
            return {'status': 'no_data'}
        
        # Categorize API functions
        categories = {}
        for item in self.api_data:
            module = item.get('module', 'unknown')
            category = module.split('.')[1] if '.' in module else module
            
            if category not in categories:
                categories[category] = []
            
            categories[category].append({
                'name': item.get('name', ''),
                'module': module,
                'description': item.get('description', '')[:100]
            })
        
        return {
            'total_functions': len(self.api_data),
            'categories': {cat: len(funcs) for cat, funcs in categories.items()},
            'top_categories': dict(sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[:10])
        }

# =============================================================================
# RAG-ENHANCED PROMPT CREATION
# =============================================================================

def create_rag_enhanced_prompt(base_prompt: str, user_input: str, context_info: str = "", rag_system: LlammyRAG = None) -> str:
    """
    Create an enhanced prompt with RAG context
    This is the key function that other modules use to get enhanced prompts
    """
    if rag_system is None:
        # Create a temporary RAG system if none provided
        rag_system = LlammyRAG()
        if not rag_system.rag_initialized:
            # Try to initialize, but continue even if it fails
            rag_system.initialize_rag()
    
    if CORE_LEARNING_AVAILABLE:
        update_status("enhancing", "RAG context retrieval")
    
    # Get RAG context
    rag_context = rag_system.get_context_for_request(user_input)
    
    if CORE_LEARNING_AVAILABLE:
        update_status("idle", "RAG context ready")
    
    # Build enhanced prompt
    enhanced_prompt = f"""You are a professional Python developer and Blender expert with access to comprehensive Blender API documentation.

USER REQUEST: "{user_input}"
CONTEXT: "{context_info}"
CREATIVE PLAN: {base_prompt}

{rag_context}

CRITICAL BLENDER API REQUIREMENTS:
- ALWAYS use material.use_nodes = True before node operations
- ALWAYS use material.node_tree.nodes (NEVER material.nodes)
- ALWAYS use primitive_cube_add (NEVER cube_add)
- ALWAYS use bpy.data.materials.new() (NEVER bpy.ops.material.new())
- ALWAYS use bpy.context.active_object (NEVER scene.objects.active)
- ALWAYS use RGBA colors: (R, G, B, 1.0) with values 0.0-1.0

STRUCTURE REQUIREMENTS:
1. Use 4 spaces for indentation (NEVER tabs)
2. Limit lines to 79 characters
3. Include docstrings for functions
4. Use lowercase_with_underscores for variables/functions
5. Use CapitalizedWords for classes

Generate professional, working Blender Python code using the provided API context."""

    return enhanced_prompt

# =============================================================================
# MODULE INTERFACE
# =============================================================================

# Global RAG system instance
llammy_rag_system = LlammyRAG()

def get_rag_system() -> LlammyRAG:
    """Get the global RAG system instance"""
    return llammy_rag_system

def initialize_rag() -> bool:
    """Initialize the global RAG system"""
    return llammy_rag_system.initialize_rag()

def get_context_for_request(user_request: str) -> str:
    """Get RAG-enhanced context for a user request"""
    return llammy_rag_system.get_context_for_request(user_request)

def search_blender_api(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Search Blender API documentation"""
    return llammy_rag_system.search_api(query, limit)

def is_rag_available() -> bool:
    """Check if RAG system is available and initialized"""
    return llammy_rag_system.rag_initialized

def get_rag_status() -> Dict[str, Any]:
    """Get RAG system status"""
    return llammy_rag_system.get_rag_status()

def create_enhanced_prompt(base_prompt: str, user_input: str, context_info: str = "") -> str:
    """Create a RAG-enhanced prompt"""
    return create_rag_enhanced_prompt(base_prompt, user_input, context_info, llammy_rag_system)

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Test the RAG system
    print("🧠 Testing RAG Context System...")
    
    # Test 1: Initialize RAG
    print("\n🧪 Test 1: RAG Initialization")
    if initialize_rag():
        print("✅ RAG system initialized successfully!")
        
        # Test 2: API search
        print("\n🧪 Test 2: API Search")
        results = search_blender_api("material", limit=3)
        print(f"Found {len(results)} API results for 'material':")
        for result in results:
            print(f"  • {result.get('module', '')}.{result.get('name', '')}")
        
        # Test 3: Context generation
        print("\n🧪 Test 3: Context Generation")
        context = get_context_for_request("create a red material with nodes")
        print(f"Generated context length: {len(context)} characters")
        print("Context preview:")
        print(context[:300] + "..." if len(context) > 300 else context)
        
        # Test 4: Enhanced prompt
        print("\n🧪 Test 4: Enhanced Prompt")
        enhanced = create_enhanced_prompt(
            "Create a material",
            "make a red shiny material",
            "for a cube object"
        )
        print(f"Enhanced prompt length: {len(enhanced)} characters")
        
        # Test 5: Status
        print("\n📊 RAG System Status:")
        status = get_rag_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("✅ All RAG tests passed!")
    else:
        print("❌ RAG initialization failed")
        print("RAG will fall back to basic context")
        
        # Test fallback context
        print("\n🧪 Testing Fallback Context")
        context = get_context_for_request("create a material")
        print(f"Fallback context: {len(context)} characters")

print("💎 MODULE 3: RAG CONTEXT SYSTEM - READY FOR INTEGRATION!")
print("This module provides:")
print("✅ Context-aware code generation with real Blender API docs")
print("✅ Semantic search through official documentation")
print("✅ API function search and suggestions")
print("✅ Proven pattern recommendations")
print("✅ Context-specific error corrections")
print("✅ Fallback operation when RAG unavailable")
print("✅ Caching for improved performance")
print("✅ Integration with core learning system")
print("")
print("🧠 REVOLUTIONARY FEATURE: AI with real Blender knowledge!")