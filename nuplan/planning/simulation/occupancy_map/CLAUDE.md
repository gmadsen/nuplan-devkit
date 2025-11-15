# nuplan/planning/simulation/occupancy_map/

## 1. Purpose & Responsibility

This module provides **spatial indexing** for efficient collision detection, occupancy queries, and geometric intersection tests during simulation. The `OccupancyMap` abstraction enables O(log N) spatial queries instead of O(N²) pairwise checks, critical for multi-agent scenarios with 100+ objects. The primary implementation uses STRTree (R-tree variant) from Shapely for bounding box intersection tests.

## 2. Key Abstractions

### Core Concepts

**OccupancyMap** (Abstract Interface)
- **Purpose**: Spatial index for efficient "what objects intersect this geometry?" queries
- **Key operations**:
  - `insert(token, geometry)` - Add object to spatial index
  - `intersects(geometry) -> Set[token]` - Find all objects intersecting query geometry
  - `remove(token)` - Remove object from index
  - `get(token) -> geometry` - Retrieve object geometry by token
- **Use cases**:
  - IDM agent lead vehicle detection (find agents along path)
  - Collision checking (does ego footprint intersect any object?)
  - Stop line queries (which stop lines are near ego?)

**STRTreeOccupancyMap** (Concrete Implementation)
- **Purpose**: R-tree based spatial index (Sort-Tile-Recursive tree)
- **Performance**: O(log N) intersection queries, O(N log N) build time
- **Limitations**: Immutable after construction (rebuild required for insertions/removals)
- **Workaround**: Batch insertions, rebuild periodically

**Agent Tokens**
- Objects identified by unique string tokens (e.g., `agent.track_token`)
- Tokens map to Shapely geometries (Polygon, LineString, Point)
- Token-based API enables efficient updates (remove old, insert new)

### Key Classes

```python
class AbstractOccupancyMap(ABC):
    """
    Abstract spatial index for collision detection.
    """
    @abstractmethod
    def insert(self, token: str, geometry: BaseGeometry):
        """Add object to spatial index"""
    
    @abstractmethod
    def intersects(self, geometry: BaseGeometry) -> Set[str]:
        """Find all object tokens intersecting geometry"""
    
    @abstractmethod
    def remove(self, token: str):
        """Remove object from index"""
    
    @abstractmethod
    def get(self, token: str) -> Optional[BaseGeometry]:
        """Retrieve geometry by token"""

class STRTreeOccupancyMap(AbstractOccupancyMap):
    """
    R-tree based occupancy map using Shapely STRTree.
    High-performance spatial queries, but immutable after construction.
    """
    def __init__(self):
        self._token_to_geometry: Dict[str, BaseGeometry] = {}
        self._tree: Optional[STRTree] = None
        self._dirty = True  # Needs rebuild
    
    def _rebuild_tree(self):
        """Rebuild STRTree from current geometries"""
        self._tree = STRTree(self._token_to_geometry.values())
        self._dirty = False
```

## 3. Architecture & Design Patterns

### Design Patterns

**Abstract Factory Pattern**
- `AbstractOccupancyMap` defines interface
- Concrete implementations (STRTree, grid-based, etc.) are interchangeable
- Enables switching spatial index strategies without changing client code

**Lazy Evaluation**
- STRTree rebuild deferred until first query (after insertions)
- `_dirty` flag tracks when rebuild needed
- Amortizes O(N log N) rebuild cost across many insertions

**Immutable Spatial Index**
- STRTree is immutable after construction (Shapely limitation)
- Workaround: Batch updates, then rebuild
- Trade-off: Fast queries (O(log N)) vs slow updates (O(N log N) rebuild)

### Relationships

```
Simulation Components
    ↓
AbstractOccupancyMap
    ├─ STRTreeOccupancyMap (R-tree)
    ├─ GridOccupancyMap (grid-based, if implemented)
    └─ Used by:
        ├─ IDM agents (lead agent search)
        ├─ Collision checkers (ego vs objects)
        └─ Path validators (geometric feasibility)
```

### Typical Usage Flow (IDM Agent Lead Search)

```
IDM Agent Propagation:
  1. Get agent path (InterpolatedPath)
  2. Extract path geometry (LineString)
  3. Query occupancy map:
       intersecting_tokens = occupancy_map.intersects(path_geometry)
  4. Filter tokens (only agents ahead, same direction)
  5. Find nearest intersecting agent → lead agent
  6. Compute IDM acceleration based on lead agent
```

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- (None - pure geometry module)

**Indirect Dependencies**:
- ✅ `nuplan.common.actor_state.tracked_objects` - Agent geometries (via footprint)
- ✅ `nuplan.common.maps.abstract_map_objects` - StopLine geometries

### External

- **Shapely** - Geometry library
  - `shapely.geometry` - Polygon, LineString, Point, MultiPolygon
  - `shapely.strtree.STRTree` - R-tree spatial index
  - `shapely.ops.unary_union` - Merge overlapping geometries
- `typing` - Set, Dict, Optional type hints

### Dependency Notes

**AIDEV-NOTE**: OccupancyMap is geometry-only - no nuPlan-specific logic. Could be used standalone for spatial indexing tasks.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**IDM Agents**:
- ✅ `nuplan/planning/simulation/observation/idm/idm_agent_manager.py` - Lead agent detection
  - Insert all agents + ego + stop lines into occupancy map
  - Query with agent path to find intersecting objects
  - Filter to get nearest lead agent

**Collision Checkers**:
- Planners that validate trajectory collision-free
- Ego footprint vs static/dynamic obstacles

**Stop Line Handling**:
- IDM agents insert stop lines for red lights
- Query occupancy map to find stop lines along agent path
- Compute braking distance to stop line

### Use Cases

1. **IDM Agent Lead Search** (Primary Use Case)
   ```python
   # Insert all agents into occupancy map
   for agent in agents:
       occupancy_map.insert(agent.track_token, agent.polygon)
   
   # Query with path geometry
   path_line = agent.get_path_to_go().linestring
   intersecting_tokens = occupancy_map.intersects(path_line)
   
   # Filter to find lead agent (nearest ahead)
   lead_agent = find_nearest_agent(intersecting_tokens, current_progress)
   ```

2. **Collision Validation**
   ```python
   # Check if ego trajectory collides with any object
   for state in planned_trajectory:
       ego_footprint = get_ego_polygon(state)
       collisions = occupancy_map.intersects(ego_footprint)
       if collisions:
           return False  # Collision detected
   return True  # Collision-free
   ```

3. **Stop Line Detection**
   ```python
   # Insert stop lines for red lights
   for stop_line in red_light_stop_lines:
       occupancy_map.insert(stop_line.id, stop_line.polygon)
   
   # Query with agent path
   stop_lines_ahead = occupancy_map.intersects(agent_path)
   
   # Find nearest stop line
   nearest_stop = min(stop_lines_ahead, key=lambda sl: distance_to_stop(sl))
   ```

4. **Spatial Filtering**
   ```python
   # Find all agents within radius of ego
   ego_position = ego_state.center
   query_circle = Point(ego_position.x, ego_position.y).buffer(radius)
   nearby_agents = occupancy_map.intersects(query_circle)
   ```

**AIDEV-NOTE**: OccupancyMap is essential for performance - without it, IDM agents would need O(N²) pairwise distance checks every timestep.

## 6. Critical Files (Prioritized)

### Priority 1: Core Implementation

1. **`abstract_occupancy_map.py`**
   - AbstractOccupancyMap interface
   - API contract for spatial index
   - **Key for**: Understanding occupancy map operations

2. **`strtree_occupancy_map.py`**
   - STRTreeOccupancyMap concrete implementation
   - R-tree construction and queries
   - **Key for**: Primary implementation details

### Priority 2: Utilities

3. **`occupancy_map_factory.py`** (if exists)
   - Factory functions for creating occupancy maps
   - Configuration-based instantiation
   - **Key for**: How to create occupancy maps

4. **Test files** (`test/` directory)
   - Intersection query tests
   - Performance benchmarks
   - Edge case validation
   - **Key for**: Expected behavior

5. **`__init__.py`**
   - Module exports
   - Public API surface

**AIDEV-NOTE**: Module is small (~200-400 lines total). Start with abstract_occupancy_map.py, then strtree_occupancy_map.py.

## 7. Common Usage Patterns

### Creating Occupancy Map

```python
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap

# Create empty occupancy map
occupancy_map = STRTreeOccupancyMap()
```

### Inserting Objects

```python
from shapely.geometry import Polygon, Point

# Insert agent footprint
agent_polygon = Polygon([
    (0, 0), (4, 0), (4, 2), (0, 2)  # 4m × 2m box
])
occupancy_map.insert(token="agent_123", geometry=agent_polygon)

# Insert ego footprint
ego_polygon = get_ego_footprint(ego_state)
occupancy_map.insert(token="ego", geometry=ego_polygon)

# Insert stop line
stop_line_geom = LineString([(10, 0), (10, 10)])
occupancy_map.insert(token="stop_line_456", geometry=stop_line_geom)
```

### Querying Intersections

```python
from shapely.geometry import LineString

# Query with path geometry
path_geometry = LineString([
    (0, 1), (10, 1), (20, 1)  # Straight path
])

intersecting_tokens = occupancy_map.intersects(path_geometry)

# Returns: {"agent_123", "stop_line_456"} if they intersect path

# Check for specific token
if "agent_123" in intersecting_tokens:
    print("Agent 123 intersects path")
```

### Removing Objects

```python
# Remove agent from occupancy map
occupancy_map.remove(token="agent_123")

# Verify removal
assert "agent_123" not in occupancy_map.intersects(path_geometry)
```

### Retrieving Geometry

```python
# Get geometry by token
agent_geom = occupancy_map.get(token="agent_123")

if agent_geom:
    print(f"Agent bounds: {agent_geom.bounds}")  # (minx, miny, maxx, maxy)
```

### IDM Agent Lead Search Pattern

```python
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap

# Initialize occupancy map
occupancy_map = STRTreeOccupancyMap()

# Insert ego
ego_polygon = get_ego_footprint(ego_state)
occupancy_map.insert("ego", ego_polygon)

# Insert all IDM agents
for agent in idm_agents:
    agent_polygon = agent.projected_footprint  # Future occupancy
    occupancy_map.insert(agent.track_token, agent_polygon)

# Insert stop lines (for red lights)
for stop_line in red_light_stop_lines:
    occupancy_map.insert(stop_line.id, stop_line.polygon)

# Query with agent path
agent_path = agent.get_path_to_go()  # InterpolatedPath
path_geometry = agent_path.linestring  # Convert to Shapely LineString

intersecting_tokens = occupancy_map.intersects(path_geometry)

# Filter intersecting objects
lead_candidates = []
for token in intersecting_tokens:
    if token == "ego":
        lead_candidates.append(ego_state)
    elif token.startswith("stop_line"):
        lead_candidates.append(occupancy_map.get(token))
    else:
        # Another IDM agent
        lead_candidates.append(idm_agents[token])

# Find nearest object along path
lead_agent = find_nearest_along_path(lead_candidates, agent_path, agent.state.progress)
```

### Batch Updates (Efficient Pattern)

```python
# BAD: Rebuild tree after every insertion (slow)
for agent in agents:
    occupancy_map.insert(agent.track_token, agent.polygon)
    # STRTree rebuilt N times → O(N² log N)

# GOOD: Batch insertions, rebuild once
for agent in agents:
    occupancy_map._token_to_geometry[agent.track_token] = agent.polygon
    occupancy_map._dirty = True

# Rebuild happens on first query (amortized cost)
intersecting_tokens = occupancy_map.intersects(query_geometry)
```

## 8. Gotchas & Edge Cases

### 1. **STRTree Immutability**
- **Issue**: STRTree cannot be updated incrementally (Shapely limitation)
- **Symptom**: Every insert/remove triggers O(N log N) rebuild
- **Workaround**: Batch updates, rebuild once
- **AIDEV-NOTE**: Consider alternative spatial index if many incremental updates needed

### 2. **Rebuild Performance**
- **Issue**: Large occupancy maps (1000+ objects) rebuild slowly
- **Symptom**: Simulation lags after insertions
- **Fix**: Profile rebuild time, reduce object count if possible

### 3. **Empty Occupancy Map**
- **Issue**: Querying empty map (no objects inserted)
- **Symptom**: STRTree is None, query crashes
- **Fix**: Return empty set if `_tree is None` or `_dirty = True`

### 4. **Duplicate Tokens**
- **Issue**: Inserting same token twice without removing first
- **Symptom**: Geometry overwritten, old geometry orphaned in tree
- **Fix**: Remove old token before inserting new: `remove(token); insert(token, new_geom)`

### 5. **Invalid Geometries**
- **Issue**: Inserting self-intersecting polygon or degenerate geometry
- **Symptom**: STRTree construction fails or returns incorrect results
- **Fix**: Validate geometries before insertion: `geometry.is_valid`

### 6. **Geometry Type Mismatch**
- **Issue**: Expecting Polygon, but get LineString or Point
- **Symptom**: Intersection logic incorrect (e.g., Point doesn't block path)
- **Fix**: Document expected geometry types per token type

### 7. **Token Not Found**
- **Issue**: Calling `remove(token)` or `get(token)` for non-existent token
- **Symptom**: KeyError
- **Fix**: Check `if token in occupancy_map._token_to_geometry` before remove/get

### 8. **Precision Issues in Intersection**
- **Issue**: Geometries touch but don't overlap (floating-point precision)
- **Symptom**: Expected intersection not detected
- **Fix**: Use tolerance: `geometry.buffer(1e-6)` before query

### 9. **Memory Leak from Unbounded Growth**
- **Issue**: Inserting objects every timestep without removing old
- **Symptom**: Occupancy map grows unbounded, memory leak
- **Fix**: Remove stale objects: `remove(token)` when object no longer relevant

### 10. **Intersection vs Contains Confusion**
- **Issue**: `intersects(geometry)` returns objects that overlap, not fully contained
- **Symptom**: Expected only fully-contained objects
- **Fix**: Post-filter with `geometry.contains(obj_geom)` if needed

### 11. **Coordinate System Mismatch**
- **Issue**: Query geometry in different coordinate frame than inserted geometries
- **Symptom**: No intersections found despite objects overlapping visually
- **Fix**: Transform all geometries to same coordinate frame (global or ego-relative)

### 12. **Stop Line Insertion/Removal Overhead**
- **Issue**: IDM agents insert/remove stop lines for EVERY agent EVERY timestep
- **Symptom**: Rebuild overhead dominates runtime
- **Fix**: Insert all stop lines once, remove once (batch operation)
- **AIDEV-NOTE**: See `idm_agent_manager.py:70-72` for optimization opportunity

## 9. Performance Considerations

**Computational Cost**:
- `insert(token, geometry)`: O(1) dict update + mark dirty
- `intersects(geometry)`: O(log N) query (if tree built) + O(N log N) rebuild (if dirty)
- `remove(token)`: O(1) dict removal + mark dirty
- `get(token)`: O(1) dict lookup

**Rebuild Cost**:
- STRTree construction: O(N log N)
- Amortized via lazy rebuild (only on first query after insertions)

**Memory Usage**:
- `_token_to_geometry` dict: O(N) × geometry size
- STRTree: O(N) bounding boxes (~64 bytes per object)
- Example: 100 agents → ~10 KB

**Scaling**:
- 10 agents: Negligible overhead
- 100 agents: ~1-5 ms rebuild time
- 1000 agents: ~50-100 ms rebuild time (may impact real-time constraint)

**Optimization Strategies**:
1. **Batch updates**: Insert all objects, rebuild once
2. **Spatial filtering**: Only insert objects within radius of ego
3. **Geometry simplification**: Use bounding boxes instead of detailed polygons
4. **Alternative index**: Grid-based occupancy map for very dynamic scenarios

**AIDEV-NOTE**: For typical scenarios (<100 agents), STRTree is fast enough. Consider grid-based index if >500 agents.

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/simulation/observation/idm/CLAUDE.md` - Primary consumer (lead agent search)
- ✅ `nuplan/planning/simulation/path/CLAUDE.md` - Path geometries for queries
- ✅ `nuplan/common/actor_state/CLAUDE.md` - Agent geometries (oriented boxes)
- ✅ `nuplan/common/maps/CLAUDE.md` - Stop line geometries
- `nuplan/planning/simulation/controller/CLAUDE.md` - Collision checking (Phase 2B)

### External Resources
- **Shapely STRTree**: https://shapely.readthedocs.io/en/stable/strtree.html
- **R-tree spatial index**: https://en.wikipedia.org/wiki/R-tree
- **Shapely geometry**: https://shapely.readthedocs.io/en/stable/geometry.html

## 11. AIDEV Notes

**Design Philosophy**:
- Occupancy map is a "pure" spatial index - no simulation logic
- STRTree is fast for queries (O(log N)), slow for updates (O(N log N))
- Lazy rebuild amortizes cost across many insertions

**Common Mistakes**:
- Forgetting to remove old objects → memory leak
- Inserting same token twice without removing → orphaned geometries
- Not batching updates → excessive rebuilds

**Future Improvements**:
- **AIDEV-TODO**: Implement incremental spatial index (R* tree, Quad tree)
- **AIDEV-TODO**: Add grid-based occupancy map for highly dynamic scenarios
- **AIDEV-TODO**: Profile rebuild overhead in large-scale simulations (1000+ agents)
- **AIDEV-TODO**: Add geometry validation (check `is_valid` before insert)

**AIDEV-NOTE**: If rebuild overhead becomes bottleneck, consider:
1. Grid-based occupancy map (O(1) insert/remove, O(N) query worst case)
2. Hybrid approach (R-tree for static objects, grid for dynamic)
3. Spatial hashing (O(1) insert/remove/query average case)

**AIDEV-QUESTION**: Should occupancy map support time-parameterized queries? (e.g., "what objects will be here at t=5s?") Currently only supports spatial queries, not temporal.
