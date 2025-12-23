#!/usr/bin/env python3

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
import math

def solver(env) -> Dict:
    """
    Enhanced MWVRP solver with multi-order routing and optimized vehicle utilization.
    Includes comprehensive edge case handling and validation.
    """
    
    def bfs_path(start_node_id: int, end_node_id: int, adj_list: Dict) -> Optional[List[int]]:
        """Find shortest path using BFS with validation."""
        if start_node_id == end_node_id:
            return [start_node_id]
        
        queue = deque([(start_node_id, [start_node_id])])
        visited = {start_node_id}
        
        while queue:
            current_node, path = queue.popleft()
            for neighbor in adj_list.get(current_node, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == end_node_id:
                        return new_path
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)
        return None

    def calculate_path_distance(path: List[int], env) -> float:
        """Calculate actual distance for a path using env.get_distance."""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            distance = env.get_distance(path[i], path[i+1])
            if distance is None:
                # If no direct connection, this path is invalid
                return float('inf')
            total_distance += distance
        return total_distance

    def calculate_order_weight_volume(order_id: str, env) -> Tuple[float, float]:
        """Calculate total weight and volume for an order."""
        requirements = env.get_order_requirements(order_id)
        total_weight = 0.0
        total_volume = 0.0
        
        for sku_id, quantity in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            total_weight += sku_details['weight'] * quantity
            total_volume += sku_details['volume'] * quantity
            
        return total_weight, total_volume

    def can_fulfill_order(order_id: str, warehouse_id: str, 
                         warehouse_inventories: Dict, allocated_inventory: Dict) -> bool:
        """Check if warehouse can fulfill order considering allocated inventory."""
        order_reqs = env.get_order_requirements(order_id)
        available_inv = warehouse_inventories[warehouse_id].copy()
        
        # Subtract allocated inventory
        for sku_id, allocated_qty in allocated_inventory[warehouse_id].items():
            available_inv[sku_id] = available_inv.get(sku_id, 0) - allocated_qty
        
        for sku_id, req_qty in order_reqs.items():
            if available_inv.get(sku_id, 0) < req_qty:
                return False
        return True

    def allocate_order_inventory(order_id: str, warehouse_id: str,
                               allocated_inventory: Dict) -> None:
        """Allocate inventory for an order to prevent race conditions."""
        order_reqs = env.get_order_requirements(order_id)
        for sku_id, req_qty in order_reqs.items():
            allocated_inventory[warehouse_id][sku_id] = allocated_inventory[warehouse_id].get(sku_id, 0) + req_qty

    def deallocate_order_inventory(order_id: str, warehouse_id: str,
                                 allocated_inventory: Dict) -> None:
        """Deallocate inventory when order assignment fails."""
        order_reqs = env.get_order_requirements(order_id)
        for sku_id, req_qty in order_reqs.items():
            allocated_inventory[warehouse_id][sku_id] = allocated_inventory[warehouse_id].get(sku_id, 0) - req_qty

    def find_best_warehouse_for_order(order_id: str, warehouses: List[str], 
                                    warehouse_inventories: Dict, 
                                    order_locations: Dict, 
                                    warehouse_locations: Dict,
                                    adj_list: Dict,
                                    allocated_inventory: Dict) -> Optional[str]:
        """Find the best warehouse to fulfill an order considering allocated inventory and distance."""
        order_reqs = env.get_order_requirements(order_id)
        order_loc = order_locations[order_id]
        best_warehouse = None
        min_distance = float('inf')
        
        for wh_id in warehouses:
            # Check inventory availability considering allocations
            if not can_fulfill_order(order_id, wh_id, warehouse_inventories, allocated_inventory):
                continue
                
            # Calculate distance
            wh_loc = warehouse_locations[wh_id]
            path = bfs_path(wh_loc, order_loc, adj_list)
            if path and len(path) < min_distance:
                min_distance = len(path)
                best_warehouse = wh_id
                
        return best_warehouse

    def build_multi_order_route(vehicle, assigned_orders: List[str], 
                              warehouse_locations: Dict, order_locations: Dict,
                              order_requirements: Dict, adj_list: Dict,
                              order_capacities: Dict) -> Optional[Dict]:
        """Build an optimized route with multiple orders for a single vehicle with comprehensive validation."""
        
        wh_id = vehicle.home_warehouse_id
        wh_node = warehouse_locations[wh_id]
        current_node = wh_node
        steps = []
        
        # Calculate total capacity needed
        total_weight = 0.0
        total_volume = 0.0
        for order_id in assigned_orders:
            total_weight += order_capacities[order_id]['weight']
            total_volume += order_capacities[order_id]['volume']
        
        # Check capacity with epsilon for floating point precision
        capacity_epsilon = 1e-6
        if (total_weight > vehicle.capacity_weight + capacity_epsilon or 
            total_volume > vehicle.capacity_volume + capacity_epsilon):
            return None
        
        # Step 1: Start at warehouse - pick up ALL items for all assigned orders
        all_pickups = []
        for order_id in assigned_orders:
            order_reqs = order_requirements[order_id]
            for sku_id, quantity in order_reqs.items():
                all_pickups.append({
                    'warehouse_id': wh_id,
                    'sku_id': sku_id, 
                    'quantity': quantity
                })
        
        steps.append({
            'node_id': wh_node,
            'pickups': all_pickups,
            'deliveries': [],
            'unloads': []
        })
        
        # Step 2: Visit each order location in optimized sequence
        remaining_orders = assigned_orders.copy()
        total_route_distance = 0.0
        last_node = wh_node
        
        while remaining_orders:
            # Find closest order from current position
            closest_order = None
            min_path_distance = float('inf')
            best_path = None
            
            for order_id in remaining_orders:
                order_node = order_locations[order_id]
                path = bfs_path(current_node, order_node, adj_list)
                if path:
                    path_distance = calculate_path_distance(path, env)
                    if path_distance < min_path_distance:
                        min_path_distance = path_distance
                        closest_order = order_id
                        best_path = path
            
            if not closest_order:
                # No path found to any remaining order - abort route
                return None
                
            # Check distance constraint
            # Estimate return distance from this point
            return_path = bfs_path(order_locations[closest_order], wh_node, adj_list)
            if return_path:
                return_distance = calculate_path_distance(return_path, env)
                estimated_total_distance = total_route_distance + min_path_distance + return_distance
                if estimated_total_distance > vehicle.max_distance:
                    # Would exceed max distance - skip this order
                    remaining_orders.remove(closest_order)
                    continue
            else:
                # No return path - skip this order
                remaining_orders.remove(closest_order)
                continue
            
            # Add path to closest order (excluding start node)
            for node_id in best_path[1:]:
                # Avoid duplicate consecutive nodes
                if not steps or steps[-1]['node_id'] != node_id:
                    steps.append({
                        'node_id': node_id,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
            
            # Deliver at order location
            order_reqs = order_requirements[closest_order]
            deliveries = []
            for sku_id, quantity in order_reqs.items():
                deliveries.append({
                    'order_id': closest_order,
                    'sku_id': sku_id,
                    'quantity': quantity
                })
            
            # Ensure delivery happens at correct node
            delivery_node = order_locations[closest_order]
            if steps and steps[-1]['node_id'] == delivery_node:
                # Add delivery to existing step
                steps[-1]['deliveries'] = deliveries
            else:
                # Create new step for delivery
                steps.append({
                    'node_id': delivery_node,
                    'pickups': [],
                    'deliveries': deliveries,
                    'unloads': []
                })
            
            total_route_distance += min_path_distance
            current_node = order_locations[closest_order]
            remaining_orders.remove(closest_order)
        
        # If no orders could be delivered, return None
        if not any(step.get('deliveries') for step in steps):
            return None
        
        # Step 3: Return to warehouse
        return_path = bfs_path(current_node, wh_node, adj_list)
        if return_path:
            return_distance = calculate_path_distance(return_path, env)
            # Check final distance constraint
            if total_route_distance + return_distance > vehicle.max_distance:
                return None
                
            for node_id in return_path[1:]:
                # Avoid duplicate consecutive nodes
                if not steps or steps[-1]['node_id'] != node_id:
                    steps.append({
                        'node_id': node_id,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
        else:
            # No return path - invalid route
            return None
        
        route = {
            'vehicle_id': vehicle.id,
            'steps': steps
        }
        
        return route

    def build_fallback_routes(vehicle, candidate_orders: List[str], 
                            warehouse_locations: Dict, order_locations: Dict,
                            order_requirements: Dict, adj_list: Dict,
                            order_capacities: Dict) -> Optional[Dict]:
        """Try building routes with smaller order subsets if full route fails."""
        
        # Try different subset sizes, from largest to smallest
        for subset_size in range(len(candidate_orders), 0, -1):
            # Generate combinations (simplified - try first N orders sorted by size)
            sorted_orders = sorted(candidate_orders, 
                                 key=lambda o: order_capacities[o]['weight'], 
                                 reverse=True)
            
            for i in range(len(sorted_orders) - subset_size + 1):
                test_orders = sorted_orders[i:i + subset_size]
                route = build_multi_order_route(
                    vehicle, test_orders, warehouse_locations,
                    order_locations, order_requirements, adj_list, order_capacities
                )
                if route:
                    return route, test_orders
        return None, []

    # --- MAIN SOLVER LOGIC ---
    
    # Get environment data
    orders = env.orders
    warehouses = env.warehouses
    vehicles = env.get_all_vehicles()
    road_network = env.get_road_network_data()
    adj_list = road_network['adjacency_list']
    
    # Precompute data structures
    warehouse_locations = {wh_id: wh.location.id for wh_id, wh in warehouses.items()}
    warehouse_inventories = {wh_id: env.get_warehouse_inventory(wh_id) for wh_id in warehouses}
    order_locations = {order_id: env.get_order_location(order_id) for order_id in orders}
    order_requirements = {order_id: env.get_order_requirements(order_id) for order_id in orders}
    
    # Track allocated inventory to prevent race conditions
    allocated_inventory = {
        wh_id: {sku_id: 0 for sku_id in env.skus} 
        for wh_id in warehouses
    }
    
    # Calculate order weights and volumes
    order_capacities = {}
    for order_id in orders:
        weight, volume = calculate_order_weight_volume(order_id, env)
        order_capacities[order_id] = {'weight': weight, 'volume': volume}
    
    # Group vehicles by warehouse
    vehicles_by_warehouse = {}
    for vehicle in vehicles:
        wh_id = vehicle.home_warehouse_id
        if wh_id not in vehicles_by_warehouse:
            vehicles_by_warehouse[wh_id] = []
        vehicles_by_warehouse[wh_id].append(vehicle)
    
    # Assign orders to warehouses based on inventory and proximity
    warehouse_orders = {wh_id: [] for wh_id in warehouses}
    unassigned_orders = set(orders.keys())
    
    # First pass: assign orders to warehouses
    for order_id in list(unassigned_orders):
        best_wh = find_best_warehouse_for_order(
            order_id, list(warehouses.keys()), warehouse_inventories,
            order_locations, warehouse_locations, adj_list, allocated_inventory
        )
        if best_wh:
            warehouse_orders[best_wh].append(order_id)
            allocate_order_inventory(order_id, best_wh, allocated_inventory)
            unassigned_orders.remove(order_id)
    
    solution_routes = []
    used_vehicles = set()
    fulfilled_orders = set()
    
    # Build routes for each warehouse
    for wh_id, wh_orders in warehouse_orders.items():
        if not wh_orders:
            continue
            
        available_vehicles = [v for v in vehicles_by_warehouse.get(wh_id, []) 
                            if v.id not in used_vehicles]
        
        # Sort vehicles by capacity (largest first)
        available_vehicles.sort(key=lambda v: v.capacity_weight, reverse=True)
        # Sort orders by size (mix large and small for better packing)
        wh_orders.sort(key=lambda o: order_capacities[o]['weight'], reverse=True)
        
        for vehicle in available_vehicles:
            if not wh_orders:
                break
                
            # Select orders that fit in vehicle capacity
            vehicle_candidate_orders = []
            current_weight = 0.0
            current_volume = 0.0
            capacity_epsilon = 1e-6
            
            # Try to pack orders efficiently - mix of large and small
            temp_orders = wh_orders.copy()
            while temp_orders:
                best_fit = None
                best_fit_remaining_capacity = float('inf')
                
                for order_id in temp_orders:
                    order_weight = order_capacities[order_id]['weight']
                    order_volume = order_capacities[order_id]['volume']
                    
                    weight_ok = current_weight + order_weight <= vehicle.capacity_weight + capacity_epsilon
                    volume_ok = current_volume + order_volume <= vehicle.capacity_volume + capacity_epsilon
                    
                    if weight_ok and volume_ok:
                        # Calculate remaining capacity (prefer orders that fill capacity well)
                        remaining_cap = (vehicle.capacity_weight - (current_weight + order_weight)) + \
                                      (vehicle.capacity_volume - (current_volume + order_volume))
                        if remaining_cap < best_fit_remaining_capacity:
                            best_fit_remaining_capacity = remaining_cap
                            best_fit = order_id
                
                if best_fit:
                    vehicle_candidate_orders.append(best_fit)
                    current_weight += order_capacities[best_fit]['weight']
                    current_volume += order_capacities[best_fit]['volume']
                    temp_orders.remove(best_fit)
                else:
                    break
            
            if vehicle_candidate_orders:
                # Build multi-order route with fallback mechanism
                route = build_multi_order_route(
                    vehicle, vehicle_candidate_orders, warehouse_locations,
                    order_locations, order_requirements, adj_list, order_capacities
                )
                
                # If route building fails, try smaller subsets
                if not route:
                    route, successful_orders = build_fallback_routes(
                        vehicle, vehicle_candidate_orders, warehouse_locations,
                        order_locations, order_requirements, adj_list, order_capacities
                    )
                    if route:
                        vehicle_candidate_orders = successful_orders
                
                if route:
                    # Validate route before adding
                    is_valid, validation_msg = env.validator.validate_route_steps(
                        vehicle.id, route['steps']
                    )
                    if is_valid:
                        solution_routes.append(route)
                        used_vehicles.add(vehicle.id)
                        fulfilled_orders.update(vehicle_candidate_orders)
                        
                        # Remove successfully assigned orders from warehouse list
                        for order_id in vehicle_candidate_orders:
                            if order_id in wh_orders:
                                wh_orders.remove(order_id)
                    else:
                        # Validation failed - deallocate inventory for these orders
                        for order_id in vehicle_candidate_orders:
                            deallocate_order_inventory(order_id, wh_id, allocated_inventory)
                else:
                    # Route building failed - deallocate inventory
                    for order_id in vehicle_candidate_orders:
                        deallocate_order_inventory(order_id, wh_id, allocated_inventory)
    
    # Construct final solution
    solution = {'routes': solution_routes}
    
    print(f"Assigned {len(fulfilled_orders)} orders using {len(used_vehicles)} vehicles")
    if unassigned_orders:
        print(f"Could not assign {len(unassigned_orders)} orders")
    
    return solution

if __name__ == '__main__':
    env = LogisticsEnvironment()
    result = solver(env)