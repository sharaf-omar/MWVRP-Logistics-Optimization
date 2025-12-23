"""
2025-12-10 17:50:23,612 - INFO - Starting ACO optimization with 500 ants and 5 iterations
2025-12-10 17:50:23,612 - INFO - Iteration 1/5
2025-12-10 17:50:48,102 - INFO -   Valid solutions: 500/500
2025-12-10 17:50:48,102 - INFO -   Avg cost: £4878.21, Avg fulfillment: 98.3%
2025-12-10 17:50:48,102 - INFO -   Best cost: £4542.72, Best fulfillment: 100.0%
2025-12-10 17:50:48,102 - INFO - Iteration 2/5
2025-12-10 17:50:56,956 - INFO -   Valid solutions: 500/500
2025-12-10 17:50:56,957 - INFO -   Avg cost: £4876.22, Avg fulfillment: 98.3%
2025-12-10 17:50:56,957 - INFO -   Best cost: £4542.72, Best fulfillment: 100.0%
2025-12-10 17:50:56,957 - INFO - Iteration 3/5
2025-12-10 17:51:05,469 - INFO -   Valid solutions: 500/500
2025-12-10 17:51:05,469 - INFO -   Avg cost: £4823.64, Avg fulfillment: 98.3%
2025-12-10 17:51:05,469 - INFO -   Best cost: £4542.72, Best fulfillment: 100.0%
2025-12-10 17:51:05,469 - INFO - Iteration 4/5
2025-12-10 17:51:14,155 - INFO -   Valid solutions: 500/500
2025-12-10 17:51:14,155 - INFO -   Avg cost: £4834.16, Avg fulfillment: 98.3%
2025-12-10 17:51:14,155 - INFO -   Best cost: £4541.61, Best fulfillment: 100.0%
2025-12-10 17:51:14,155 - INFO - Iteration 5/5
2025-12-10 17:51:22,638 - INFO -   Valid solutions: 500/500
2025-12-10 17:51:22,638 - INFO -   Avg cost: £4838.18, Avg fulfillment: 98.3%
2025-12-10 17:51:22,638 - INFO -   Best cost: £4541.61, Best fulfillment: 100.0%
2025-12-10 17:51:22,638 - INFO - ACO optimization completed

=== FINAL SOLUTION ===
Total routes: 10
Total cost: £4541.61
Fulfillment rate: 100.0%
Validation: PASS
"""

from robin_logistics import LogisticsEnvironment
import numpy as np
from collections import defaultdict
import heapq
import logging
import time
import random
import math
from typing import Dict, List, Tuple, Set


class AntColonyOptimizer:
    """Ant Colony Optimization for MWVRP with road network validation"""
    
    def __init__(self, env, num_ants, num_iterations, decay, alpha, beta):
        self.env = env
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # ACO parameters
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay  # Pheromone decay rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        
        # Problem data
        self.order_ids = env.get_all_order_ids()
        self.available_vehicles = env.get_available_vehicles()
        self.warehouses = self._get_warehouse_info()
        self.orders = self._get_order_info()
        
        # Vehicle and SKU specifications
        self.vehicle_specs = {
            'LightVan': {'weight': 800, 'volume': 3, 'max_distance': 99.5, 'cost_per_km': 1.0, 'fixed_cost': 300.0},
            'MediumTruck': {'weight': 1600, 'volume': 6, 'max_distance': 149.5, 'cost_per_km': 1.25, 'fixed_cost': 625.0},
            'HeavyTruck': {'weight': 5000, 'volume': 20, 'max_distance': 199.5, 'cost_per_km': 1.5, 'fixed_cost': 1200.0}
        }
        
        self.sku_specs = {
            'Light_Item': {'weight': 5, 'volume': 0.02},
            'Medium_Item': {'weight': 15, 'volume': 0.06},
            'Heavy_Item': {'weight': 30, 'volume': 0.12}
        }
        
        # Pheromone matrices
        self.order_pheromones = {}  # pheromones for order selection
        self.sequence_pheromones = {}  # pheromones for order sequencing
        
        # Distance cache
        self.distance_cache = {}
        self.path_cache = {}  # Cache for actual paths
        
        # Best solution tracking
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_fulfillment = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ACO')
        
        # Initialize pheromones
        self._initialize_pheromones()
    
    def _get_warehouse_info(self):
        """Get warehouse information"""
        warehouses = {}

        # Get all warehouses from environment
        for wh_id in self.env.warehouses:
            warehouse = self.env.warehouses[wh_id]
            warehouses[wh_id] = {
                'node': warehouse.location.id,
                'vehicles': [],
                'inventory': warehouse.inventory.copy()
            }

        # Assign vehicles to warehouses
        for vehicle_id in self.available_vehicles:
            home_node = self.env.get_vehicle_home_warehouse(vehicle_id)
            for wh_id, wh_info in warehouses.items():
                if wh_info['node'] == home_node:
                    warehouses[wh_id]['vehicles'].append(vehicle_id)
                    break

        return warehouses
    
    def _get_order_info(self):
        """Get order information"""
        orders = {}
        for order_id in self.order_ids:
            requirements = self.env.get_order_requirements(order_id)
            orders[order_id] = {
                'node': self.env.get_order_location(order_id),
                'requirements': requirements,
                'remaining': requirements.copy(),
                'fulfilled': {sku: 0 for sku in requirements}
            }
        return orders
    
    def _initialize_pheromones(self):
        """Initialize pheromone matrices"""
        # Initialize order selection pheromones
        for wh_id in self.warehouses:
            self.order_pheromones[wh_id] = {}
            for order_id in self.order_ids:
                self.order_pheromones[wh_id][order_id] = 1.0
        
        # Initialize sequence pheromones
        for order_id1 in self.order_ids:
            self.sequence_pheromones[order_id1] = {}
            for order_id2 in self.order_ids:
                if order_id1 != order_id2:
                    self.sequence_pheromones[order_id1][order_id2] = 1.0
    
    def _get_distance(self, from_node: int, to_node: int) -> float:
        """Get cached distance between nodes"""
        cache_key = (from_node, to_node)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Simple Dijkstra
        distances = {from_node: 0}
        heap = [(0, from_node)]
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node == to_node:
                self.distance_cache[cache_key] = current_dist
                return current_dist
            
            for neighbor in self.adjacency_list.get(current_node, []):
                edge_distance = self.env.get_distance(current_node, neighbor)
                if edge_distance is not None:
                    new_dist = current_dist + edge_distance
                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
        
        self.distance_cache[cache_key] = float('inf')
        return float('inf')
    
    def _get_path(self, from_node: int, to_node: int) -> List[int]:
        """Get actual path between nodes with caching"""
        cache_key = (from_node, to_node)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Dijkstra with path reconstruction
        distances = {from_node: 0}
        predecessors = {from_node: None}
        heap = [(0, from_node)]
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node == to_node:
                # Reconstruct path
                path = []
                node = to_node
                while node is not None:
                    path.append(node)
                    node = predecessors[node]
                path.reverse()
                self.path_cache[cache_key] = path
                return path
            
            for neighbor in self.adjacency_list.get(current_node, []):
                edge_distance = self.env.get_distance(current_node, neighbor)
                if edge_distance is not None:
                    new_dist = current_dist + edge_distance
                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current_node
                        heapq.heappush(heap, (new_dist, neighbor))
        
        self.path_cache[cache_key] = None
        return None
    
    def _get_vehicle_type(self, vehicle_id: str) -> str:
        """Extract vehicle type from ID"""
        if 'LightVan' in vehicle_id:
            return 'LightVan'
        elif 'MediumTruck' in vehicle_id:
            return 'MediumTruck'
        elif 'HeavyTruck' in vehicle_id:
            return 'HeavyTruck'
        return 'LightVan'
    
    def _calculate_order_attractiveness(self, wh_id: str, order_id: str, current_load: Dict,
                                      current_distance: float, vehicle_specs: Dict) -> float:
        """Calculate attractiveness of an order for an ant"""
        order_info = self.orders[order_id]
        warehouse_node = self.warehouses[wh_id]['node']
        order_node = order_info['node']

        # Distance component
        distance = self._get_distance(warehouse_node, order_node)
        if distance == float('inf'):
            return 0

        # Check if adding this distance would exceed vehicle max distance
        if current_distance + distance * 2 > vehicle_specs['max_distance']:
            return 0

        # Fulfillment potential component
        warehouse_inventory = self.warehouses[wh_id]['inventory']
        requirements = order_info['remaining']

        fulfillable_items = 0
        total_weight = 0
        total_volume = 0

        for sku_id, required_qty in requirements.items():
            available = warehouse_inventory.get(sku_id, 0)
            if available > 0:
                sku_weight = self.sku_specs.get(sku_id, {}).get('weight', 5)
                sku_volume = self.sku_specs.get(sku_id, {}).get('volume', 0.02)

                # Check capacity constraints
                current_weight = sum(qty * self.sku_specs.get(sku, {}).get('weight', 5)
                                    for sku, qty in current_load.items())
                current_volume = sum(qty * self.sku_specs.get(sku, {}).get('volume', 0.02)
                                    for sku, qty in current_load.items())

                remaining_weight = vehicle_specs['weight'] - current_weight
                remaining_volume = vehicle_specs['volume'] - current_volume

                max_by_weight = remaining_weight // sku_weight if sku_weight > 0 else float('inf')
                max_by_volume = remaining_volume // sku_volume if sku_volume > 0 else float('inf')

                can_fulfill = min(available, required_qty, max_by_weight, max_by_volume)

                if can_fulfill > 0:
                    fulfillable_items += can_fulfill
                    total_weight += can_fulfill * sku_weight
                    total_volume += can_fulfill * sku_volume

        # Calculate heuristic value
        if fulfillable_items == 0:
            return 0

        # Distance efficiency (closer is better)
        distance_efficiency = 1.0 / (1.0 + distance)

        # Load efficiency (better utilization is better)
        weight_util = total_weight / vehicle_specs['weight'] if vehicle_specs['weight'] > 0 else 0
        volume_util = total_volume / vehicle_specs['volume'] if vehicle_specs['volume'] > 0 else 0
        load_efficiency = max(weight_util, volume_util)

        # Combined heuristic
        heuristic = distance_efficiency * (1 + load_efficiency) * fulfillable_items

        return heuristic
    
    def _select_next_order(self, wh_id: str, available_orders: List[str], current_load: Dict,
                          current_distance: float, vehicle_specs: Dict, last_order: str = None) -> str:
        """Select next order using probabilistic decision based on pheromones and heuristics"""
        if not available_orders:
            return None
        
        probabilities = []
        
        for order_id in available_orders:
            # Get pheromone level
            if last_order:
                pheromone = self.sequence_pheromones.get(last_order, {}).get(order_id, 1.0)
            else:
                pheromone = self.order_pheromones.get(wh_id, {}).get(order_id, 1.0)
            
            # Get heuristic value
            heuristic = self._calculate_order_attractiveness(wh_id, order_id, current_load, 
                                                         current_distance, vehicle_specs)
            
            if heuristic > 0:
                # ACO probability formula
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append((order_id, prob))
        
        if not probabilities:
            return random.choice(available_orders)
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in probabilities)
        if total_prob == 0:
            return random.choice(available_orders)
        
        probabilities = [(order_id, prob/total_prob) for order_id, prob in probabilities]
        
        # Roulette wheel selection
        r = random.random()
        cumulative = 0
        for order_id, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return order_id
        
        return probabilities[-1][0]  # Fallback

    def _construct_ant_solution(self, ant_id: int) -> Dict:
        """Construct a complete solution for an ant with proper multi-order routing"""
        solution = {"routes": []}
        remaining_orders = self.order_ids.copy()
        global_inventory = {wh_id: wh_info['inventory'].copy() for wh_id, wh_info in self.warehouses.items()}

        # Reset order fulfillment tracking for this ant
        for order_id in self.order_ids:
            self.orders[order_id]['fulfilled'] = {sku: 0 for sku in self.orders[order_id]['requirements']}
            self.orders[order_id]['remaining'] = self.orders[order_id]['requirements'].copy()

        # Create routes for each warehouse
        for wh_id, wh_info in self.warehouses.items():
            vehicles = wh_info['vehicles'].copy()
            warehouse_node = wh_info['node']

            # Process each vehicle
            for vehicle_id in vehicles:
                if not remaining_orders:
                    break

                vehicle_type = self._get_vehicle_type(vehicle_id)
                vehicle_specs = self.vehicle_specs[vehicle_type]

                # Initialize route state
                route_orders = []  # Orders in this route
                current_node = warehouse_node
                current_distance = 0
                current_load = {}
                total_load_weight = 0
                total_load_volume = 0

                # Build route with multiple deliveries
                while remaining_orders and current_distance < vehicle_specs['max_distance'] * 0.9:
                    candidates = []
                    for order_id in remaining_orders:
                        order_info = self.orders[order_id]
                        order_node = order_info['node']

                        # 1. Check inventory: must have full requirement
                        has_inventory = True
                        for sku_id, req_qty in order_info['requirements'].items():
                            if global_inventory[wh_id].get(sku_id, 0) < req_qty:
                                has_inventory = False
                                break
                        if not has_inventory:
                            continue

                        # 2. Check capacity: current load + new order
                        order_weight = 0
                        order_volume = 0
                        for sku_id, req_qty in order_info['requirements'].items():
                            sku_weight = self.sku_specs.get(sku_id, {}).get('weight', 5)
                            sku_volume = self.sku_specs.get(sku_id, {}).get('volume', 0.02)
                            order_weight += req_qty * sku_weight
                            order_volume += req_qty * sku_volume

                        if total_load_weight + order_weight > vehicle_specs['weight'] or \
                           total_load_volume + order_volume > vehicle_specs['volume']:
                            continue

                        # 3. Check distance: current path + new order + return to warehouse
                        dist_to_order = self._get_distance(current_node, order_node)
                        if dist_to_order == float('inf'):
                            continue

                        # Estimated return distance (minimal path)
                        dist_back = self._get_distance(order_node, warehouse_node)
                        if dist_back == float('inf'):
                            continue

                        if current_distance + dist_to_order + dist_back > vehicle_specs['max_distance']:
                            continue

                        candidates.append(order_id)

                    if not candidates:
                        break

                    # Select next order using ACO
                    next_order = self._select_next_order(
                        wh_id, candidates, current_load, current_distance,
                        vehicle_specs, last_order=route_orders[-1] if route_orders else None
                    )

                    if not next_order:
                        break

                    # Add order to route
                    order_info = self.orders[next_order]
                    order_node = order_info['node']
                    dist_to_order = self._get_distance(current_node, order_node)

                    # Update route state
                    current_distance += dist_to_order
                    current_node = order_node
                    route_orders.append(next_order)

                    # Update load
                    for sku_id, req_qty in order_info['requirements'].items():
                        current_load[sku_id] = current_load.get(sku_id, 0) + req_qty
                        total_load_weight += req_qty * self.sku_specs.get(sku_id, {}).get('weight', 5)
                        total_load_volume += req_qty * self.sku_specs.get(sku_id, {}).get('volume', 0.02)

                    # Remove from remaining orders
                    remaining_orders.remove(next_order)

                # Build the actual route if we have orders
                if route_orders:
                    steps = []
                    # Start at warehouse (will add pickups later)
                    steps.append({
                        "node_id": warehouse_node,
                        "pickups": [],
                        "deliveries": [],
                        "unloads": []
                    })

                    # Current position
                    current_node = warehouse_node

                    # Add path to each order
                    for order_id in route_orders:
                        order_info = self.orders[order_id]
                        order_node = order_info['node']

                        # Get path from current node to order node
                        path = self._get_path(current_node, order_node)
                        if not path:
                            continue

                        # Add intermediate nodes
                        for node in path[1:]:
                            steps.append({
                                "node_id": node,
                                "pickups": [],
                                "deliveries": [],
                                "unloads": []
                            })

                        # Add delivery step
                        deliveries = []
                        for sku_id, req_qty in order_info['requirements'].items():
                            deliveries.append({
                                "order_id": order_id,
                                "sku_id": sku_id,
                                "quantity": req_qty
                            })
                            # Update fulfillment tracking
                            self.orders[order_id]['fulfilled'][sku_id] += req_qty
                            self.orders[order_id]['remaining'][sku_id] -= req_qty

                        steps.append({
                            "node_id": order_node,
                            "pickups": [],
                            "deliveries": deliveries,
                            "unloads": []
                        })

                        current_node = order_node

                    # Return to warehouse
                    path_back = self._get_path(current_node, warehouse_node)
                    if path_back:
                        for node in path_back[1:]:
                            steps.append({
                                "node_id": node,
                                "pickups": [],
                                "deliveries": [],
                                "unloads": []
                            })

                    # Set pickups at warehouse (for all orders in route)
                    pickups = []
                    for sku_id in current_load:
                        pickups.append({
                            "warehouse_id": wh_id,
                            "sku_id": sku_id,
                            "quantity": current_load[sku_id]
                        })
                    steps[0]["pickups"] = pickups

                    # Update global inventory
                    for order_id in route_orders:
                        for sku_id, req_qty in self.orders[order_id]['requirements'].items():
                            global_inventory[wh_id][sku_id] -= req_qty

                    # Add route to solution
                    solution["routes"].append({
                        "vehicle_id": vehicle_id,
                        "steps": steps
                    })

        return solution
    
    def _get_fulfillable_items(self, vehicle_id: str, requirements: Dict, inventory: Dict, 
                             current_load: Dict, vehicle_specs: Dict) -> Dict:
        """Calculate what items can be fulfilled"""
        fulfillable = {}
        
        for sku_id, required_qty in requirements.items():
            available = inventory.get(sku_id, 0)
            if available <= 0:
                continue
            
            sku_weight = self.sku_specs.get(sku_id, {}).get('weight', 5)
            sku_volume = self.sku_specs.get(sku_id, {}).get('volume', 0.02)
            
            current_weight = sum(qty * self.sku_specs.get(sku, {}).get('weight', 5) 
                               for sku, qty in current_load.items())
            current_volume = sum(qty * self.sku_specs.get(sku, {}).get('volume', 0.02) 
                               for sku, qty in current_load.items())
            
            remaining_weight = vehicle_specs['weight'] - current_weight
            remaining_volume = vehicle_specs['volume'] - current_volume
            
            max_by_weight = remaining_weight // sku_weight if sku_weight > 0 else float('inf')
            max_by_volume = remaining_volume // sku_volume if sku_volume > 0 else float('inf')
            
            can_fulfill = min(available, required_qty, max_by_weight, max_by_volume)
            
            if can_fulfill > 0:
                fulfillable[sku_id] = can_fulfill
        
        return fulfillable
    
    def _calculate_solution_cost(self, solution: Dict) -> Tuple[float, float]:
        """Calculate total cost and fulfillment rate for a solution"""
        total_cost = 0
        total_fulfilled = 0
        total_required = 0

        # Calculate total required items
        for order_id in self.order_ids:
            requirements = self.env.get_order_requirements(order_id)
            total_required += sum(requirements.values())

        # Calculate fulfillment from solution
        fulfilled_by_order = {order_id: {sku: 0 for sku in self.env.get_order_requirements(order_id)}
                            for order_id in self.order_ids}

        for route in solution["routes"]:
            vehicle_id = route["vehicle_id"]
            vehicle_type = self._get_vehicle_type(vehicle_id)
            vehicle_specs = self.vehicle_specs[vehicle_type]

            # Calculate route distance
            route_distance = 0
            for i in range(len(route["steps"]) - 1):
                from_node = route["steps"][i]["node_id"]
                to_node = route["steps"][i+1]["node_id"]
                route_distance += self._get_distance(from_node, to_node)

            # Check distance constraint
            if route_distance > vehicle_specs['max_distance']:
                return float('inf'), 0  # Invalid solution

            # Calculate route cost
            route_cost = route_distance * vehicle_specs['cost_per_km'] + vehicle_specs['fixed_cost']
            total_cost += route_cost

            # Calculate fulfillment
            for step in route["steps"]:
                for delivery in step["deliveries"]:
                    order_id = delivery["order_id"]
                    sku_id = delivery["sku_id"]
                    qty = delivery["quantity"]
                    if order_id in fulfilled_by_order and sku_id in fulfilled_by_order[order_id]:
                        fulfilled_by_order[order_id][sku_id] += qty

        # Calculate fulfillment rate (only count fully fulfilled orders)
        fulfilled_orders = 0
        for order_id, fulfilled in fulfilled_by_order.items():
            requirements = self.env.get_order_requirements(order_id)
            if all(fulfilled.get(sku, 0) >= required for sku, required in requirements.items()):
                fulfilled_orders += 1
                total_fulfilled += sum(requirements.values())

        fulfillment_rate = (fulfilled_orders / len(self.order_ids) * 100) if self.order_ids else 0

        return total_cost, fulfillment_rate
    
    def _update_pheromones(self, solutions: List[Dict], costs: List[float], fulfillments: List[float]):
        """Update pheromone levels based on solution quality"""
        # Decay existing pheromones
        for wh_id in self.order_pheromones:
            for order_id in self.order_pheromones[wh_id]:
                self.order_pheromones[wh_id][order_id] *= (1 - self.decay)

        for order_id1 in self.sequence_pheromones:
            for order_id2 in self.sequence_pheromones[order_id1]:
                self.sequence_pheromones[order_id1][order_id2] *= (1 - self.decay)

        # Deposit new pheromones
        for solution, cost, fulfillment in zip(solutions, costs, fulfillments):
            # Quality of solution (higher fulfillment and lower cost is better)
            if cost == float('inf') or fulfillment == 0:
                quality = 0
            else:
                quality = fulfillment / (cost + 1)  # Add 1 to avoid division by zero

            # Update order pheromones per warehouse
            for route in solution["routes"]:
                vehicle_id = route["vehicle_id"]
                warehouse_node = self.env.get_vehicle_home_warehouse(vehicle_id)

                # Find warehouse ID
                wh_id = None
                for w_id, w_info in self.warehouses.items():
                    if w_info['node'] == warehouse_node:
                        wh_id = w_id
                        break

                if wh_id:
                    # Update pheromones for orders served by this warehouse
                    orders_served = set()
                    for step in route["steps"]:
                        for delivery in step["deliveries"]:
                            order_id = delivery["order_id"]
                            orders_served.add(order_id)

                    for order_id in orders_served:
                        if order_id in self.order_pheromones[wh_id]:
                            self.order_pheromones[wh_id][order_id] += quality

            # Update sequence pheromones based on order sequences within routes
            for route in solution["routes"]:
                order_sequence = []
                for step in route["steps"]:
                    for delivery in step["deliveries"]:
                        order_id = delivery["order_id"]
                        if order_id not in order_sequence:
                            order_sequence.append(order_id)

                for i in range(len(order_sequence) - 1):
                    order1 = order_sequence[i]
                    order2 = order_sequence[i + 1]
                    if order1 in self.sequence_pheromones and order2 in self.sequence_pheromones[order1]:
                        self.sequence_pheromones[order1][order2] += quality
    
    def optimize(self) -> Dict:
        """Main optimization loop with validation"""
        self.logger.info(f"Starting ACO optimization with {self.num_ants} ants and {self.num_iterations} iterations")

        for iteration in range(self.num_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")

            # Construct solutions for all ants
            solutions = []
            costs = []
            fulfillments = []

            for ant in range(self.num_ants):
                solution = self._construct_ant_solution(ant)

                # Validate solution before calculating cost
                is_valid, message = self.env.validate_solution_business_logic(solution)
                if not is_valid:
                    self.logger.debug(f"Ant {ant} solution invalid: {message}")
                    cost, fulfillment = float('inf'), 0
                else:
                    cost, fulfillment = self._calculate_solution_cost(solution)

                solutions.append(solution)
                costs.append(cost)
                fulfillments.append(fulfillment)

                # Update best solution (only if valid and better)
                if cost != float('inf') and (fulfillment > self.best_fulfillment or
                   (fulfillment == self.best_fulfillment and cost < self.best_cost)):
                    self.best_solution = solution
                    self.best_cost = cost
                    self.best_fulfillment = fulfillment

            # Update pheromones only with valid solutions
            valid_solutions = [s for s, c in zip(solutions, costs) if c != float('inf')]
            valid_costs = [c for c in costs if c != float('inf')]
            valid_fulfillments = [f for f, c in zip(fulfillments, costs) if c != float('inf')]

            if valid_solutions:
                self._update_pheromones(valid_solutions, valid_costs, valid_fulfillments)

            # Log progress
            valid_costs_filtered = [c for c in costs if c != float('inf')]
            valid_fulfillments_filtered = [f for f, c in zip(fulfillments, costs) if c != float('inf')]

            if valid_costs_filtered:
                avg_cost = np.mean(valid_costs_filtered)
                avg_fulfillment = np.mean(valid_fulfillments_filtered)
                self.logger.info(f"  Valid solutions: {len(valid_costs_filtered)}/{self.num_ants}")
                self.logger.info(f"  Avg cost: £{avg_cost:.2f}, Avg fulfillment: {avg_fulfillment:.1f}%")
            else:
                self.logger.warning("  No valid solutions found in this iteration")

            self.logger.info(f"  Best cost: £{self.best_cost:.2f}, Best fulfillment: {self.best_fulfillment:.1f}%")

        self.logger.info("ACO optimization completed")
        return self.best_solution


def solver(env):
    """Main solver function using ACO with optimized parameters"""
    # Create ACO optimizer with tuned parameters
    aco = AntColonyOptimizer(
        env,
        num_ants=500,   # Reduced for faster convergence
        num_iterations=5,  # Increased for better optimization
        decay=0.2,     # Lower decay for better memory
        alpha=1.5,     # Higher pheromone importance
        beta=3.0       # Higher heuristic importance
    )

    # Run optimization
    solution = aco.optimize()

    # Validate final solution
    is_valid, message = env.validate_solution_business_logic(solution)
    if not is_valid:
        print(f"Warning: Final solution validation failed: {message}")
        # Return empty solution if invalid
        return {"routes": []}

    # Print final statistics
    cost, fulfillment = aco._calculate_solution_cost(solution)
    print(f"\n=== FINAL SOLUTION ===")
    print(f"Total routes: {len(solution['routes'])}")
    print(f"Total cost: £{cost:.2f}")
    print(f"Fulfillment rate: {fulfillment:.1f}%")
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")

    return solution


if __name__ == "__main__":
    env = LogisticsEnvironment()
    solution = solver(env)