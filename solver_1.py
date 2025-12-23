"""
Contestant solver for Robin Logistics Environment.

Implements a hybrid metaheuristic approach with Clarke-Wright initialization,
Ant Colony Optimization, and local search operators.
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time
import random
import copy


def solver(env) -> Dict:
    """Generate a solution using hybrid metaheuristics.

    Args:
        env: LogisticsEnvironment instance

    Returns:
        A complete solution dict with routes and sequential steps.
    """
    # Initialize solution
    solution = {"routes": []}
    
    # Extract data from environment
    order_ids: List[str] = env.get_all_order_ids()
    available_vehicle_ids: List[str] = env.get_available_vehicles()
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get("adjacency_list", {})
    warehouses = env.warehouses
    
    # ACO parameters
    num_ants = 15
    num_iterations = 8
    alpha = 1.0  # Pheromone importance
    beta = 2.0    # Heuristic importance
    decay = 0.1    # Pheromone decay
    q0 = 0.9       # Pseudorandom proportional rule
    
    # Pheromone matrix
    pheromone = defaultdict(lambda: 0.1)
    
    # Best solution tracking
    best_solution = None
    best_cost = float('inf')
    best_fulfillment = 0
    
    # Helper functions
    def bfs_path(start, end):
        """Find shortest path between two nodes using BFS"""
        if start == end:
            return [start]
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            if node == end:
                return path
            
            if node not in adjacency_list:
                continue
                
            for neighbor in adjacency_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_path_distance(path):
        """Calculate total distance of a path"""
        if not path or len(path) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(len(path) - 1):
            segment_dist = env.get_distance(path[i], path[i+1])
            if segment_dist is None:
                return float('inf')
            distance += segment_dist
        return distance
    
    def calculate_order_requirements(order_id):
        """Calculate total weight and volume for an order"""
        requirements = env.get_order_requirements(order_id)
        weight = 0
        volume = 0
        
        for sku_id, qty in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            weight += qty * sku_details['weight']
            volume += qty * sku_details['volume']
        
        return weight, volume
    
    def can_fulfill_from_warehouse(warehouse_id, order_id):
        """Check if warehouse can fulfill an order"""
        inventory = env.get_warehouse_inventory(warehouse_id)
        requirements = env.get_order_requirements(order_id)
        
        for sku_id, qty in requirements.items():
            if inventory.get(sku_id, 0) < qty:
                return False
        return True
    
    def clarke_wright_initialization():
        """Clarke-Wright Savings Algorithm for initial solution"""
        print("Running Clarke-Wright initialization...")
        
        # Calculate savings
        savings = []
        for wh_id, wh in warehouses.items():
            wh_node = wh.location.id
            
            for i, order1 in enumerate(order_ids):
                order1_node = env.get_order_location(order1)
                
                for order2 in order_ids[i+1:]:
                    order2_node = env.get_order_location(order2)
                    
                    # Check if warehouse can fulfill both orders
                    if not (can_fulfill_from_warehouse(wh_id, order1) and 
                            can_fulfill_from_warehouse(wh_id, order2)):
                        continue
                    
                    # Calculate savings
                    path_w1 = bfs_path(wh_node, order1_node)
                    path_w2 = bfs_path(wh_node, order2_node)
                    path_12 = bfs_path(order1_node, order2_node)
                    
                    if not path_w1 or not path_w2 or not path_12:
                        continue
                    
                    d_w1 = get_path_distance(path_w1)
                    d_w2 = get_path_distance(path_w2)
                    d_12 = get_path_distance(path_12)
                    
                    if d_w1 < float('inf') and d_w2 < float('inf') and d_12 < float('inf'):
                        saving = d_w1 + d_w2 - d_12
                        savings.append({
                            'warehouse': wh_id,
                            'order1': order1,
                            'order2': order2,
                            'saving': saving
                        })
        
        # Sort by savings (descending)
        savings.sort(key=lambda x: x['saving'], reverse=True)
        
        # Initialize routes (each order as separate route)
        routes = {}
        for order_id in order_ids:
            # Find a warehouse that can fulfill this order
            for wh_id in warehouses:
                if can_fulfill_from_warehouse(wh_id, order_id):
                    routes[order_id] = {
                        'orders': [order_id],
                        'warehouse': wh_id,
                        'merged': False
                    }
                    break
        
        # Merge routes based on savings
        for saving in savings:
            order1, order2 = saving['order1'], saving['order2']
            wh_id = saving['warehouse']
            
            if order1 in routes and order2 in routes:
                route1 = routes[order1]
                route2 = routes[order2]
                
                if (not route1['merged'] and not route2['merged'] and
                    route1['warehouse'] == wh_id and route2['warehouse'] == wh_id):
                    
                    # Check feasibility of merge
                    if can_merge_routes(route1, route2, wh_id):
                        route1['orders'].extend(route2['orders'])
                        route1['merged'] = True
                        # Update all orders in route2 to point to route1
                        for order in route2['orders']:
                            routes[order] = route1
        
        return routes
    
    def can_merge_routes(route1, route2, warehouse_id):
        """Check if two routes can be merged"""
        all_orders = route1['orders'] + route2['orders']
        
        # Check inventory
        total_demand = defaultdict(int)
        for order_id in all_orders:
            requirements = env.get_order_requirements(order_id)
            for sku_id, qty in requirements.items():
                total_demand[sku_id] += qty
        
        inventory = env.get_warehouse_inventory(warehouse_id)
        for sku_id, qty in total_demand.items():
            if inventory.get(sku_id, 0) < qty:
                return False
        
        # Check capacity (use largest available vehicle)
        available_vehicles = warehouses[warehouse_id].vehicles
        if not available_vehicles:
            return False
        
        # Find largest vehicle
        max_capacity_weight = 0
        max_capacity_volume = 0
        for vehicle in available_vehicles:
            if vehicle.capacity_weight > max_capacity_weight:
                max_capacity_weight = vehicle.capacity_weight
            if vehicle.capacity_volume > max_capacity_volume:
                max_capacity_volume = vehicle.capacity_volume
        
        total_weight = 0
        total_volume = 0
        for order_id in all_orders:
            weight, volume = calculate_order_requirements(order_id)
            total_weight += weight
            total_volume += volume
        
        if total_weight > max_capacity_weight or total_volume > max_capacity_volume:
            return False
        
        return True
    
    def construct_ant_solution():
        """Construct a solution using ACO"""
        nonlocal pheromone, alpha, beta, q0
        
        ant_solution = {"routes": []}
        used_vehicles = set()
        unassigned_orders = set(order_ids)
        
        for wh_id, wh in warehouses.items():
            if not unassigned_orders:
                break
            
            available_vehicles = [
                v.id for v in wh.vehicles if v.id not in used_vehicles
            ]
            
            for vehicle_id in available_vehicles:
                if not unassigned_orders:
                    break
                
                # Build route for this vehicle
                route_orders = []
                current_node = wh.location.id
                current_weight = 0
                current_volume = 0
                
                while unassigned_orders:
                    # Find feasible orders with probabilities
                    feasible_orders = []
                    probabilities = []
                    
                    for order_id in unassigned_orders:
                        if not can_fulfill_from_warehouse(wh_id, order_id):
                            continue
                        
                        order_node = env.get_order_location(order_id)
                        path = bfs_path(current_node, order_node)
                        
                        if not path:
                            continue
                        
                        # Check capacity
                        order_weight, order_volume = calculate_order_requirements(order_id)
                        vehicle = next(v for v in wh.vehicles if v.id == vehicle_id)
                        
                        if (current_weight + order_weight > vehicle.capacity_weight or
                            current_volume + order_volume > vehicle.capacity_volume):
                            continue
                        
                        # Calculate probability using pheromone and heuristic
                        dist = get_path_distance(path)
                        pheromone_level = pheromone[(current_node, order_node)]
                        heuristic = 1.0 / (1.0 + dist)
                        
                        prob = (pheromone_level ** alpha) * (heuristic ** beta)
                        feasible_orders.append(order_id)
                        probabilities.append(prob)
                    
                    if not feasible_orders:
                        break
                    
                    # Normalize probabilities
                    total_prob = sum(probabilities)
                    if total_prob == 0:
                        break
                    
                    probabilities = [p / total_prob for p in probabilities]
                    
                    # Select next order using pseudorandom proportional rule
                    if random.random() < q0:
                        # Exploitation: choose the best order
                        idx = probabilities.index(max(probabilities))
                    else:
                        # Exploration: choose probabilistically
                        r = random.random()
                        cum = 0
                        idx = 0
                        for i, p in enumerate(probabilities):
                            cum += p
                            if r <= cum:
                                idx = i
                                break
                    
                    selected_order = feasible_orders[idx]
                    route_orders.append(selected_order)
                    unassigned_orders.remove(selected_order)
                    
                    # Update current state
                    order_node = env.get_order_location(selected_order)
                    order_weight, order_volume = calculate_order_requirements(selected_order)
                    current_weight += order_weight
                    current_volume += order_volume
                    current_node = order_node
                
                # Build route if orders were assigned
                if route_orders:
                    route = build_route(wh_id, vehicle_id, route_orders)
                    if route:
                        ant_solution['routes'].append(route)
                        used_vehicles.add(vehicle_id)
        
        return ant_solution
    
    def build_route(warehouse_id, vehicle_id, order_ids):
        """Build a complete route with all intermediate nodes"""
        if not order_ids:
            return None
        
        wh_node = warehouses[warehouse_id].location.id
        steps = []
        
        # Start at warehouse
        steps.append({
            'node_id': wh_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
        
        # Pick up all required items
        total_pickups = defaultdict(int)
        for order_id in order_ids:
            requirements = env.get_order_requirements(order_id)
            for sku_id, qty in requirements.items():
                total_pickups[sku_id] += qty
        
        steps.append({
            'node_id': wh_node,
            'pickups': [
                {'warehouse_id': warehouse_id, 'sku_id': sku_id, 'quantity': qty}
                for sku_id, qty in total_pickups.items()
            ],
            'deliveries': [],
            'unloads': []
        })
        
        # Visit each order
        current_node = wh_node
        for order_id in order_ids:
            order_node = env.get_order_location(order_id)
            
            # Get path to order
            path = bfs_path(current_node, order_node)
            if not path:
                return None
            
            # Add intermediate nodes
            for node in path[1:-1]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            
            # Deliver at order node
            requirements = env.get_order_requirements(order_id)
            steps.append({
                'node_id': order_node,
                'pickups': [],
                'deliveries': [
                    {'order_id': order_id, 'sku_id': sku_id, 'quantity': qty}
                    for sku_id, qty in requirements.items()
                ],
                'unloads': []
            })
            
            current_node = order_node
        
        # Return to warehouse
        path = bfs_path(current_node, wh_node)
        if not path:
            return None
        
        # Add intermediate nodes
        for node in path[1:-1]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
        
        # Final warehouse step
        steps.append({
            'node_id': wh_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
        
        return {
            'vehicle_id': vehicle_id,
            'steps': steps
        }
    
    def apply_2opt(route):
        """Apply 2-opt local search to a route"""
        steps = route['steps']
        improved = True
        
        while improved:
            improved = False
            best_improvement = 0
            best_i, best_j = None, None
            
            # Find delivery nodes
            delivery_nodes = [
                i for i, step in enumerate(steps) if step['deliveries']
            ]
            
            for i in range(len(delivery_nodes) - 1):
                for j in range(i + 1, len(delivery_nodes)):
                    idx1, idx2 = delivery_nodes[i], delivery_nodes[j]
                    
                    # Calculate current distance
                    path1 = bfs_path(steps[idx1]['node_id'], steps[idx1+1]['node_id'])
                    path2 = bfs_path(steps[idx2]['node_id'], steps[idx2+1]['node_id'])
                    
                    if not path1 or not path2:
                        continue
                    
                    current_dist = get_path_distance(path1) + get_path_distance(path2)
                    
                    # Calculate new distance after swap
                    path3 = bfs_path(steps[idx1]['node_id'], steps[idx2]['node_id'])
                    path4 = bfs_path(steps[idx1+1]['node_id'], steps[idx2+1]['node_id'])
                    
                    if not path3 or not path4:
                        continue
                    
                    new_dist = get_path_distance(path3) + get_path_distance(path4)
                    
                    improvement = current_dist - new_dist
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = idx1, idx2
            
            if best_improvement > 0.01:
                # Reverse segment
                route['steps'] = (
                    steps[:best_i+1] +
                    steps[best_i+1:best_j+1][::-1] +
                    steps[best_j+1:]
                )
                improved = True
                steps = route['steps']
        
        return route
    
    def local_search(solution):
        """Apply local search to improve solution"""
        # Apply 2-opt to each route
        for route in solution['routes']:
            route = apply_2opt(route)
        
        return solution
    
    def calculate_solution_cost(solution):
        """Calculate solution cost and fulfillment"""
        if not solution or not solution['routes']:
            return float('inf'), 0
        
        try:
            cost = env.calculate_solution_cost(solution)
            fulfillment = env.get_solution_fulfillment_summary(solution)
            fulfillment_rate = fulfillment.get('overall_fulfillment_rate', 0)
            return cost, fulfillment_rate
        except:
            return float('inf'), 0
    
    def adapt_parameters(iteration, max_iterations):
        """Adapt ACO parameters based on progress"""
        nonlocal alpha, beta, decay, num_ants
        
        progress = iteration / max_iterations
        
        if progress < 0.3:  # Exploration phase
            alpha = 0.5
            beta = 5.0
            decay = 0.2
        elif progress < 0.7:  # Balance phase
            alpha = 1.0 + progress
            beta = 4.0 - progress
            decay = 0.3
        else:  # Exploitation phase
            alpha = 2.0
            beta = 2.0
            decay = 0.4
    
    # Main solving process
    start_time = time.time()
    
    # Phase 1: Clarke-Wright initialization
    routes = clarke_wright_initialization()
    
    # Convert to solution format
    used_vehicles = set()
    processed_routes = set()
    
    for order_id, route_info in routes.items():
        if route_info['warehouse'] in processed_routes:
            continue
        
        processed_routes.add(route_info['warehouse'])
        
        # Find available vehicle
        available_vehicles = [
            v.id for v in warehouses[route_info['warehouse']].vehicles
            if v.id not in used_vehicles
        ]
        
        if not available_vehicles:
            continue
        
        vehicle_id = available_vehicles[0]
        used_vehicles.add(vehicle_id)
        
        # Build route
        route = build_route(
            route_info['warehouse'],
            vehicle_id,
            route_info['orders']
        )
        
        if route:
            solution['routes'].append(route)
    
    # Set as initial best solution
    best_solution = copy.deepcopy(solution)
    best_cost, best_fulfillment = calculate_solution_cost(best_solution)
    
    print(f"Initial solution: Cost={best_cost:.2f}, Fulfillment={best_fulfillment:.1f}%")
    
    # Phase 2: ACO with local search
    for iteration in range(num_iterations):
        adapt_parameters(iteration, num_iterations)
        
        for ant in range(num_ants):
            # Construct solution using ACO
            ant_solution = construct_ant_solution()
            
            # Apply local search
            ant_solution = local_search(ant_solution)
            
            # Evaluate solution
            ant_cost, ant_fulfillment = calculate_solution_cost(ant_solution)
            
            # Update best solution
            if ant_cost < best_cost or (ant_cost == best_cost and ant_fulfillment > best_fulfillment):
                best_solution = copy.deepcopy(ant_solution)
                best_cost = ant_cost
                best_fulfillment = ant_fulfillment
            
            # Update pheromones
            for route in ant_solution['routes']:
                for i in range(len(route['steps']) - 1):
                    node1 = route['steps'][i]['node_id']
                    node2 = route['steps'][i+1]['node_id']
                    pheromone[(node1, node2)] += 1.0 / (1.0 + ant_cost)
        
        # Evaporate pheromones
        for key in list(pheromone.keys()):
            pheromone[key] *= (1 - decay)
        
        # Print progress
        if iteration % 2 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iteration}/{num_iterations}: "
                  f"Best Cost={best_cost:.2f}, "
                  f"Fulfillment={best_fulfillment:.1f}%, "
                  f"Time={elapsed:.1f}s")
    
    # Final local search on best solution
    best_solution = local_search(best_solution)
    best_cost, best_fulfillment = calculate_solution_cost(best_solution)
    
    # Validate solution
    is_valid, message = env.validate_solution_business_logic(best_solution)
    
    if not is_valid:
        print(f"Validation failed: {message}")
        return {"routes": []}
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total routes: {len(best_solution['routes'])}")
    print(f"Total cost: £{best_cost:.2f}")
    print(f"Fulfillment rate: {best_fulfillment:.1f}%")
    print(f"Validation: PASS")
    print(f"Total time: {elapsed_time:.2f}s")
    
    return best_solution


if __name__ == '__main__':
    env = LogisticsEnvironment()
    solver(env)