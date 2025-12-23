## 👥 Team: CTRL ALT ELITE

**Developed by:**

*   **Omar Wafa** (23-101281)
*   **Dania Hassan** (23-101147)
*   **Omar Shafiy** (23-201356)
*   **Omar Sharaf** (24-101236)
*   **Eiad Essam** (23-101108)
*   **Youssef Sayed** (23-101227)
*   **Habiba Elzahaby** (23-101128)

Here is the comprehensive analysis of **Solver #1**.

### 1. High-Level Approach: "Seeded Swarm Intelligence"
This solver utilizes a **Hybrid Metaheuristic** architecture. Instead of relying on a single technique, it layers three distinct phases to balance speed and solution quality:

1.  **Initialization (Deterministic):** It doesn't start with random routes. It uses a classical heuristic (**Clarke-Wright**) to generate a mathematically sound "First Draft" solution.
2.  **Global Search (Probabilistic):** It uses **Ant Colony Optimization (ACO)**. Multiple artificial "ants" explore the solution space, learning from previous iterations via "pheromones" to find better route combinations that the deterministic approach missed.
3.  **Refinement (Local):** It applies **2-opt Local Search** at the end of every ant's journey to "untangle" any messy routes and ensure local optimality.

**Why this works:** The initialization ensures the solver never performs poorly, while the ACO allows it to break out of local optima to find superior solutions over time.

---

### 2. Key Algorithms Used

#### A. Clarke-Wright Savings Algorithm
*   **Where:** `clarke_wright_initialization()`
*   **Function:** Generates the initial population.
*   **Logic:** It calculates the "savings" obtained by merging two separate deliveries into one route ($S_{ij} = Dist(D,i) + Dist(D,j) - Dist(i,j)$). It greedily merges the pairs with the highest savings first, respecting vehicle capacity.

#### B. Ant Colony Optimization (ACO)
*   **Where:** `construct_ant_solution()`
*   **Function:** The core optimization engine.
*   **Logic:**
    *   **Construction:** Ants build routes node-by-node.
    *   **Decision Rule:** The probability of moving to a specific node depends on **Pheromone intensity** (Past success) and **Heuristic visibility** (Distance).
    *   **Pseudorandom Proportional Rule ($q_0$):** A parameter `q0 = 0.9` determines the strategy: 90% of the time, the ant picks the *best* move (Exploitation); 10% of the time, it picks a random move based on probability (Exploration).

#### C. Breadth-First Search (BFS)
*   **Where:** `bfs_path()`
*   **Function:** Pathfinding (The "GPS").
*   **Logic:** Since the road network is a graph, BFS is used to find the shortest sequence of road nodes to connect a Warehouse to an Order, or an Order to another Order.

#### D. 2-Opt Local Search
*   **Where:** `apply_2opt()`
*   **Function:** Route polishing.
*   **Logic:** It iterates through a route and checks if swapping two edges results in a shorter distance (effectively untangling crossing paths). If $Distance(A \to C \to B \to D) < Distance(A \to B \to C \to D)$, it performs the swap.

---

### 3. Optimization Principles & Techniques

#### 1. Adaptive Parameter Control
*   **Code Reference:** `adapt_parameters()`
*   **Concept:** The solver does not use fixed settings.
    *   **Early Phase:** High `Beta` (Heuristic importance) and low `Alpha` (Pheromone importance) encourage **Exploration** (looking for new paths).
    *   **Late Phase:** High `Alpha` and low `Beta` encourage **Exploitation** (converging on the best path found so far).

#### 2. Stigmergy (Pheromone Memory)
*   **Code Reference:** `pheromone` dictionary.
*   **Concept:** The solver implements indirect communication. When an ant finds a good route, it deposits pheromones. Future ants are more likely to follow that path. This allows the system to "remember" successful complex patterns.

#### 3. Capacity-Constrained Bin Packing
*   **Code Reference:** `can_merge_routes` and `construct_ant_solution`.
*   **Concept:** The solver strictly enforces **Weight** and **Volume** constraints during the construction phase. It tracks the remaining capacity of the vehicle dynamically, ensuring that no invalid solutions are generated that would need to be discarded later.

#### 4. The "Seeding" Principle
*   **Concept:** Pure ACO often starts very slowly because ants wander randomly at first. By seeding the best solution with **Clarke-Wright**, the pheromone trails are initialized on "decent" paths immediately, saving computation time and allowing the ants to focus on refinement rather than basic discovery.


Here is the comprehensive analysis of **Solver #2**.

### 1. High-Level Approach: "Pure Probabilistic Swarm"
Unlike Solver #1 which used a hybrid approach, this solver implements a **Pure Ant Colony Optimization (ACO)** strategy without a deterministic initialization phase (like Clarke-Wright).

1.  **Massive Parallel Exploration:** It relies on a large population of agents (500 ants per iteration) to brute-force the discovery of good routes through probability.
2.  **Strict Validation Feedback:** It integrates the environment's validation logic directly into the optimization loop. Any route an ant builds that is invalid is immediately discarded, ensuring the pheromone update step only learns from *feasible* solutions.
3.  **Multi-Objective Pheromones:** It uses two distinct pheromone matrices—one for **Warehouse-to-Order** selection and another for **Order-to-Order** sequencing—to capture different aspects of the routing decision.

**Why this works:** By throwing computational power (500 ants) at the problem and strictly filtering results, it statistically converges on high-quality solutions, albeit at a higher computational cost than the hybrid approach.

---

### 2. Key Algorithms Used

#### A. Dual-Matrix Ant Colony Optimization
*   **Where:** `_select_next_order()` and `_update_pheromones()`
*   **Function:** Decision making.
*   **Logic:**
    *   **Order Pheromones:** Tracks how good it is for a specific *Warehouse* to serve a specific *Order* (Assignment problem).
    *   **Sequence Pheromones:** Tracks how good it is to visit *Order B* immediately after *Order A* (Routing problem).
    *   **Combined Probability:** The ant combines these two pheromone values with heuristic info (Distance + Load Efficiency) to make its next move.

#### B. Dijkstra's Algorithm
*   **Where:** `_get_distance()` and `_get_path()`
*   **Function:** Pathfinding.
*   **Logic:** Unlike Solver #1 which used simple BFS (Breadth-First Search) for unweighted graphs, this solver implements **Dijkstra**. This suggests it is prepared to handle **weighted graphs** (e.g., if roads had different speed limits or costs), offering more precision in cost calculation than BFS.

#### C. Roulette Wheel Selection
*   **Where:** `_select_next_order()`
*   **Function:** Stochastic choice.
*   **Logic:** Instead of just picking the best option, it assigns a slice of a "roulette wheel" to each candidate order based on its fitness probability. This ensures diversity; even slightly less optimal routes have a small chance of being picked, preventing the colony from getting stuck in local optima too early.

---

### 3. Optimization Principles & Techniques

#### 1. Heavy Caching
*   **Code Reference:** `distance_cache` and `path_cache`.
*   **Concept:** Dijkstra is expensive. This solver memoizes (caches) every calculated path. If an ant needs to go from Node A to Node B, and another ant has already calculated that trip, it retrieves the result instantly in $O(1)$ time. This is critical for scaling to 500 ants.

#### 2. Multi-Factor Heuristic Function
*   **Code Reference:** `_calculate_order_attractiveness()`
*   **Concept:** The heuristic isn't just "Distance." It calculates a composite score:
    *   $Heuristic = DistanceEfficiency \times (1 + LoadEfficiency) \times FulfillableItems$
    *   This encourages ants to prioritize orders that are close **AND** help fill the truck's capacity efficiently **AND** can actually be fulfilled by the current warehouse inventory.

#### 3. Exploration vs. Exploitation Tuning
*   **Code Reference:** `alpha=1.5`, `beta=3.0`.
*   **Concept:**
    *   **High Beta (3.0):** The solver heavily favors the Heuristic (Distance/Load logic). It trusts the "greedy" math more than the pheromone history.
    *   **Moderate Alpha (1.5):** It uses pheromones as a gentle guide rather than a strict rule.
    *   This tuning is aggressive: it forces ants to find "sensible" routes first (short distance), and then refine them with pheromones.

#### 4. Valid-Only Pheromone Updates
*   **Code Reference:** `optimize()` method loop.
*   **Concept:** The solver calculates the fitness of all 500 ants but **only** deposits pheromones for the ants that generated **Valid** solutions. Invalid solutions (e.g., truck overloaded) contribute nothing. This acts as an evolutionary filter, purifying the "collective memory" of the colony to only include feasible strategies.


Here is the comprehensive analysis of **Solver #3** for.

### 1. High-Level Approach: "Deterministic Constructive Heuristic"
This solver represents **Baseline Model**. Unlike Solvers 1 and 2, it does not use AI, randomization, or iterative improvement (Metaheuristics). Instead, it uses a strict, logical pipeline to build a solution from scratch in a single pass.

The strategy is **"Cluster-First, Route-Second"**:
1.  **Clustering (Assignment):** First, it decides which warehouse handles which order based on inventory and proximity.
2.  **Bin Packing (Loading):** It aggressively packs the largest trucks with the largest orders first.
3.  **Routing (Sequencing):** It determines the delivery sequence using a Greedy Nearest Neighbor approach.

**Why this is important** This solver acts as the "Control Group." It is extremely fast (milliseconds vs. seconds) but "dumb.", Hybrid AI (Solver 1) and ACO (Solver 2) must outperform this solver to justify their computational cost.

---

### 2. Key Algorithms Used

#### A. First-Fit Decreasing (Bin Packing)
*   **Where:** Main logic loop (lines 280-330).
*   **Function:** Vehicle Loading.
*   **Logic:**
    1.  **Sort Vehicles:** Largest capacity $\to$ Smallest.
    2.  **Sort Orders:** Heaviest/Largest Volume $\to$ Lightest.
    3.  **Pack:** It takes the largest available vehicle and tries to fit the largest unassigned orders into it until it is full.
*   **Significance:** This is a classic approximation algorithm for the Bin Packing Problem. It minimizes the number of vehicles used by prioritizing "hard-to-fit" large orders.

#### B. Greedy Nearest Neighbor (Assignment)
*   **Where:** `find_best_warehouse_for_order`
*   **Function:** Warehouse selection.
*   **Logic:** For every order, it calculates the distance to all warehouses. It assigns the order to the **closest** warehouse that currently has the required stock. It essentially asks: "Who is the nearest supplier that can help me right now?"

#### C. Greedy Nearest Neighbor (TSP - Routing)
*   **Where:** `build_multi_order_route`
*   **Function:** Determining the stop sequence.
*   **Logic:** Once a truck is loaded with a set of orders:
    1.  Start at Warehouse.
    2.  Find the closest unvisited customer. Drive there.
    3.  Repeat until empty.
    4.  Return home.
*   **Note:** This is fast but "short-sighted." It can sometimes lead to suboptimal paths where the truck has to drive a long way back because it left a distant customer for last.

#### D. Breadth-First Search (BFS)
*   **Where:** `bfs_path`
*   **Function:** Pathfinding.
*   **Logic:** Used to calculate distances and find the specific road nodes to connect two points on the map.

---

### 3. Optimization Principles & Techniques

#### 1. Deterministic Execution
*   **Concept:** Unlike the ACO solvers, if you run this solver 100 times on the same input, you get the exact same result 100 times. This stability makes it perfect for debugging the environment constraints.

#### 2. Virtual Inventory Locking
*   **Code Reference:** `allocated_inventory` dictionary.
*   **Concept:** To prevent race conditions (where two trucks think they can take the last item in stock), the solver "locks" inventory immediately upon assignment. If a route turns out to be invalid later, it "unlocks" (deallocates) the inventory so other vehicles can use it.

#### 3. Subset Fallback (Constraint Relaxation)
*   **Code Reference:** `build_fallback_routes`
*   **Concept:** Sometimes, a truck is packed perfectly by weight, but the resulting route is too long (violates max distance). Instead of failing completely, the solver attempts **Backtracking**: it removes the last order added and tries to route the smaller subset. It keeps shrinking the load until a valid route is found.

#### 4. The "Big Rocks" Principle
*   **Concept:** By sorting vehicles and orders by size (descending), the solver handles the most difficult constraints first. It is much easier to squeeze a small order into a small gap later than it is to fit a huge order at the end of the process.

# The "CTRL ALT ELITE" Dashboard: A Technical Guide for Defense

This document explains the Graphical User Interface (GUI) of your project from scratch.

---

## 1. The Core Concept: What is Streamlit?

the dashboard is built using a Python library called **Streamlit**.

**The Analogy:**
Imagine writing a standard Python script. Usually, you print output to a black console window. Streamlit intercepts that script. When you write `st.write("Hello")`, instead of printing to the console, it prints to a web browser.

**The Execution Flow (Crucial for Defense):**
*   **Traditional App:** You have a "Frontend" (HTML/JS) and a "Backend" (Python) that talk via API.
*   **Streamlit:** The script runs from **Top to Bottom**. Every time you interact with a widget (click a button, drag a slider), **the entire script re-runs from line 1**.

---

## 2. How the UI Structure is Built

### Layout & Styling
We don't use HTML files. We define layout using Python context managers (`with` statements).

*   **`st.set_page_config`**: Sets the tab title ("MWVRP Optimization") and layout mode to "Wide".
*   **CSS Injection**: We use `st.markdown("<style>...</style>")` to inject raw CSS. This allows us to:
    *   Hide the default Streamlit "Hamburger menu".
    *   Force the headers (`h1`, `h2`) to match EUI's blue/gold colors.
*   **Columns**: `c1, c2 = st.columns([1, 5])` splits the screen. `c1` takes 1 part (Logo), `c2` takes 5 parts (Title).

### The Tabs System
We use `tab1, tab2, tab3, tab4 = st.tabs(["Name1", "Name2"...])`.
*   **How it works:** This creates four "Containers".
*   **Code Structure:**
    ```python
    with tab1:
        # Everything indented here appears ONLY in Tab 1
        st.write("Overview")
    with tab2:
        # Everything here appears ONLY in Tab 2
        st.slider(...)
    ```

---

## 3. The "Memory": Session State

**The Problem:** Since Streamlit re-runs the whole script every time you click a button, variables normally get deleted. If you calculate a solution, then click "Analysis", the script re-runs, and the solution variable is lost.

**The Solution:** `st.session_state`.
*   Think of this as the **browser's backpack**.
*   When we run the solver, we save the result into the backpack: `st.session_state['solution'] = sol`.
*   When the script re-runs to show the "Analysis" tab, we check the backpack: `if st.session_state['solution']: show_graphs()`.

---

## 4. Feature Deep Dive: How It Works

### A. The Simulation Tab (Inputs & Logic)

**1. The Inputs:**
We use widgets like `st.slider`, `st.number_input`, and `st.multiselect`.
*   **Reactivity:** When you change the "Radius" slider, the variable `rad` in Python updates immediately.

**2. The "Run Solver" Button (The Brain):**
When clicked, the following chain reaction happens:
1.  **Config Construction:** We take all the variables from the sliders (Orders, Vehicles, Inventory) and pack them into a Python Dictionary (`config = {...}`).
2.  **Environment Reset:** `env.reset_all_state()` wipes the previous simulation.
3.  **Scenario Generation:** `env.generate_scenario_from_config(config)` tells the backend to build a *new* Cairo map based on specific inputs (e.g., "Create 50 orders within 20km").
4.  **Solver Execution:** `sol = my_custom_solver(env)` runs the algorithms (Hybrid/ACO).
5.  **State Saving:** The result is saved to `st.session_state`.
6.  **Rerun:** `st.rerun()` forces the page to refresh so the results appear immediately.

### B. The Map Rendering (Folium)

**How do we draw the map?**
We use a library called **Folium**, which is a Python wrapper for Leaflet.js (a famous JavaScript map library).

**The Process (inside `render_map` function):**
1.  **Coordinate Cache:** The code loops through every Warehouse and Order *once* and saves their lat/lon in a dictionary. This makes drawing 100x faster (Optimization).
2.  **Base Map:** Creates a blank map centered on Cairo.
3.  **Layers:**
    *   **Warehouses:** Adds a `folium.Marker` (Pin icon).
    *   **Orders:** Adds a `folium.CircleMarker`.
        *   *Color Logic:* The code checks the fulfillment %. If `100%` -> Green. If `0%` -> Red.
    *   **Routes:** We iterate through the `steps` in the solution. We collect the list of coordinates `[node1, node2, node3...]` and draw a `folium.PolyLine` connecting them.

### C. The Flowcharts (Graphviz)

**How are they drawn?**
We don't draw them as images. We write code that *describes* the graph using the DOT language.
*   **Nodes:** `g.node('A', 'Label')` creates a box.
*   **Edges:** `g.edge('A', 'B')` draws an arrow from A to B.
*   **Rendering:** The library calculates the best layout automatically so the arrows don't cross messily.

# to run streamlit run ctrl_alt_elite_dashboard.py