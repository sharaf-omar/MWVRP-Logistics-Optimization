import streamlit as st
import pandas as pd
import folium
import os
import time
import graphviz
from streamlit_folium import st_folium
from robin_logistics import LogisticsEnvironment
from robin_logistics.core.config import (
    SKU_DEFINITIONS, WAREHOUSE_LOCATIONS, VEHICLE_FLEET_SPECS
)

# --- SOLVER IMPORT ---
try:
    from solver_3 import solver as my_custom_solver
except ImportError:
    from robin_logistics.solvers import test_solver as my_custom_solver
    st.toast("⚠️ Solver not found. Using test solver.", icon="⚠️")

# --- CONFIG & CSS ---
st.set_page_config(page_title="MWVRP Optimization | CTRL ALT ELITE", page_icon="🚛", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    h1 {color: #0056b3; font-family: 'Helvetica', sans-serif;}
    h2 {color: #c5a065; font-size: 1.5rem;} 
    h3 {color: #008080;}
    .metric-card {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-top: 3px solid #0056b3;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; padding: 10px;}
    .stTabs [aria-selected="true"] {background-color: #0056b3; color: white;}
    .stDataFrame {border: 1px solid #f0f2f6; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- VISUALIZATION HELPERS ---
def draw_solver_flowcharts():
    """Renders COMPACT flowcharts for all 3 solvers"""
    st.markdown("### 🧠 Algorithmic Approaches")
    t1, t2, t3 = st.tabs(["Solver 1: Hybrid AI", "Solver 2: Pure ACO", "Solver 3: Constructive"])
    
    # Compact Left-to-Right Styling
    graph_attr = {'rankdir': 'LR', 'nodesep': '0.2', 'ranksep': '0.3', 'margin': '0.1'}
    node_style = {'shape': 'box', 'style': 'filled,rounded', 'fontname': 'Helvetica', 'fontsize': '10', 'height': '0.4'}

    with t1:
        g = graphviz.Digraph()
        g.attr(**graph_attr)
        g.attr('node', **node_style)
        g.edge('Start', 'CW Init', label='Determ.')
        g.edge('CW Init', 'ACO Loop', label='Seed')
        g.edge('ACO Loop', 'Local Search', label='Refine')
        g.edge('Local Search', 'Pheromone Upd')
        g.edge('Pheromone Upd', 'ACO Loop')
        g.edge('Pheromone Upd', 'Output', label='Done')
        st.graphviz_chart(g, use_container_width=True)

    with t2:
        g = graphviz.Digraph()
        g.attr(**graph_attr)
        g.attr('node', **node_style)
        g.edge('Start', 'Init Matrix')
        g.edge('Init Matrix', 'Ant Const.', label='500 Ants')
        g.edge('Ant Const.', 'Validation')
        g.edge('Validation', 'Discard', label='Invalid')
        g.edge('Validation', 'Update Phero', label='Valid')
        g.edge('Update Phero', 'Ant Const.', label='Next')
        g.edge('Update Phero', 'Output', label='Max Iter')
        st.graphviz_chart(g, use_container_width=True)

    with t3:
        g = graphviz.Digraph()
        g.attr(**graph_attr)
        g.attr('node', **node_style)
        g.edge('Start', 'Cluster', label='Nearest WH')
        g.edge('Cluster', 'Bin Pack', label='Lrg->Sml')
        g.edge('Bin Pack', 'Sequence', label='Greedy TSP')
        g.edge('Sequence', 'Fallback', label='If > Dist')
        g.edge('Fallback', 'Output')
        st.graphviz_chart(g, use_container_width=True)

def get_coords_map(env):
    """Cache coordinates for O(1) lookup"""
    coords = {w.location.id: [w.location.lat, w.location.lon] for w in env.warehouses.values()}
    coords.update({o.destination.id: [o.destination.lat, o.destination.lon] for o in env.orders.values()})
    if hasattr(env, 'nodes'): coords.update({n.id: [n.lat, n.lon] for n in env.nodes.values()})
    return coords

def render_map(env, solution, route_index=None):
    """Renders map. If route_index is provided, highlights that specific route."""
    if not env.warehouses and not env.orders: return None
    coords = get_coords_map(env)
    vals = list(coords.values())
    center = [sum(x[0] for x in vals)/len(vals), sum(x[1] for x in vals)/len(vals)] if vals else [30.0444, 31.2357]
    
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    # Static Markers
    for wh in env.warehouses.values():
        folium.Marker(coords[wh.location.id], popup=f"<b>WH: {wh.id}</b>", icon=folium.Icon(color='darkblue', icon='industry', prefix='fa')).add_to(m)
    
    full_stats = env.get_solution_fulfillment_summary(solution).get('order_fulfillment_details', {})
    for oid, o in env.orders.items():
        rate = full_stats.get(oid, {}).get('fulfillment_rate', 0)
        c = 'green' if rate >= 99 else 'orange' if rate > 0 else 'red'
        folium.CircleMarker(coords[o.destination.id], radius=6, color=c, fill=True, fill_opacity=0.9, popup=f"<b>{oid}</b>: {rate:.1f}%").add_to(m)

    # Routes
    if solution and solution.get('routes'):
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # If specific route selected, show only that one (or highlight it)
        routes_to_draw = [solution['routes'][route_index]] if route_index is not None else solution['routes']
        
        for i, r in enumerate(routes_to_draw):
            pts = [coords.get(s['node_id']) for s in r.get('steps', []) if s.get('node_id') in coords]
            if len(pts) > 1:
                col = colors[route_index % len(colors)] if route_index is not None else colors[i % len(colors)]
                op = 1.0 if route_index is not None else 0.7
                wei = 5 if route_index is not None else 3
                folium.PolyLine(pts, color=col, weight=wei, opacity=op, tooltip=f"Veh: {r.get('vehicle_id')}").add_to(m)
                
                # Start/End markers for single route view
                if route_index is not None:
                    folium.Marker(pts[0], icon=folium.Icon(color='green', icon='play', prefix='fa'), tooltip="Start").add_to(m)
                    folium.Marker(pts[-1], icon=folium.Icon(color='red', icon='stop', prefix='fa'), tooltip="End").add_to(m)

    return m

def render_infrastructure():
    st.markdown("### 🏗️ Fixed Infrastructure")
    with st.expander("📦 SKU Types", expanded=False):
        st.dataframe(pd.DataFrame([{'SKU ID': x['sku_id'], 'Weight (kg)': x['weight_kg'], 'Volume (m³)': x['volume_m3']} for x in SKU_DEFINITIONS]), hide_index=True, use_container_width=True)
    with st.expander("🚚 Vehicle Fleet Specifications", expanded=False):
        st.dataframe(pd.DataFrame([{'Type': x['type'], 'Cap (kg)': x['capacity_weight_kg'], 'Cap (m³)': x['capacity_volume_m3'], 'Max Dist': x['max_distance_km'], 'Cost/km': f"£{x['cost_per_km']}"} for x in VEHICLE_FLEET_SPECS]), hide_index=True, use_container_width=True)
    with st.expander("🏭 Warehouse Locations", expanded=False):
        st.dataframe(pd.DataFrame([{'ID': x['id'], 'Name': x['name'], 'Lat': f"{x['lat']:.4f}", 'Lon': f"{x['lon']:.4f}"} for x in WAREHOUSE_LOCATIONS]), hide_index=True, use_container_width=True)

# --- MAIN APP ---
def main():
    c1, c2 = st.columns([1, 5])
    with c1: 
        # FIX: Unrolled if/else to prevent Streamlit parsing error
        if os.path.exists("eui_logo.png"):
            st.image("eui_logo.png", width=130) 
        else: 
            st.markdown("## 🏛️")
            
    with c2:
        st.markdown("# Logistics Optimization Benchmark")
        st.markdown("**Introduction to AI (C-AI311) | Team CTRL ALT ELITE**")
        st.caption("Comparative analysis of Hybrid Metaheuristics, Pure ACO, and Constructive Heuristics.")

    if 'env' not in st.session_state:
        st.session_state.update({'env': LogisticsEnvironment(), 'solution': None})
        st.session_state['env'].set_solver(my_custom_solver)

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "⚙️ Simulation", "📊 Analysis", "👥 Team"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1: st.info("The MWVRP involves fulfilling customer orders using a heterogeneous fleet across multiple depots, constrained by inventory, capacity, and connectivity.")
        with c2: st.success("Target: Maximize Fulfillment | Minimize Cost")
        st.divider(); draw_solver_flowcharts(); st.divider(); render_infrastructure()

    with tab2:
        st.markdown("### 🧪 Scenario Config")
        with st.expander("🌍 Geography & Demand", expanded=True):
            c1, c2, c3 = st.columns(3)
            n_ord = c1.number_input("Orders", 5, 200, 25)
            rad = c3.slider("Radius", 5, 75, 20)
        with st.expander("🏭 Supply Chain", expanded=True):
            wh_names = [f"{w['id']}" for w in WAREHOUSE_LOCATIONS]
            sel_wh = st.multiselect("Warehouses", wh_names, default=wh_names[:2])
            wh_cfgs = []
            if sel_wh:
                tabs = st.tabs(sel_wh)
                for i, w in enumerate(sel_wh):
                    with tabs[i]:
                        c1, c2 = st.columns(2)
                        nl = c1.number_input(f"Light", 0, 20, 5, key=f"l{i}")
                        nm = c1.number_input(f"Medium", 0, 20, 5, key=f"m{i}")
                        nh = c1.number_input(f"Heavy", 0, 20, 5, key=f"h{i}")
                        inv = c2.slider("Stock %", 50, 200, 120, key=f"i{i}")
                        wh_cfgs.append({'vehicle_counts':{'LightVan':nl,'MediumTruck':nm,'HeavyTruck':nh},'sku_inventory_percentages':[inv]*3})

        if st.button("🚀 Run Solver", type="primary", use_container_width=True):
            cfg = {'num_orders': n_ord, 'num_warehouses': len(sel_wh), 'distance_control': {'radius_km': rad, 'density_strategy': 'clustered', 'clustering_factor': 0.9}, 'warehouse_configs': wh_cfgs}
            with st.spinner("Optimizing..."):
                try:
                    st.session_state['env'].reset_all_state()
                    st.session_state['env'].generate_scenario_from_config(cfg)
                    t0 = time.time()
                    sol = my_custom_solver(st.session_state['env'])
                    _, _, val = st.session_state['env'].validate_solution_complete(sol)
                    st.session_state['env'].execute_solution(sol)
                    st.session_state.update({'solution': sol, 'validation': val, 'time': time.time()-t0})
                    st.success(f"Done in {st.session_state['time']:.2f}s"); st.rerun()
                except Exception as e: st.error(f"Error: {e}")

    # --- EXPANDED ANALYSIS TAB ---
    with tab3:
        if st.session_state['solution']:
            sol, env, val = st.session_state['solution'], st.session_state['env'], st.session_state.get('validation', {})
            stats = env.get_solution_statistics(sol, val)
            
            # Sub-tabs for deep analysis
            sub_overview, sub_item, sub_route, sub_diag = st.tabs(["📊 Overview", "📦 Item Tracking", "🚛 Route Analysis", "❌ Diagnostics"])

            with sub_overview:
                if val.get('invalid_count', 0) == 0: st.success(f"✅ VALID: {stats.get('total_routes')} routes feasible.")
                else: st.error(f"❌ INVALID: {val.get('invalid_count')} routes failed validation.")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fulfillment", f"{stats.get('average_fulfillment_rate', 0):.1f}%")
                m2.metric("Total Cost", f"£{stats.get('total_cost', 0):,.2f}", help=f"Fixed: £{stats.get('fixed_cost_total',0):,.2f}")
                m3.metric("Total Distance", f"{stats.get('total_distance', 0):.1f} km")
                m4.metric("Vehicles", f"{stats.get('unique_vehicles_used', 0)} / {stats.get('total_vehicles', 0)}")
                
                if map_obj := render_map(env, sol): st.components.v1.html(map_obj._repr_html_(), height=500)

            with sub_item:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🏭 Warehouse Inventory Status**")
                    inv_rows = []
                    for sku in env.skus:
                        r = {'SKU': sku, 'Total': 0}
                        for wid, w in env.warehouses.items():
                            qty = w.inventory.get(sku, 0)
                            r[wid] = qty; r['Total'] += qty
                        inv_rows.append(r)
                    st.dataframe(pd.DataFrame(inv_rows), hide_index=True, use_container_width=True)
                
                with c2:
                    st.markdown("**📦 Order Breakdown**")
                    orders = []
                    full_stats = env.get_solution_fulfillment_summary(sol).get('order_fulfillment_details', {})
                    for oid in env.orders:
                        req = env.orders[oid].requested_items
                        deliv = full_stats.get(oid, {}).get('delivered', {})
                        rate = full_stats.get(oid, {}).get('fulfillment_rate', 0)
                        
                        # Detailed SKU string
                        detail_str = ", ".join([f"{k}: {deliv.get(k,0)}/{v}" for k,v in req.items()])
                        orders.append({"ID": oid, "Rate": f"{rate:.0f}%", "Details": detail_str})
                    st.dataframe(pd.DataFrame(orders), hide_index=True, use_container_width=True)

            with sub_route:
                if sol['routes']:
                    route_opts = [f"Route {i+1}: {r['vehicle_id']}" for i, r in enumerate(sol['routes'])]
                    sel = st.selectbox("Select Route", route_opts)
                    idx = route_opts.index(sel)
                    tgt = sol['routes'][idx]
                    
                    # Compute progression
                    veh = env.get_vehicle_by_id(tgt['vehicle_id'])
                    cap_w = veh.capacity_weight if veh else 1000
                    cap_v = veh.capacity_volume if veh else 10
                    
                    prog = []
                    cw, cv, dist = 0.0, 0.0, 0.0
                    
                    for i, step in enumerate(tgt['steps']):
                        nid = step.get('node_id')
                        # Sum pickups/drops
                        pw = sum(op['quantity'] * env.skus[op['sku_id']].weight for op in step.get('pickups', []))
                        pv = sum(op['quantity'] * env.skus[op['sku_id']].volume for op in step.get('pickups', []))
                        dw = sum(op['quantity'] * env.skus[op['sku_id']].weight for op in step.get('deliveries', []))
                        dv = sum(op['quantity'] * env.skus[op['sku_id']].volume for op in step.get('deliveries', []))
                        
                        cw = cw + pw - dw
                        cv = cv + pv - dv
                        
                        type_icon = "📍"
                        if any(w.location.id == nid for w in env.warehouses.values()): type_icon = "🏭"
                        elif any(o.destination.id == nid for o in env.orders.values()): type_icon = "📦"
                        
                        action = ""
                        if pw > 0: action += f"Load +{pw:.0f}kg "
                        if dw > 0: action += f"Drop -{dw:.0f}kg"
                        
                        prog.append({
                            "Step": i+1, "Node": nid, "Type": type_icon, 
                            "Load (kg)": f"{cw:.1f} / {cap_w}", 
                            "Util %": f"{(cw/cap_w*100):.1f}%", 
                            "Action": action
                        })
                    
                    c1, c2 = st.columns([1, 1])
                    with c1: 
                        st.dataframe(pd.DataFrame(prog), hide_index=True, use_container_width=True)
                    with c2:
                        if map_obj := render_map(env, sol, route_index=idx):
                            st.components.v1.html(map_obj._repr_html_(), height=400)
                else:
                    st.info("No routes to analyze.")

            with sub_diag:
                if val.get('invalid_routes'):
                    st.error(f"Found {len(val['invalid_routes'])} invalid routes.")
                    for err in val['invalid_routes']:
                        with st.expander(f"Route {err['route_index']+1} ({err.get('vehicle_id', 'Unknown')})", expanded=True):
                            st.error(f"Reason: {err['error']}")
                            st.json(err)
                else:
                    st.success("No invalid routes found. System healthy.")

        else: st.info("Run simulation first.")

    with tab4:
        st.markdown("### 👥 Project Team")
        team = [("Omar Wafa", "23-101281"), ("Dania Hassan", "23-101147"), 
                ("Omar Shafiy", "23-201356"), ("Omar Sharaf", "24-101236"), 
                ("Eiad Essam", "23-101108"), ("Youssef Sayed", "23-101227"), 
                ("Habiba Elzahaby", "23-101128")]
        
        cols = st.columns(4)
        for i, (name, uid) in enumerate(team):
            with cols[i % 4]:
                st.markdown(f'<div class="metric-card"><b>{name}</b><br><span style="color:gray;">{uid}</span></div><br>', unsafe_allow_html=True)

if __name__ == "__main__": main()