"""
Cauvery Basin Dairies Operations Research - Erode District Milk Collection Network
Upgraded Mixed-Integer Linear Programming (MILP) Heterogeneous Vehicle Routing Problem (HVRP)

Key Upgrades:
1. True Fuel Minimization Objective: Directly minimizes diesel consumption (Liters)
   accounting for heterogeneous vehicle fuel economy (Heavy: 4.0 km/L, Light: 8.0 km/L).
2. Intra-Class Load Symmetry Breaking: Enforces descending load ordering across identical
   vehicles to prune symmetric permutations and accelerate branch-and-bound convergence:
   - Load(Heavy_1) >= Load(Heavy_2) >= Load(Heavy_3)
   - Load(Light_1) >= Load(Light_2) >= Load(Light_3) >= Load(Light_4)
3. Financial ROI Engine: Compares optimized fleet routing against unoptimized legacy baseline
   (154.2 L/shift, 308.4 L/day), computing daily and annual savings at ₹92.00/L.
4. Dashboard Data Export: Exports active route schedule to `optimized_fleet_routes.csv`.
"""

import sys
from pathlib import Path
import pandas as pd
import pulp

# Set paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from data_ingestion import load_network_data
except ImportError:
    def load_network_data():
        nodes_df = pd.read_csv(BASE_DIR / "erode_collection_nodes.csv")
        nodes_df.columns = nodes_df.columns.str.strip()
        nodes_df["Shift_Demand"] = nodes_df["Daily Demand Base Liters"] / 2.0

        dist_df = pd.read_csv(BASE_DIR / "distance_matrix.csv")
        dist_df.columns = [int(c) for c in dist_df.columns]
        dist_df.index = [int(i) for i in dist_df.index]

        time_df = pd.read_csv(BASE_DIR / "time_matrix.csv")
        time_df.columns = [int(c) for c in time_df.columns]
        time_df.index = [int(i) for i in time_df.index]

        node_ids = nodes_df["Node ID"].astype(int).tolist()
        return {
            "nodes_df": nodes_df,
            "node_ids": node_ids,
            "demand_dict": dict(zip(node_ids, nodes_df["Shift_Demand"])),
            "node_name_dict": dict(zip(node_ids, nodes_df["Node Name"])),
            "distance_dict": dist_df.to_dict(orient="index"),
            "time_dict": time_df.to_dict(orient="index"),
            "total_base_demand": nodes_df["Daily Demand Base Liters"].sum(),
            "total_shift_demand": nodes_df["Shift_Demand"].sum(),
            "fleet_capacity": 23000,
        }

# ==========================================
# CONSTANTS & FLEET CONFIGURATION
# ==========================================
DIESEL_PRICE = 92.0  # INR / Liter

# Baseline Operations: Legacy fixed-sector radial dispatching benchmarks
BASELINE_SHIFT_FUEL = 154.20  # Liters / shift
BASELINE_DAILY_FUEL = BASELINE_SHIFT_FUEL * 2.0  # 308.40 Liters / day

FLEET_CONFIG = {
    "Heavy_1": {"type": "Heavy Truck", "capacity": 5000, "mileage": 4.0},
    "Heavy_2": {"type": "Heavy Truck", "capacity": 5000, "mileage": 4.0},
    "Heavy_3": {"type": "Heavy Truck", "capacity": 5000, "mileage": 4.0},
    "Light_1": {"type": "Light Truck", "capacity": 2000, "mileage": 8.0},
    "Light_2": {"type": "Light Truck", "capacity": 2000, "mileage": 8.0},
    "Light_3": {"type": "Light Truck", "capacity": 2000, "mileage": 8.0},
    "Light_4": {"type": "Light Truck", "capacity": 2000, "mileage": 8.0},
}


def build_and_solve_fleet_model():
    # ---------------------------------------------------------
    # 1. LOAD VERIFIED NETWORK DATA
    # ---------------------------------------------------------
    data = load_network_data()
    nodes_df = data["nodes_df"]
    nodes = data["node_ids"]
    collection_nodes = [i for i in nodes if i != 0]
    shift_demand = data["demand_dict"]
    node_names = data["node_name_dict"]
    distance_dict = data["distance_dict"]
    time_dict = data["time_dict"]

    rural_nodes = (
        nodes_df[nodes_df["Infrastructure Type"] == "Rural"]["Node ID"]
        .astype(int)
        .tolist()
    )
    primary_collection_nodes = [i for i in collection_nodes if i not in rural_nodes]

    trucks = list(FLEET_CONFIG.keys())
    heavy_trucks = [k for k, v in FLEET_CONFIG.items() if "Heavy" in v["type"]]
    light_trucks = [k for k, v in FLEET_CONFIG.items() if "Light" in v["type"]]
    capacities = {k: v["capacity"] for k, v in FLEET_CONFIG.items()}
    mileages = {k: v["mileage"] for k, v in FLEET_CONFIG.items()}

    # ---------------------------------------------------------
    # 2. DEFINE MILP MODEL & DECISION VARIABLES
    # ---------------------------------------------------------
    prob = pulp.LpProblem("Cauvery_Basin_Fuel_Minimization_MILP", pulp.LpMinimize)

    # Binary routing variables
    x = pulp.LpVariable.dicts(
        "x",
        ((i, j, k) for i in nodes for j in nodes for k in trucks),
        cat=pulp.LpBinary,
    )

    # Continuous MTZ subtour elimination variables with native bounds
    u = {}
    for k in trucks:
        valid_nodes = primary_collection_nodes if k in heavy_trucks else collection_nodes
        for i in valid_nodes:
            u[i, k] = pulp.LpVariable(
                f"u_{i}_{k}",
                lowBound=shift_demand[i],
                upBound=capacities[k],
            )

    # ---------------------------------------------------------
    # 3. OBJECTIVE: MINIMIZE TOTAL FLEET FUEL CONSUMPTION (LITERS)
    # ---------------------------------------------------------
    prob += (
        pulp.lpSum(
            (distance_dict[i][j] / mileages[k]) * x[i, j, k]
            for i in nodes
            for j in nodes
            for k in trucks
            if i != j
        ),
        "Total_Fuel_Burned_Liters",
    )

    # ---------------------------------------------------------
    # 4. CONSTRAINTS
    # ---------------------------------------------------------

    # (A) Anti-looping: No vehicle travels to itself
    for i in nodes:
        for k in trucks:
            prob += x[i, i, k] == 0, f"AntiLoop_{i}_{k}"

    # (B) Visit Exactly Once: Every collection node visited exactly once
    for j in collection_nodes:
        prob += (
            pulp.lpSum(x[i, j, k] for i in nodes for k in trucks if i != j) == 1,
            f"VisitOnce_{j}",
        )

    # (C) Infrastructure Wall: Heavy trucks restricted from Rural nodes
    for j in rural_nodes:
        for k in heavy_trucks:
            for i in nodes:
                prob += x[i, j, k] == 0, f"InfraWall_In_{i}_{j}_{k}"
                prob += x[j, i, k] == 0, f"InfraWall_Out_{j}_{i}_{k}"

    # (D) Depot Flow: Every truck departs depot 0 at most once
    for k in trucks:
        prob += (
            pulp.lpSum(x[0, j, k] for j in collection_nodes) <= 1,
            f"DepotStart_{k}",
        )

    # (E) Flow Conservation: Inflow equals outflow at all vertices
    for j in nodes:
        for k in trucks:
            prob += (
                pulp.lpSum(x[i, j, k] for i in nodes if i != j)
                == pulp.lpSum(x[j, i, k] for i in nodes if i != j),
                f"FlowConservation_{j}_{k}",
            )

    # (F) Vehicle Capacity Limits
    truck_load = {
        k: pulp.lpSum(
            shift_demand[j] * pulp.lpSum(x[i, j, k] for i in nodes if i != j)
            for j in collection_nodes
        )
        for k in trucks
    }

    for k in trucks:
        prob += truck_load[k] <= capacities[k], f"CapacityLimit_{k}"

    # (G) Intra-Class Load Symmetry Breaking
    prob += truck_load["Heavy_1"] >= truck_load["Heavy_2"], "Symmetry_H1_H2"
    prob += truck_load["Heavy_2"] >= truck_load["Heavy_3"], "Symmetry_H2_H3"
    prob += truck_load["Light_1"] >= truck_load["Light_2"], "Symmetry_L1_L2"
    prob += truck_load["Light_2"] >= truck_load["Light_3"], "Symmetry_L2_L3"
    prob += truck_load["Light_3"] >= truck_load["Light_4"], "Symmetry_L3_L4"

    # (H) MTZ Subtour Elimination
    for k in trucks:
        valid_nodes = primary_collection_nodes if k in heavy_trucks else collection_nodes
        for i in valid_nodes:
            for j in valid_nodes:
                if i != j:
                    prob += (
                        u[i, k] - u[j, k] + capacities[k] * x[i, j, k]
                        <= capacities[k] - shift_demand[j],
                        f"MTZ_Subtour_{i}_{j}_{k}",
                    )

    # ---------------------------------------------------------
    # 5. SOLVE MILP MODEL (CBC SOLVER)
    # ---------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=90, gapRel=0.01)
    status_code = prob.solve(solver)
    status_label = pulp.LpStatus[status_code]

    # ---------------------------------------------------------
    # 6. ROUTE RECONSTRUCTION & METRICS CALCULATION
    # ---------------------------------------------------------
    active_routes = []
    total_opt_distance = 0.0
    total_opt_load = 0.0
    total_opt_fuel = 0.0

    for k in trucks:
        active_arcs = {
            i: j
            for i in nodes
            for j in nodes
            if i != j and pulp.value(x[i, j, k]) is not None and pulp.value(x[i, j, k]) > 0.5
        }

        if 0 in active_arcs:
            route_nodes = [0]
            curr = active_arcs[0]
            while curr != 0:
                route_nodes.append(curr)
                curr = active_arcs.get(curr, 0)
            route_nodes.append(0)

            route_dist = sum(
                distance_dict[route_nodes[idx]][route_nodes[idx + 1]]
                for idx in range(len(route_nodes) - 1)
            )
            route_time = sum(
                time_dict[route_nodes[idx]][route_nodes[idx + 1]]
                for idx in range(len(route_nodes) - 1)
            )
            route_load = sum(shift_demand[n] for n in route_nodes if n != 0)
            route_fuel = route_dist / mileages[k]
            route_cost = route_fuel * DIESEL_PRICE

            total_opt_distance += route_dist
            total_opt_load += route_load
            total_opt_fuel += route_fuel

            path_readable = " -> ".join(
                f"{node_names[nid]} ({nid})" if nid != 0 else "Depot (0)"
                for nid in route_nodes
            )

            active_routes.append(
                {
                    "Truck ID": k,
                    "Vehicle Class": FLEET_CONFIG[k]["type"],
                    "Route Sequence": path_readable,
                    "Distance km": round(route_dist, 2),
                    "Time min": round(route_time, 1),
                    "Load L": round(route_load, 1),
                    "Utilization %": round((route_load / capacities[k]) * 100.0, 1),
                    "Fuel Burned L": round(route_fuel, 2),
                    "Fuel Cost INR": round(route_cost, 2),
                }
            )

    # ---------------------------------------------------------
    # 7. FINANCIAL ROI & COMPARATIVE OPERATIONAL ENGINE
    # ---------------------------------------------------------
    opt_shift_fuel = total_opt_fuel
    opt_daily_fuel = opt_shift_fuel * 2.0  # 2 shifts/day

    opt_shift_cost = opt_shift_fuel * DIESEL_PRICE
    opt_daily_cost = opt_daily_fuel * DIESEL_PRICE
    opt_annual_cost = opt_daily_cost * 365.0

    base_shift_cost = BASELINE_SHIFT_FUEL * DIESEL_PRICE
    base_daily_cost = BASELINE_DAILY_FUEL * DIESEL_PRICE
    base_annual_cost = base_daily_cost * 365.0

    daily_fuel_saved = BASELINE_DAILY_FUEL - opt_daily_fuel
    daily_cost_saved = base_daily_cost - opt_daily_cost
    annual_cost_saved = base_annual_cost - opt_annual_cost

    fuel_reduction_pct = (daily_fuel_saved / BASELINE_DAILY_FUEL) * 100.0

    # Fuel Efficiency: Liters of Milk Collected per Liter of Fuel Burned
    base_efficiency = (data["total_shift_demand"]) / BASELINE_SHIFT_FUEL
    opt_efficiency = (total_opt_load) / opt_shift_fuel

    # ---------------------------------------------------------
    # 8. CSV EXPORT FOR DASHBOARD INTEGRATION
    # ---------------------------------------------------------
    routes_export_path = BASE_DIR / "optimized_fleet_routes.csv"
    routes_df = pd.DataFrame(active_routes)
    routes_df.to_csv(routes_export_path, index=False)

    # ---------------------------------------------------------
    # 9. TERMINAL CONFIRMATION & STRUCTURED REPORT
    # ---------------------------------------------------------
    print("=" * 88)
    print("      CAUVERY BASIN DAIRIES (ERODE) - FUEL-OPTIMIZED FLEET ROUTING & ROI ENGINE")
    print("=" * 88)
    print(f"Solver Status               : {status_label.upper()}")
    print(f"Objective Function          : Minimizing Total Fuel Burned (Liters)")
    print(f"Shift Milk Volume Collected : {total_opt_load:,.1f} L (100% Demand Met: {data['total_shift_demand']:,.1f} L)")
    print(f"Active Dispatched Trucks    : {len(active_routes)} / {len(trucks)} (Spare Fleet: {len(trucks) - len(active_routes)} Trucks)")
    print(f"Total Shift Fleet Travel    : {total_opt_distance:.2f} km")
    print("-" * 88)
    print("UPGRADED DISPATCHED ROUTE SCHEDULE:")
    print("-" * 88)

    for r in active_routes:
        print(f"[{r['Truck ID']}] {r['Vehicle Class']} | Load: {r['Load L']:,.1f}L ({r['Utilization %']}%) | Mileage: {FLEET_CONFIG[r['Truck ID']]['mileage']} km/L")
        print(f"  * Sequence    : {r['Route Sequence']}")
        print(
            f"  * Metrics     : Distance = {r['Distance km']:.2f} km | Time = {r['Time min']:.1f} min | "
            f"Fuel = {r['Fuel Burned L']:.2f} L | Cost = INR {r['Fuel Cost INR']:,.2f}"
        )
        print("-" * 88)

    print("FINANCIAL ROI & OPERATIONAL COMPARISON BOX:")
    print("=" * 88)
    print(f"  Operational Metric              | Baseline Operations  | Optimized Routing     | Net Savings / Gain")
    print(f"  --------------------------------|----------------------|-----------------------|-------------------")
    print(f"  Single-Shift Fuel Burned (L)    | {BASELINE_SHIFT_FUEL:>18.2f} L | {opt_shift_fuel:>19.2f} L | -{BASELINE_SHIFT_FUEL - opt_shift_fuel:>15.2f} L ({fuel_reduction_pct:.1f}%)")
    print(f"  Daily Fuel Consumption (2 Shifts) | {BASELINE_DAILY_FUEL:>18.2f} L | {opt_daily_fuel:>19.2f} L | -{daily_fuel_saved:>15.2f} L/day")
    print(f"  Single-Shift Fuel Cost (INR)    | INR {base_shift_cost:>14,.2f} | INR {opt_shift_cost:>15,.2f} | -INR {base_shift_cost - opt_shift_cost:>11,.2f}")
    print(f"  Daily Fuel Expenditure (INR)    | INR {base_daily_cost:>14,.2f} | INR {opt_daily_cost:>15,.2f} | -INR {daily_cost_saved:>11,.2f}/day")
    print(f"  Projected Annual Expenditure    | INR {base_annual_cost:>14,.2f} | INR {opt_annual_cost:>15,.2f} | -INR {annual_cost_saved:>11,.2f}/year")
    print(f"  Fleet Milk Fuel Efficiency      | {base_efficiency:>16.2f} L/L | {opt_efficiency:>17.2f} L/L | +{opt_efficiency - base_efficiency:>13.2f} L/L (+{(opt_efficiency/base_efficiency - 1)*100:.1f}%)")
    print("=" * 88)
    print(f"DASHBOARD EXPORT CONFIRMATION:")
    print(f"  * Exported CSV File : {routes_export_path.name}")
    print(f"  * Export Location   : {routes_export_path.resolve()}")
    print(f"  * Total Rows Stored : {len(routes_df)} active routes")
    print("=" * 88)

    return {
        "status": status_label,
        "opt_shift_fuel": opt_shift_fuel,
        "opt_daily_fuel": opt_daily_fuel,
        "daily_fuel_saved": daily_fuel_saved,
        "daily_cost_saved": daily_cost_saved,
        "annual_cost_saved": annual_cost_saved,
        "routes_export_path": str(routes_export_path),
    }


if __name__ == "__main__":
    build_and_solve_fleet_model()
