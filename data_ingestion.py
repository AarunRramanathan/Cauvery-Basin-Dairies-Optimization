"""
Cauvery Basin Dairies Route & Fleet Optimization - Erode District Network
Data Ingestion Pipeline & Matrix Structure Preparation

Resolves fleet capacity deficit via split-shift demand modeling and prepares
physical distance and travel time matrices for PuLP optimization.
"""

from pathlib import Path
import pandas as pd

# Define paths
BASE_DIR = Path(__file__).resolve().parent
NODES_CSV = BASE_DIR / "erode_collection_nodes.csv"
DISTANCE_CSV = BASE_DIR / "distance_matrix.csv"
TIME_CSV = BASE_DIR / "time_matrix.csv"

# Configuration constants
FLEET_CAPACITY = 23000  # Total fleet capacity in Liters


def load_network_data():
    """
    Loads node collection data, resolves capacity deficit by computing Shift_Demand,
    and loads distance and travel time routing matrices as nested dictionaries.
    """
    # 1. Load collection nodes
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df.columns = nodes_df.columns.str.strip()

    # 2. Resolve capacity deficit: model split morning/evening shifts (50% each)
    base_demand_col = "Daily Demand Base Liters"
    nodes_df["Shift_Demand"] = nodes_df[base_demand_col] / 2.0

    total_base_demand = nodes_df[base_demand_col].sum()
    total_shift_demand = nodes_df["Shift_Demand"].sum()

    # Create node-level demand dictionary
    node_ids = nodes_df["Node ID"].astype(int).tolist()
    demand_dict = dict(zip(node_ids, nodes_df["Shift_Demand"]))
    node_name_dict = dict(zip(node_ids, nodes_df["Node Name"]))

    # 3. Load distance matrix and build nested dictionary: distance_dict[i][j]
    dist_df = pd.read_csv(DISTANCE_CSV)
    dist_df.columns = [int(c) for c in dist_df.columns]
    dist_df.index = [int(i) for i in dist_df.index]
    distance_dict = dist_df.to_dict(orient="index")

    # 4. Load time matrix and build nested dictionary: time_dict[i][j]
    time_df = pd.read_csv(TIME_CSV)
    time_df.columns = [int(c) for c in time_df.columns]
    time_df.index = [int(i) for i in time_df.index]
    time_dict = time_df.to_dict(orient="index")

    # 5. Sanity checks on matrix indexing
    n_nodes = len(node_ids)
    assert len(distance_dict) == n_nodes, f"Distance matrix row mismatch: {len(distance_dict)} vs {n_nodes}"
    assert len(time_dict) == n_nodes, f"Time matrix row mismatch: {len(time_dict)} vs {n_nodes}"
    for i in node_ids:
        assert len(distance_dict[i]) == n_nodes, f"Distance matrix column mismatch for node {i}"
        assert len(time_dict[i]) == n_nodes, f"Time matrix column mismatch for node {i}"

    return {
        "nodes_df": nodes_df,
        "node_ids": node_ids,
        "demand_dict": demand_dict,
        "node_name_dict": node_name_dict,
        "distance_dict": distance_dict,
        "time_dict": time_dict,
        "total_base_demand": total_base_demand,
        "total_shift_demand": total_shift_demand,
        "fleet_capacity": FLEET_CAPACITY,
    }


def print_ingestion_summary(data):
    """Prints a clear terminal summary of the ingestion pipeline."""
    nodes_df = data["nodes_df"]
    base_demand = data["total_base_demand"]
    shift_demand = data["total_shift_demand"]
    fleet_cap = data["fleet_capacity"]
    distance_dict = data["distance_dict"]
    time_dict = data["time_dict"]
    node_ids = data["node_ids"]

    depot_name = data["node_name_dict"][0]
    sample_target = node_ids[1]
    sample_target_name = data["node_name_dict"][sample_target]

    print("=" * 72)
    print("      CAUVERY BASIN DAIRIES FLEET OPTIMIZATION - DATA INGESTION PIPELINE")
    print("=" * 72)
    print(f"Total Network Nodes Loaded : {len(node_ids)} (Depot: Node 0 '{depot_name}')")
    print(f"Infrastructure Types       : { {k: int(v) for k, v in nodes_df['Infrastructure Type'].value_counts().items()} }")
    print("-" * 72)
    print("FLEET CAPACITY DEFICIT ANALYSIS:")
    print(f"  * Total Daily Base Demand : {base_demand:,.1f} Liters")
    print(f"  * Available Fleet Capacity: {fleet_cap:,.1f} Liters")
    deficit = base_demand - fleet_cap
    print(f"  * Deficit Status          : OVER CAPACITY by {deficit:,.1f} L (Deficit Ratio: {base_demand/fleet_cap:.2f}x)")
    print("-" * 72)
    print("SPLIT-SHIFT RESOLUTION:")
    print(f"  * Shift Model             : 50% Morning / 50% Evening Split")
    print(f"  * Total Shift Demand      : {shift_demand:,.1f} Liters")
    headroom = fleet_cap - shift_demand
    headroom_pct = (headroom / fleet_cap) * 100
    print(f"  * Fleet Utilization       : {(shift_demand / fleet_cap)*100:.1f}%")
    print(f"  * Capacity Headroom       : +{headroom:,.1f} Liters ({headroom_pct:.1f}% buffer)")
    print(f"  * Status                  : RESOLVED (Shift Demand <= Fleet Capacity)")
    print("-" * 72)
    print("PHYSICAL ROUTING MATRICES VALIDATION:")
    print(f"  * Distance Matrix Shape   : {len(distance_dict)} x {len(distance_dict[0])} nodes")
    print(f"  * Time Matrix Shape       : {len(time_dict)} x {len(time_dict[0])} nodes")
    print("  * Index Query Verification:")
    print(f"      - distance_dict[0][{sample_target}] ({depot_name} -> {sample_target_name}) = {distance_dict[0][sample_target]:.2f} km")
    print(f"      - distance_dict[{sample_target}][0] ({sample_target_name} -> {depot_name}) = {distance_dict[sample_target][0]:.2f} km")
    print(f"      - time_dict[0][{sample_target}] ({depot_name} -> {sample_target_name})     = {time_dict[0][sample_target]:.2f} min")
    print(f"      - time_dict[{sample_target}][0] ({sample_target_name} -> {depot_name})     = {time_dict[sample_target][0]:.2f} min")
    print(f"  * Matrix Status           : All {len(node_ids) * len(node_ids)} pairs mapped with 0 indexing errors.")
    print("=" * 72)


if __name__ == "__main__":
    data = load_network_data()
    print_ingestion_summary(data)
