"""
Cauvery Basin Dairies Operations Research - Erode District Milk Collection Network
Geospatial Visualization & Executive KPI Dashboard Generator

Generates:
1. `erode_fleet_map.html`: Interactive Folium map with color-coded toggleable route layers,
   depot and collection node markers, and floating executive KPI metrics.
2. `fleet_roi_dashboard.png`: Publication-grade 300-DPI dual-panel comparison chart:
   - Panel A: Daily Fuel Consumption & Diesel Expenditure across Baseline, Distance-MILP, and Fuel-MILP.
   - Panel B: Fleet Capacity Utilization across all 7 trucks highlighting the operational reserve unit.
"""

import re
import time
from pathlib import Path
import requests
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set directories
BASE_DIR = Path(__file__).resolve().parent
NODES_CSV = BASE_DIR / "erode_collection_nodes.csv"
ROUTES_CSV = BASE_DIR / "optimized_fleet_routes.csv"
MAP_HTML_OUT = BASE_DIR / "erode_fleet_map.html"
CHART_PNG_OUT = BASE_DIR / "fleet_roi_dashboard.png"

# Color palette for vehicle routes
ROUTE_PALETTE = {
    "Heavy_1": "#D90429",  # Vivid Crimson Red
    "Heavy_2": "#7209B7",  # Deep Royal Purple
    "Light_1": "#0077B6",  # Ocean Blue
    "Light_2": "#F72585",  # Neon Pink / Magenta
    "Light_3": "#FB8500",  # Vivid Amber Orange
    "Light_4": "#008170",  # Dark Teal Green
}

VEHICLE_CAPACITIES = {
    "Heavy_1": 5000,
    "Heavy_2": 5000,
    "Heavy_3": 5000,
    "Light_1": 2000,
    "Light_2": 2000,
    "Light_3": 2000,
    "Light_4": 2000,
}

MAP_TRUCK_LABELS = {
    "Light_1": "Light_1 (TN-33-L01)",
    "Light_2": "Light_2 (TN-33-L02)",
    "Light_3": "Light_3 (TN-33-L03)",
    "Light_4": "Light_4 (TN-33-L04)",
    "Heavy_1": "Heavy_1 (TN-33-H01)",
    "Heavy_2": "Heavy_2 (TN-33-H02)",
}


def load_data():
    """Loads and standardizes node data and solved route schedules."""
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df.columns = nodes_df.columns.str.strip()
    nodes_df["Shift_Demand"] = nodes_df["Daily Demand Base Liters"] / 2.0
    nodes_dict = nodes_df.set_index("Node ID").to_dict(orient="index")

    routes_df = pd.read_csv(ROUTES_CSV)
    return nodes_df, nodes_dict, routes_df


def generate_interactive_map(nodes_df, nodes_dict, routes_df):
    """Creates a rich, interactive Folium map with toggleable vehicle route layers."""
    depot_info = nodes_dict[0]
    depot_lat, depot_lon = depot_info["Latitude"], depot_info["Longitude"]

    # Initialize map centered on Chithode depot with key-free base tiles
    m = folium.Map(
        location=[depot_lat, depot_lon],
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )

    # 100% Key-Free, Local-Compatible Base Map Layers
    folium.TileLayer("Esri.WorldStreetMap", name="Esri Street Map (Default)").add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Streets",
    ).add_to(m)
    folium.TileLayer("Esri.WorldTopoMap", name="Esri Light Topographic").add_to(m)

    # 1. Base Layer for Collection Nodes
    nodes_group = folium.FeatureGroup(name="Milk Collection Nodes", show=True)

    # Plot Depot Marker (Chithode)
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup=folium.Popup(
            f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <h4 style="margin: 0; color: #D90429;"><b>Central Dairy Depot</b></h4>
                <hr style="margin: 5px 0;">
                <b>Node ID:</b> 0<br>
                <b>Location:</b> Chithode Plant<br>
                <b>Role:</b> Network Dispatch & Processing<br>
                <b>Coordinates:</b> {depot_lat:.3f}° N, {depot_lon:.3f}° E
            </div>
            """,
            max_width=300,
        ),
        tooltip="Node 0: Chithode Central Dairy Plant (Depot)",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(nodes_group)

    # Add a glowing circle around depot
    folium.Circle(
        location=[depot_lat, depot_lon],
        radius=1200,
        color="#D90429",
        fill=True,
        fill_color="#D90429",
        fill_opacity=0.15,
        weight=2,
    ).add_to(nodes_group)

    # Plot Collection Nodes (1 to 20)
    for node_id, row in nodes_dict.items():
        if node_id == 0:
            continue
        lat, lon = row["Latitude"], row["Longitude"]
        name = row["Node Name"].strip()
        infra = row["Infrastructure Type"].strip()
        base_demand = row["Daily Demand Base Liters"]
        shift_demand = row["Shift_Demand"]

        is_rural = infra == "Rural"
        color = "#2A9D8F" if is_rural else "#1D3557"  # Teal for Rural, Navy for Primary
        icon_name = "leaf" if is_rural else "building"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 190px;">
            <h4 style="margin: 0; color: {color};"><b>{name}</b></h4>
            <hr style="margin: 4px 0;">
            <b>Node ID:</b> {node_id}<br>
            <b>Infrastructure:</b> <span style="font-weight: bold; color: {color};">{infra}</span><br>
            <b>Shift Demand:</b> {shift_demand:,.0f} L<br>
            <b>Daily Demand:</b> {base_demand:,.0f} L<br>
            <b>Vehicle Access:</b> {'Light Trucks Only' if is_rural else 'Heavy & Light Trucks'}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Node {node_id}: {name} ({infra}) - {shift_demand:,.0f}L",
        ).add_to(nodes_group)

    nodes_group.add_to(m)

    # 2. Add Vehicle Route Polylines as Individual FeatureGroups
    for _, route in routes_df.iterrows():
        truck_id = route["Truck ID"]
        vclass = route["Vehicle Class"]
        dist_km = route["Distance km"]
        time_min = route["Time min"]
        load_l = route["Load L"]
        util = route["Utilization %"]
        fuel_l = route["Fuel Burned L"]
        cost_inr = route["Fuel Cost INR"]
        color = ROUTE_PALETTE.get(truck_id, "#333333")

        # Parse node sequence: "Depot (0) -> Node (X) -> ... -> Depot (0)"
        node_ids = [int(n) for n in re.findall(r"\((\d+)\)", route["Route Sequence"])]
        
        # Snap route to actual Erode road network via public OSRM API
        route_coords = []
        for s in range(len(node_ids) - 1):
            nid_i, nid_j = node_ids[s], node_ids[s + 1]
            lat_i, lon_i = nodes_dict[nid_i]["Latitude"], nodes_dict[nid_i]["Longitude"]
            lat_j, lon_j = nodes_dict[nid_j]["Latitude"], nodes_dict[nid_j]["Longitude"]

            url = f"http://router.project-osrm.org/route/v1/driving/{lon_i},{lat_i};{lon_j},{lat_j}?overview=full&geometries=geojson"
            try:
                response = requests.get(url, timeout=6)
                if response.status_code == 200:
                    data = response.json()
                    osrm_coords = data["routes"][0]["geometry"]["coordinates"]
                    # Flip [lon, lat] from OSRM to [lat, lon] for Folium
                    flipped = [[c[1], c[0]] for c in osrm_coords]
                    if route_coords and flipped:
                        route_coords.extend(flipped[1:])
                    else:
                        route_coords.extend(flipped)
                else:
                    route_coords.extend([[lat_i, lon_i], [lat_j, lon_j]])
            except Exception:
                route_coords.extend([[lat_i, lon_i], [lat_j, lon_j]])

            time.sleep(0.1)

        # Fallback if empty
        if not route_coords:
            route_coords = [
                [nodes_dict[nid]["Latitude"], nodes_dict[nid]["Longitude"]]
                for nid in node_ids
            ]

        base_truck_name = MAP_TRUCK_LABELS.get(truck_id, f"{truck_id} ({vclass})")

        route_group = folium.FeatureGroup(
            name=f"{base_truck_name} - {load_l:,.0f}L ({util}%)",
            show=True,
        )

        tooltip_text = (
            f"<b>{base_truck_name}</b><br>"
            f"Distance: {dist_km:.1f} km | Time: {time_min:.0f} min<br>"
            f"Load: {load_l:,.0f}L ({util}%) | Fuel: {fuel_l:.1f} L"
        )

        # Draw main route polyline
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=4.5,
            opacity=0.88,
            tooltip=tooltip_text,
            popup=folium.Popup(
                f"""
                <div style="font-family: Arial, sans-serif; min-width: 220px;">
                    <h4 style="margin: 0; color: {color};"><b>{base_truck_name} Active Route</b></h4>
                    <hr style="margin: 4px 0;">
                    <b>Class:</b> {vclass}<br>
                    <b>Distance:</b> {dist_km:.2f} km<br>
                    <b>Duration:</b> {time_min:.1f} mins<br>
                    <b>Shift Load:</b> {load_l:,.1f} L ({util}%)<br>
                    <b>Diesel Burned:</b> {fuel_l:.2f} L<br>
                    <b>Shift Fuel Cost:</b> INR {cost_inr:,.2f}<br>
                    <b>Stops:</b> {len(node_ids)-2} collection centers
                </div>
                """,
                max_width=300,
            ),
        ).add_to(route_group)

        # Add sequence step markers along the path
        for step_idx, nid in enumerate(node_ids):
            if nid != 0 and step_idx < len(node_ids) - 1:
                folium.CircleMarker(
                    location=[nodes_dict[nid]["Latitude"], nodes_dict[nid]["Longitude"]],
                    radius=3,
                    color="#FFFFFF",
                    fill=True,
                    fill_color=color,
                    fill_opacity=1.0,
                    weight=1.5,
                ).add_to(route_group)

        route_group.add_to(m)

    # 3. Add Floating Executive KPI Legend Box
    floating_kpi_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 330px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.18);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #212529;
        border-left: 6px solid #D90429;
        font-size: 13px;
        line-height: 1.5;
    ">
        <h4 style="margin: 0 0 8px 0; color: #1D3557; font-size: 15px; font-weight: 700;">
            Cauvery Basin Dairies - Erode Fleet Routing
        </h4>
        <div style="font-size: 11px; color: #6C757D; margin-bottom: 10px;">
            Optimized MILP Vehicle Dispatch Schedule
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <tr style="border-bottom: 1px solid #E9ECEF;">
                <td style="padding: 4px 0; color: #495057;">Total Shift Demand</td>
                <td style="padding: 4px 0; text-align: right; font-weight: bold; color: #1D3557;">15,450 L (100%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #E9ECEF;">
                <td style="padding: 4px 0; color: #495057;">Active Fleet Deployed</td>
                <td style="padding: 4px 0; text-align: right; font-weight: bold; color: #1D3557;">6 / 7 Trucks</td>
            </tr>
            <tr style="border-bottom: 1px solid #E9ECEF;">
                <td style="padding: 4px 0; color: #495057;">Reserve Contingency</td>
                <td style="padding: 4px 0; text-align: right; font-weight: bold; color: #6C757D;">1 Heavy Truck</td>
            </tr>
            <tr style="border-bottom: 1px solid #E9ECEF;">
                <td style="padding: 4px 0; color: #495057;">Daily Fuel Burned</td>
                <td style="padding: 4px 0; text-align: right; font-weight: bold; color: #2A9D8F;">202.33 L/day</td>
            </tr>
            <tr style="border-bottom: 1px solid #E9ECEF;">
                <td style="padding: 4px 0; color: #495057;">Daily Fuel Saved</td>
                <td style="padding: 4px 0; text-align: right; font-weight: bold; color: #0077B6;">-106.07 L (-34.4%)</td>
            </tr>
            <tr>
                <td style="padding: 5px 0 0 0; color: #495057; font-weight: bold;">Annual Cost Savings</td>
                <td style="padding: 5px 0 0 0; text-align: right; font-weight: bold; color: #D90429; font-size: 13px;">INR 35.62 Lakhs</td>
            </tr>
        </table>
        <hr style="margin: 10px 0 8px 0; border: none; border-top: 1px solid #E9ECEF;">
        <div style="font-size: 11px;">
            <b>Node Infrastructure:</b><br>
            <span style="display: inline-block; width: 10px; height: 10px; background: #D90429; border-radius: 50%; margin-right: 5px;"></span> Central Depot (Chithode)<br>
            <span style="display: inline-block; width: 10px; height: 10px; background: #1D3557; border-radius: 50%; margin-right: 5px;"></span> Primary Node (All Trucks)<br>
            <span style="display: inline-block; width: 10px; height: 10px; background: #2A9D8F; border-radius: 50%; margin-right: 5px;"></span> Rural Node (Light Trucks Only)
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(floating_kpi_html))

    # Add LayerControl to allow toggling individual routes and base maps
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Save to HTML
    m.save(str(MAP_HTML_OUT))
    print(f"[OK] Interactive map saved successfully: {MAP_HTML_OUT}")


def generate_executive_kpi_chart(routes_df):
    """Generates a publication-grade 300 DPI 2-panel chart comparing regimes and fleet utilization."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # -------------------------------------------------------------
    # PANEL A: REGIME COMPARISON (Baseline vs. Distance-MILP vs. Fuel-MILP)
    # -------------------------------------------------------------
    regimes = [
        "Legacy Baseline\n(Sector Radial)",
        "Distance-Minimized\n(Distance MILP)",
        "Fuel-Minimized\n(Upgraded MILP)",
    ]
    daily_fuel = [308.40, 222.66, 202.33]  # Liters / day
    daily_cost = [308.40 * 92.0, 222.66 * 92.0, 202.33 * 92.0]  # INR / day

    x = np.arange(len(regimes))
    bar_width = 0.38

    bars_fuel = ax1.bar(
        x - bar_width / 2,
        daily_fuel,
        width=bar_width,
        label="Daily Fuel Burned (Liters)",
        color=["#6C757D", "#457B9D", "#2A9D8F"],
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.92,
    )

    ax1_cost = ax1.twinx()
    bars_cost = ax1_cost.bar(
        x + bar_width / 2,
        [c / 1000 for c in daily_cost],  # in thousands of INR
        width=bar_width,
        label="Daily Cost (INR '000)",
        color=["#E63946", "#E76F51", "#F4A261"],
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.92,
    )

    # Labels and Titles for Panel A
    ax1.set_title(
        "Panel A: Operational Fuel & Expenditure Across Regimes",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#1D3557",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Daily Fuel Consumption (Liters)", fontsize=11, fontweight="bold", color="#1D3557")
    ax1_cost.set_ylabel("Daily Diesel Expenditure (INR '000)", fontsize=11, fontweight="bold", color="#E63946")
    ax1.set_ylim(0, 360)
    ax1_cost.set_ylim(0, 35)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1_cost.grid(False)

    # Annotations on Fuel bars
    for bar in bars_fuel:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f} L",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1D3557",
        )

    # Annotations on Cost bars
    for bar in bars_cost:
        h = bar.get_height()
        ax1_cost.annotate(
            f"INR {h:.1f}k",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#E63946",
        )

    # Highlighting Net Savings Arrow on Panel A
    ax1.annotate(
        "-34.4% Fuel\n-INR 9,758/day",
        xy=(2, 205),
        xytext=(1.4, 280),
        arrowprops=dict(facecolor="#2A9D8F", shrink=0.08, width=2, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#2A9D8F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5", edgecolor="#2A9D8F", lw=1.2),
    )

    # Combined Legend for Panel A
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_cost.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9.5)

    # -------------------------------------------------------------
    # PANEL B: VEHICLE CAPACITY UTILIZATION (%)
    # -------------------------------------------------------------
    TRUCK_ORDER = [
        "Light_1",
        "Light_2",
        "Light_3",
        "Light_4",
        "Heavy_1",
        "Heavy_2",
        "Heavy_3",
    ]

    TRUCK_DISPLAY_LABELS = {
        "Light_1": "Light_1 (TN-33-L01 | 2,000L)",
        "Light_2": "Light_2 (TN-33-L02 | 2,000L)",
        "Light_3": "Light_3 (TN-33-L03 | 2,000L)",
        "Light_4": "Light_4 (TN-33-L04 | 2,000L)",
        "Heavy_1": "Heavy_1 (TN-33-H01 | 5,000L)",
        "Heavy_2": "Heavy_2 (TN-33-H02 | 5,000L)",
        "Heavy_3": "Heavy_3 (TN-33-H03 | 5,000L)",
    }

    fleet_records = []
    routes_by_id = routes_df.set_index("Truck ID").to_dict(orient="index")

    for tid in TRUCK_ORDER:
        cap = VEHICLE_CAPACITIES[tid]
        if tid in routes_by_id:
            load = routes_by_id[tid]["Load L"]
            util = routes_by_id[tid]["Utilization %"]
            vclass = routes_by_id[tid]["Vehicle Class"]
        else:
            load = 0.0
            util = 0.0
            vclass = "Heavy Truck" if "Heavy" in tid else "Light Truck"
        fleet_records.append(
            {
                "Truck ID": tid,
                "Display Label": TRUCK_DISPLAY_LABELS[tid],
                "Class": vclass,
                "Capacity": cap,
                "Load": load,
                "Utilization": util,
            }
        )

    fleet_df = pd.DataFrame(fleet_records)

    y_pos = np.arange(len(fleet_df))
    truck_labels = [row["Display Label"] for _, row in fleet_df.iterrows()]
    utilizations = fleet_df["Utilization"].values

    bar_colors = []
    for _, row in fleet_df.iterrows():
        if row["Utilization"] == 0:
            bar_colors.append("#ADB5BD")  # Grey for reserve
        elif "Heavy" in row["Class"]:
            bar_colors.append("#1D3557")  # Deep Navy for Heavy
        else:
            bar_colors.append("#2A9D8F")  # Teal Green for Light

    bars_util = ax2.barh(
        y_pos,
        utilizations,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.8,
        height=0.6,
        alpha=0.9,
    )

    ax2.set_title(
        "Panel B: Vehicle Capacity Utilization & Fleet Allocation (%)",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#1D3557",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(truck_labels, fontsize=9.5, fontweight="bold")
    ax2.invert_yaxis()  # Enforce top-to-bottom: Light_1 to Heavy_3
    ax2.set_xlabel("Capacity Utilization (%)", fontsize=11, fontweight="bold", color="#1D3557")
    ax2.set_xlim(0, 130)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Add text labels at bar tips with white background box to prevent line collisions
    for i, (_, row) in enumerate(fleet_df.iterrows()):
        u = row["Utilization"]
        load = row["Load"]
        cap = row["Capacity"]
        if u == 0:
            ax2.text(
                3.5,
                i,
                "0.0% [PARKED RESERVE UNIT - Zero Incurred OPEX]",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#495057",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
        else:
            ax2.text(
                u + 1.5,
                i,
                f"{u:.1f}% ({load:,.0f} / {cap:,} L)",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#1D3557",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )

    # Reference threshold lines
    ax2.axvline(100, color="#D90429", linestyle=":", linewidth=1.5, label="Max Rated Capacity (100%)")
    ax2.axvline(80, color="#E76F51", linestyle="--", linewidth=1.2, alpha=0.7, label="Target Utilization (80%)")
    ax2.legend(loc="lower right", fontsize=9, frameon=True)

    plt.suptitle(
        "CAUVERY BASIN DAIRIES (ERODE) - FLEET ROUTING & CAPACITY OPTIMIZATION",
        fontsize=14,
        fontweight="bold",
        color="#1D3557",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(str(CHART_PNG_OUT), dpi=300)
    plt.close()
    print(f"[OK] Executive KPI dashboard saved successfully: {CHART_PNG_OUT}")


def run_visualization():
    print("=" * 80)
    print("     CAUVERY BASIN DAIRIES FLEET OPTIMIZATION - GEOSPATIAL VISUALIZATION & KPI ENGINE")
    print("=" * 80)
    nodes_df, nodes_dict, routes_df = load_data()
    print(f"Loaded {len(nodes_df)} network nodes and {len(routes_df)} optimized vehicle routes.")
    print("-" * 80)
    print("Generating Interactive Geospatial Route Map...")
    generate_interactive_map(nodes_df, nodes_dict, routes_df)
    print("-" * 80)
    print("Generating Executive KPI Publication Chart...")
    generate_executive_kpi_chart(routes_df)
    print("=" * 80)
    print("ALL VISUALIZATION ARTIFACTS GENERATED CONFIRMATION:")
    print(f"  * Interactive Route Map HTML : {MAP_HTML_OUT.resolve()}")
    print(f"  * Executive KPI Dashboard PNG: {CHART_PNG_OUT.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_visualization()
