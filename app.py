import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os

# ----------------------------------
# 1. Page Configuration & Theme
# ----------------------------------
st.set_page_config(
    page_title="RetailGuard AI - Demand & Inventory Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# 2. Cached Data Loading
# ----------------------------------
@st.cache_data
def load_all_data():
    predictions = pd.read_csv("reports/future_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"])
    
    inventory = pd.read_csv("reports/inventory_recommendations.csv")
    
    stores = None
    if os.path.exists("data/raw/stores.csv"):
        stores = pd.read_csv("data/raw/stores.csv")
        
    feature_importance = None
    if os.path.exists("reports/feature_importance.csv"):
        feature_importance = pd.read_csv("reports/feature_importance.csv")
        
    model_comp = None
    if os.path.exists("reports/model_comparison.csv"):
        model_comp = pd.read_csv("reports/model_comparison.csv")
        
    rec_model_comp = None
    if os.path.exists("reports/recursive_model_comparison.csv"):
        rec_model_comp = pd.read_csv("reports/recursive_model_comparison.csv")
        
    return predictions, inventory, stores, feature_importance, model_comp, rec_model_comp

try:
    predictions, inventory, stores, feature_importance, model_comp, rec_model_comp = load_all_data()
except Exception as e:
    st.error(f"Error loading required datasets: {e}. Please ensure prediction and inventory scripts have run.")
    st.stop()

# ----------------------------------
# 3. Sidebar Filters & Navigation
# ----------------------------------
st.sidebar.title("🛒 RetailGuard AI")
st.sidebar.caption("Intelligent Retail Demand Forecasting & Inventory Control")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Location & Category Filter")
store_list = sorted(inventory["store_nbr"].unique())
selected_store = st.sidebar.selectbox("Select Store Number", store_list, index=0)

# Display Store Details if available
if stores is not None:
    store_meta = stores[stores["store_nbr"] == selected_store]
    if not store_meta.empty:
        meta_row = store_meta.iloc[0]
        st.sidebar.info(
            f"**City:** {meta_row['city']}  \n"
            f"**State:** {meta_row['state']}  \n"
            f"**Store Type:** {meta_row['type']}  \n"
            f"**Cluster:** {meta_row['cluster']}"
        )

family_list = sorted(inventory["family"].unique())
selected_family = st.sidebar.selectbox("Select Product Family", ["All Families"] + family_list)

# Date filter for predictions
min_date = predictions["date"].min().date()
max_date = predictions["date"].max().date()
selected_dates = st.sidebar.date_input(
    "Forecast Window",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter datasets
filtered_inv = inventory[inventory["store_nbr"] == selected_store].copy()
filtered_pred = predictions[predictions["store_nbr"] == selected_store].copy()

# Date filter application
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    filtered_pred = filtered_pred[
        (filtered_pred["date"].dt.date >= start_d) & 
        (filtered_pred["date"].dt.date <= end_d)
    ]

if selected_family != "All Families":
    filtered_inv = filtered_inv[filtered_inv["family"] == selected_family]
    filtered_pred = filtered_pred[filtered_pred["family"] == selected_family]

# ----------------------------------
# 4. Header & Top-level KPIs
# ----------------------------------
st.title("📦 RetailGuard AI: Command Center")
st.markdown(
    "Real-time operational dashboard integrating **Machine Learning demand forecasting**, "
    "**safety stock analytics**, and **automated replenishment triggers**."
)

kpi_sales = filtered_pred["sales"].sum()
kpi_stock = filtered_inv["current_stock"].sum()
kpi_high_risk = (filtered_inv["stockout_risk"] == "HIGH").sum()
kpi_reorder_qty = filtered_inv["reorder_quantity"].sum()
safe_pct = ((filtered_inv["stockout_risk"] == "LOW").sum() / max(len(filtered_inv), 1)) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Predicted Demand", f"{kpi_sales:,.0f} units")
col2.metric("Current Stock on Hand", f"{kpi_stock:,.0f} units")
col3.metric("Stockout Risk Items", f"{int(kpi_high_risk)} items", delta=f"{int(kpi_high_risk)} Critical", delta_color="inverse")
col4.metric("Suggested Reorder", f"{kpi_reorder_qty:,.0f} units")
col5.metric("Inventory Health Index", f"{safe_pct:.1f}% Safe")

st.markdown("---")

# ----------------------------------
# 5. Multi-Tab Core Functionality
# ----------------------------------
tab_forecast, tab_inventory, tab_simulator, tab_model = st.tabs([
    "📈 Demand Forecasting",
    "🚨 Inventory & Replenishment",
    "🧪 What-If Scenario Simulator",
    "🧠 Model Intelligence & Metrics"
])

# ----------------------------------
# TAB 1: Demand Forecasting
# ----------------------------------
with tab_forecast:
    st.subheader(f"Forecasted Demand for Store #{selected_store} ({selected_family})")
    
    if filtered_pred.empty:
        st.warning("No forecast data available for the chosen selection.")
    else:
        # Time series aggregated daily
        daily_trend = filtered_pred.groupby("date")["sales"].sum().reset_index()
        daily_trend["7-Day Rolling Avg"] = daily_trend["sales"].rolling(7, min_periods=1).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily_trend["date"], 
            y=daily_trend["sales"],
            mode="lines+markers",
            name="Daily Predicted Demand",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=6)
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_trend["date"], 
            y=daily_trend["7-Day Rolling Avg"],
            mode="lines",
            name="7-Day Rolling Trend",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ))
        fig_ts.update_layout(
            title="Daily Sales Forecast with Moving Trend",
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_ts, width="stretch")

        # Category Breakdown
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.markdown("#### Top Volume Product Families")
            store_all_preds = predictions[predictions["store_nbr"] == selected_store]
            family_totals = (
                store_all_preds.groupby("family")["sales"]
                .sum()
                .reset_index()
                .sort_values(by="sales", ascending=False)
                .head(10)
            )
            fig_fam = px.bar(
                family_totals,
                x="sales",
                y="family",
                orientation="h",
                color="sales",
                color_continuous_scale="Blues",
                title=f"Top 10 Product Families in Store #{selected_store}"
            )
            fig_fam.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fam, width="stretch")

        with col_c2:
            st.markdown("#### Export Forecast Records")
            st.write(f"Total forecast data points: **{len(filtered_pred):,} rows**")
            csv_pred = filtered_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Predictions (CSV)",
                data=csv_pred,
                file_name=f"forecast_store_{selected_store}_{selected_family}.csv",
                mime="text/csv"
            )
            st.dataframe(filtered_pred[["date", "family", "sales"]].head(15), width="stretch")

# ----------------------------------
# TAB 2: Inventory & Replenishment
# ----------------------------------
with tab_inventory:
    st.subheader(f"Inventory Optimization & Replenishment Plan - Store #{selected_store}")
    
    col_inv1, col_inv2 = st.columns([2, 3])
    with col_inv1:
        # Donut Chart for Stockout Risk
        risk_counts = filtered_inv["stockout_risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
        
        fig_donut = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.45,
            color="Risk Level",
            color_discrete_map=color_map,
            title="Stockout Risk Profile"
        )
        fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, width="stretch")
        
    with col_inv2:
        # Current Stock vs Reorder Point Bar Chart
        top_items = filtered_inv.sort_values(by="reorder_quantity", ascending=False).head(10)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Current Stock",
            x=top_items["family"],
            y=top_items["current_stock"],
            marker_color="#3498db"
        ))
        fig_comp.add_trace(go.Bar(
            name="Reorder Point",
            x=top_items["family"],
            y=top_items["reorder_point"],
            marker_color="#e67e22"
        ))
        fig_comp.update_layout(
            barmode="group",
            title="Current Stock vs Reorder Point (Top 10 Reorders)",
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_comp, width="stretch")

    # Critical Alerts Banner
    high_risk_df = filtered_inv[filtered_inv["stockout_risk"] == "HIGH"]
    if not high_risk_df.empty:
        st.error(f"🚨 **Action Required:** {len(high_risk_df)} product category/categories face imminent stockout risk! Immediate replenishment recommended.")
    else:
        st.success("✅ All product categories are currently within safe operational buffers.")

    # Reorder Plan Table with Filtering
    st.markdown("### 📋 Purchase Order & Reorder Plan")
    filter_choice = st.radio(
        "Display Items:",
        ["Reorder Required Only", "All Inventory Items"],
        horizontal=True
    )
    
    display_inv = filtered_inv[filtered_inv["reorder_quantity"] > 0] if filter_choice == "Reorder Required Only" else filtered_inv

    formatted_inv = display_inv[[
        "family", "current_stock", "lead_time_days", "average_predicted_demand",
        "safety_stock", "reorder_point", "reorder_quantity", "stockout_risk", "recommendation"
    ]].copy()
    
    formatted_inv["safety_stock"] = formatted_inv["safety_stock"].round(1)
    formatted_inv["reorder_point"] = formatted_inv["reorder_point"].round(1)
    formatted_inv["average_predicted_demand"] = formatted_inv["average_predicted_demand"].round(1)

    st.dataframe(formatted_inv, width="stretch")
    
    # Download Procurement Order
    csv_order = formatted_inv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Procurement Order Sheet (CSV)",
        data=csv_order,
        file_name=f"purchase_orders_store_{selected_store}.csv",
        mime="text/csv"
    )

# ----------------------------------
# TAB 3: What-If Scenario Simulator
# ----------------------------------
with tab_simulator:
    st.subheader("🧪 Interactive Supply Chain What-If Simulator")
    st.markdown(
        "Simulate operational shocks such as **demand spikes**, **promotional surges**, "
        "and **supplier lead-time delays** to observe how reorder points and safety stocks dynamically adjust."
    )
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        demand_shock = st.slider("Demand Surge / Promotion Boost (%)", min_value=-50, max_value=150, value=25, step=5)
    with sim_col2:
        lead_time_shock = st.slider("Supplier Lead Time Delay (Days)", min_value=0, max_value=14, value=2, step=1)
    with sim_col3:
        service_level = st.selectbox("Target Service Level", ["90% (Z=1.28)", "95% (Z=1.65 - Default)", "99% (Z=2.33)"], index=1)
    
    # Calculate Z factor
    if "90%" in service_level:
        z_factor = 1.28
    elif "99%" in service_level:
        z_factor = 2.33
    else:
        z_factor = 1.65

    # Simulation engine
    sim_df = filtered_inv.copy()
    sim_df["sim_lead_time"] = sim_df["lead_time_days"] + lead_time_shock
    sim_df["sim_daily_demand"] = sim_df["average_predicted_demand"] * (1.0 + (demand_shock / 100.0))
    sim_df["sim_safety_stock"] = z_factor * sim_df["historical_std"].fillna(0) * np.sqrt(sim_df["sim_lead_time"])
    sim_df["sim_reorder_point"] = (sim_df["sim_daily_demand"] * sim_df["sim_lead_time"]) + sim_df["sim_safety_stock"]
    sim_df["sim_reorder_quantity"] = np.ceil((sim_df["sim_reorder_point"] - sim_df["current_stock"]).clip(lower=0))
    sim_df["sim_stockout_risk"] = np.where(
        sim_df["current_stock"] < (sim_df["sim_daily_demand"] * sim_df["sim_lead_time"]),
        "HIGH",
        np.where(sim_df["current_stock"] < sim_df["sim_reorder_point"], "MEDIUM", "LOW")
    )

    # Simulation KPI Comparison
    base_reorder = filtered_inv["reorder_quantity"].sum()
    new_reorder = sim_df["sim_reorder_quantity"].sum()
    reorder_delta = new_reorder - base_reorder
    
    base_high = (filtered_inv["stockout_risk"] == "HIGH").sum()
    new_high = (sim_df["sim_stockout_risk"] == "HIGH").sum()
    high_delta = new_high - base_high

    s_col1, s_col2 = st.columns(2)
    s_col1.metric("Simulated Total Reorder Quantity", f"{new_reorder:,.0f} units", delta=f"{reorder_delta:+,.0f} units")
    s_col2.metric("Simulated High Risk Categories", f"{int(new_high)} items", delta=f"{high_delta:+d} items", delta_color="inverse")

    st.markdown("#### Simulated Impact Table")
    sim_display = sim_df[[
        "family", "current_stock", "lead_time_days", "sim_lead_time",
        "average_predicted_demand", "sim_daily_demand",
        "reorder_point", "sim_reorder_point",
        "reorder_quantity", "sim_reorder_quantity",
        "stockout_risk", "sim_stockout_risk"
    ]].copy().round(1)
    st.dataframe(sim_display, width="stretch")

# ----------------------------------
# TAB 4: Model Intelligence & Metrics
# ----------------------------------
with tab_model:
    st.subheader("🧠 Machine Learning Model Intelligence & Explainability")
    st.markdown(
        "Detailed inspection of the predictive model, benchmark comparisons, and global feature importance."
    )
    
    col_m1, col_m2 = st.columns([3, 2])
    with col_m1:
        if feature_importance is not None:
            st.markdown("#### Global Feature Importance (Top Predictors)")
            top_feat = feature_importance.head(12).copy()
            top_feat["Feature"] = top_feat["Feature"].str.replace("numeric__", "").str.replace("categorical__", "")
            
            fig_imp = px.bar(
                top_feat,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 12 Features Influencing Sales Forecasts"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_imp, width="stretch")
        else:
            st.info("Feature importance data is not available.")

    with col_m2:
        st.markdown("#### Model Architecture & Specs")
        st.markdown("""
        - **Primary Regressor:** XGBoost Regressor (`tree_method='hist'`)
        - **Objective:** `reg:squarederror` on `log1p(sales)`
        - **Hyperparameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`
        - **Validation RMSLE:** `0.3834`
        - **Key Features:** Recursive Multi-step Lag features (`lag_1`, `lag_7`, `lag_14`), Rolling Averages (`rolling_mean_7`, `rolling_mean_30`), Calendar signals.
        """)
        
        if model_comp is not None:
            st.markdown("#### Direct Validation Benchmark")
            st.dataframe(model_comp.round(3), width="stretch")

        if rec_model_comp is not None:
            st.markdown("#### Recursive Multi-Step Benchmark")
            st.dataframe(rec_model_comp.round(3), width="stretch")

# ----------------------------------
# 6. Footer
# ----------------------------------
st.markdown("---")
st.caption("RetailGuard AI | End-to-End Retail Demand Forecasting & Inventory Optimization System")
