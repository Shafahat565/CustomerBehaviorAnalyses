"""
Intelligent Customer Behavior Dashboard
========================================
A Streamlit GUI for the customer-analytics pipeline (notebooks 01-05).

Run with:
    streamlit run app.py
"""

import os
import json
import pickle
import random

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

# ==============================================================================
# 0. CONSTANTS -- Updated for standalone side-by-side directories
# ==============================================================================
DATA_DIR = "data"
MODEL_DIR = "models"  # Standalone folder sitting next to 'data/' in the root
FEATURE_COLS = [
    "Homeware_Pct", "Stationery_Pct", "Gadgets_Pct", "Decorations_Pct", "Kitchenware_Pct",
    "Recency", "Frequency", "Monetary", "ProductDiversity"
]
ACTION_LABELS = {
    0: "No outreach",
    1: "Email / small discount",
    2: "Phone call / large incentive",
}
ACTION_DESCRIPTIONS = {
    0: "No marketing spend. Appropriate when predicted spend is too low to justify any outreach cost.",
    1: "A low-cost nudge (email or small discount code). Good middle ground for moderate-value customers.",
    2: "The most expensive intervention (a phone call or a larger incentive). Reserve for customers whose predicted spend clearly justifies the extra cost.",
}

# ==============================================================================
# 1. ARTIFACT LOADING -- Cached & path-corrected for clean deployment
# ==============================================================================
class QNetwork(nn.Module):
    def __init__(self, n_features=5, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _try_load(path, loader, label, missing_log):
    if os.path.exists(path):
        try:
            return loader(path), True
        except Exception as e:
            missing_log.append(f"{label} (found but failed to load: {e})")
            return None, False
    missing_log.append(label)
    return None, False


@st.cache_resource(show_spinner="Loading saved model artifacts...")
def load_artifacts():
    missing = []
    splits = {}
    for name in ["train_split.csv", "val_split.csv", "test_split.csv"]:
        path = os.path.join(DATA_DIR, name)
        df, ok = _try_load(path, lambda p: pd.read_csv(p, index_col="CustomerID"), name, missing)
        if ok:
            df["Split"] = name.replace("_split.csv", "").capitalize()
            splits[name] = df

    # Load the category mapping directly from the root directory
    category_map, _ = _try_load(
        "category_map.json", 
        lambda p: json.load(open(p)), 
        "category_map.json", 
        missing
    )

    # Scaler and transformations remain safely housed in DATA_DIR
    scaler, _ = _try_load(os.path.join(DATA_DIR, "scaler.pkl"), lambda p: pickle.load(open(p, "rb")), "scaler.pkl", missing)
    pca, _ = _try_load(os.path.join(DATA_DIR, "pca.pkl"), lambda p: pickle.load(open(p, "rb")), "pca.pkl", missing)
    lda, _ = _try_load(os.path.join(DATA_DIR, "lda.pkl"), lambda p: pickle.load(open(p, "rb")), "lda.pkl", missing)
    pipeline_config, _ = _try_load(os.path.join(DATA_DIR, "pipeline_config.json"), lambda p: json.load(open(p)), "pipeline_config.json", missing)

    # Models and model metadata load cleanly from standalone MODEL_DIR
    classifier, _ = _try_load(os.path.join(MODEL_DIR, "champion_classifier.pkl"), joblib.load, "champion_classifier.pkl", missing)
    clf_meta, _ = _try_load(os.path.join(MODEL_DIR, "champion_classifier_metadata.json"), lambda p: json.load(open(p)), "champion_classifier_metadata.json", missing)

    regressor, _ = _try_load(os.path.join(MODEL_DIR, "champion_regressor.pkl"), joblib.load, "champion_regressor.pkl", missing)
    reg_meta, _ = _try_load(os.path.join(MODEL_DIR, "champion_regressor_metadata.json"), lambda p: json.load(open(p)), "champion_regressor_metadata.json", missing)

    q_table, _ = _try_load(os.path.join(MODEL_DIR, "tabular_q_table.npy"), np.load, "tabular_q_table.npy", missing)
    kmeans, _ = _try_load(os.path.join(MODEL_DIR, "policy_kmeans.pkl"), joblib.load, "policy_kmeans.pkl", missing)

    def _load_qnet(p):
        net = QNetwork()
        net.load_state_dict(torch.load(p, map_location="cpu"))
        net.eval()
        return net

    q_net, _ = _try_load(os.path.join(MODEL_DIR, "dqn_policy.pt"), _load_qnet, "dqn_policy.pt", missing)
    policy_meta, _ = _try_load(os.path.join(MODEL_DIR, "policy_metadata.json"), lambda p: json.load(open(p)), "policy_metadata.json", missing)

    return {
        "splits": splits, "scaler": scaler, "pca": pca, "lda": lda,
        "pipeline_config": pipeline_config, "classifier": classifier, "clf_meta": clf_meta,
        "regressor": regressor, "reg_meta": reg_meta, "q_table": q_table, "kmeans": kmeans,
        "q_net": q_net, "policy_meta": policy_meta, "missing": missing,
        "category_map": category_map,
    }


def build_demo_artifacts():
    rng = np.random.default_rng(42)
    n = 60
    pct = rng.dirichlet(np.ones(len(FEATURE_COLS)), size=n) * 100
    df = pd.DataFrame(pct, columns=FEATURE_COLS)
    df["CustomerID"] = rng.integers(10000, 20000, size=n)
    df = df.set_index("CustomerID")
    df["Recency"] = rng.integers(1, 365, size=n)
    df["Frequency"] = rng.integers(1, 40, size=n)
    df["Monetary"] = rng.gamma(3, 800, size=n)
    df["ProductDiversity"] = rng.integers(1, 40, size=n)
    df["AvgSpendPerTxn"] = df["Monetary"] / df["Frequency"]
    threshold = df["Monetary"].quantile(0.8)
    df["HighValue"] = (df["Monetary"] >= threshold).astype(int)
    df["Split"] = "Demo"

    scaler = StandardScaler().fit(df[FEATURE_COLS])
    X_scaled = scaler.transform(df[FEATURE_COLS])
    pca = PCA(n_components=2).fit(X_scaled)
    lda = LinearDiscriminantAnalysis(n_components=1).fit(X_scaled, df["HighValue"])
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10).fit(df[FEATURE_COLS].to_numpy())
    q_table = rng.uniform(-5, 200, size=(8, 3))
    q_net = QNetwork()

    class DemoClassifier:
        def predict_proba(self, X):
            p = np.clip(np.asarray(X)[:, 2] / 100 + 0.1, 0.02, 0.98)
            return np.stack([1 - p, p], axis=1)

    class DemoRegressor:
        def predict(self, X):
            X = np.asarray(X)
            return X @ np.array([120.0, 200.0, 450.0, 90.0, 250.0]) / 100 * 3.5 + 120

    policy_meta = {
        "best_policy_on_test": "DQN (greedy) [DEMO]",
        "policy_comparison_test_set": {
            "Always Action 0 (No outreach)": {"Avg Reward / Customer": 40.0},
            "Always Action 2 (Phone call)": {"Avg Reward / Customer": 95.0},
            "Random": {"Avg Reward / Customer": 65.0},
            "Tabular Q-Learning (greedy)": {"Avg Reward / Customer": 110.0},
            "DQN (greedy)": {"Avg Reward / Customer": 118.0},
        },
        "used_synthetic_reward_basis": True,
    }

    return {
        "splits": {"demo_split.csv": df}, "scaler": scaler, "pca": pca, "lda": lda,
        "pipeline_config": {"highvalue_threshold": float(threshold), "pca_n_components": 2, "category_columns": FEATURE_COLS},
        "classifier": DemoClassifier(), "clf_meta": {"champion_name": "Demo Classifier", "test_metrics": {"f1": 0.89}},
        "regressor": DemoRegressor(), "reg_meta": {"champion_name": "Demo Regressor", "used_synthetic_surrogate_target": True, "test_metrics": {"RMSE": 48.10, "R2": 0.81}},
        "q_table": q_table, "kmeans": kmeans, "q_net": q_net, "policy_meta": policy_meta, "missing": [],
        "category_map": {},
    }

# ==============================================================================
# 2. PURE LOGIC HELPERS
# ==============================================================================
def combine_customer_universe(artifacts):
    splits = artifacts["splits"]
    df = pd.concat(splits.values(), axis=0) if splits else pd.DataFrame()
    scaler, pca, lda = artifacts["scaler"], artifacts["pca"], artifacts["lda"]
    n_components = (artifacts.get("pipeline_config") or {}).get("pca_n_components", 2)
    
    # Define the exact columns the scaler was fitted on
    scaler_cols = ["Homeware_Pct", "Stationery_Pct", "Gadgets_Pct", "Decorations_Pct", "Kitchenware_Pct"]

    if scaler is not None and pca is not None and len(df) > 0:
        # Use only the 5 columns for the scaler
        X_scaled = scaler.transform(df[scaler_cols])
        pcs = pca.transform(X_scaled)
        df["PC1"] = pcs[:, 0]
        df["PC2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
    else:
        df["PC1"], df["PC2"] = 0.0, 0.0

    if scaler is not None and lda is not None and len(df) > 0:
        # Use only the 5 columns for the scaler
        X_scaled = scaler.transform(df[scaler_cols])
        df["LDA_Comp"] = lda.transform(X_scaled)[:, 0]
    else:
        df["LDA_Comp"] = df["PC1"]
    return df, n_components


def get_feature_vector(row_or_dict):
    return np.array([[float(row_or_dict[c]) for c in FEATURE_COLS]], dtype=np.float64)


def classify_customer(classifier, feature_vector):
    proba = classifier.predict_proba(feature_vector)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])

def predict_spend(regressor, feature_vector):
    # Ensure feature_vector is a DataFrame or indexed array
    # We must slice the 9-feature vector down to the 5 features the regressor was trained on
    # Assuming FEATURE_COLS matches the order used during training:
    
    # Create a temporary DataFrame to make slicing by column name easy
    df_temp = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
    
    # Subset to the 5 features expected by the regressor
    scaler_cols = ["Homeware_Pct", "Stationery_Pct", "Gadgets_Pct", "Decorations_Pct", "Kitchenware_Pct"]
    subset_vector = df_temp[scaler_cols].to_numpy()
    
    return float(regressor.predict(subset_vector)[0])

def project_point(scaler, transformer, feature_vector, n_dims=2):
    if scaler is None or transformer is None:
        return (0.0, 0.0) if n_dims == 2 else (0.0,)
    
    # Convert input to DataFrame for easy column selection
    named_vector = pd.DataFrame(np.asarray(feature_vector), columns=FEATURE_COLS)
    
    # Subset to the 5 columns the scaler expects
    scaler_cols = ["Homeware_Pct", "Stationery_Pct", "Gadgets_Pct", "Decorations_Pct", "Kitchenware_Pct"]
    scaled = scaler.transform(named_vector[scaler_cols])
    
    projected = transformer.transform(scaled)[0]
    if n_dims == 2:
        return (float(projected[0]), float(projected[1]) if len(projected) > 1 else 0.0)
    return (float(projected[0]),)


def get_q_values(artifacts, feature_vector, policy_engine):
    if policy_engine == "Trained DQN Agent":
        q_net = artifacts["q_net"]
        if q_net is None:
            return np.zeros(3)
        
        # 1. Convert feature_vector to DataFrame to allow name-based slicing
        df_temp = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
        
        # 2. Subset to only the 5 features the DQN was trained on
        scaler_cols = ["Homeware_Pct", "Stationery_Pct", "Gadgets_Pct", "Decorations_Pct", "Kitchenware_Pct"]
        subset_arr = df_temp[scaler_cols].to_numpy()
        
        with torch.no_grad():
            t = torch.tensor(subset_arr, dtype=torch.float32) # Now 1x5
            out = q_net(t)
            return out.detach().numpy().reshape(-1)[:3]
    else:
        # K-Means/Tabular logic likely uses all 9 features or a different config
        kmeans, q_table = artifacts["kmeans"], artifacts["q_table"]
        if kmeans is None or q_table is None:
            return np.zeros(3)
        cluster = int(kmeans.predict(feature_vector)[0])
        return q_table[cluster]


def get_policy_comparison_table(policy_meta):
    comparison = (policy_meta or {}).get("policy_comparison_test_set", {})
    rows = []
    for policy_name, metrics in comparison.items():
        rows.append({
            "Policy": policy_name,
            "Avg Reward / Customer": metrics.get("Avg Reward / Customer", metrics.get("avg_reward", 0.0)),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# 3. GLOBAL CONFIGURATION & INTERACTIVE THEME INJECTION
# ==============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Color Scheme Specifications
VIVID_YELLOW = "#FFEA00"
BRIGHT_GOLD   = "#FFD700"
GREEN_NEON    = "#00E676"
RED_PLOT      = "#FF3333"

def inject_ultimate_ui(dark: bool):
    if dark:
        bg = "#080808"
        card_bg = "#141414"
        border = "#2E2E2E"
        text = "#FFFFFF"
        subtext = "#BBBBBB"
        header_bg = "#030303"
    else:
        bg = "#F4F6F9"
        card_bg = "#FFFFFF"
        border = "#CCCCCC"
        text = "#111111"
        subtext = "#444444"
        header_bg = "#16324A"

    st.markdown(f"""
        <style>
        /* Base View Containers */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {bg} !important;
            color: {text} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        /* Proportional Scaling Container Padding */
        [data-testid="stMainBlockContainer"] {{
            padding: 3.5rem 6rem !important;
            max-width: 100% !important;
        }}
        
        /* WIDER ACCENTED SIDEBAR SYSTEM */
        [data-testid="stSidebar"] {{
            background-color: {header_bg} !important;
            border-right: 3px solid {VIVID_YELLOW} !important;
            width: 420px !important;
        }}
        [data-testid="stSidebarUserContent"] {{
            padding: 2.5rem 2rem !important;
        }}
        
        /* Massive High-Visibility Global Headers */
        .top-header {{
            background: {header_bg};
            color: #FFFFFF;
            padding: 24px 32px;
            margin: -3.5rem -6rem 35px -6rem;
            font-size: 38px;
            font-weight: 900;
            letter-spacing: -0.5px;
            border-bottom: 5px solid {VIVID_YELLOW};
            text-shadow: 0px 2px 4px rgba(255, 234, 0, 0.25);
        }}
        
        .sidebar-heading {{
            font-size: 16px;
            font-weight: 900;
            letter-spacing: 1.5px;
            color: {VIVID_YELLOW};
            margin-bottom: 18px;
            text-transform: uppercase;
        }}
        
        .section-header {{
            font-size: 28px !important;
            font-weight: 800 !important;
            color: {text};
            margin-top: 45px;
            margin-bottom: 25px;
            text-transform: uppercase;
            border-left: 7px solid {VIVID_YELLOW};
            padding-left: 16px;
        }}
        
        /* Giant Profile Summary Widget Blocks */
        .metric-card {{
            background: {card_bg};
            border: 2px solid {border};
            border-left: 8px solid {VIVID_YELLOW} !important;
            border-radius: 14px;
            padding: 28px 24px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        }}
        .metric-card .label {{
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {subtext};
            margin-bottom: 6px;
        }}
        .metric-card .value {{
            font-size: 38px;
            font-weight: 900;
            color: {VIVID_YELLOW};
        }}
        
        /* Dashboard Presentation Panels */
        .panel {{
            background: {card_bg};
            border: 2px solid {border};
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.15);
            height: 100%;
        }}
        .panel-title {{
            font-size: 20px;
            font-weight: 800;
            color: {text};
            margin-bottom: 16px;
            border-bottom: 2px solid {border};
            padding-bottom: 8px;
        }}
        
        /* High Performance Prediction Rows */
        .pred-row {{
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
            font-size: 18px;
        }}
        .pred-row.hi {{
            background: rgba(0, 230, 118, 0.15);
            border: 2px solid {GREEN_NEON};
            color: {text};
        }}
        .pred-row.lo {{
            background: rgba(255, 51, 51, 0.15);
            border: 2px solid {RED_PLOT};
            color: {text};
        }}
        .pred-row.neutral {{
            background: {bg};
            border: 2px solid {border};
            color: {text};
        }}
        .pred-row b {{
            font-size: 22px;
            color: {VIVID_YELLOW};
        }}
        .confidence-line {{
            font-size: 14px;
            color: {subtext};
            margin-top: 10px;
            font-weight: 600;
        }}
        
        /* Giant RL Strategy Outreach Options */
        .action-card {{
            border-radius: 14px;
            padding: 28px;
            min-height: 240px;
            border: 2px solid {border};
            background: {card_bg};
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        .action-card.selected {{
            border: 3px solid {GREEN_NEON};
            background: rgba(0, 230, 118, 0.08);
        }}
        .action-tag {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 900;
        }}
        .action-title {{
            font-size: 22px;
            font-weight: 900;
            margin: 8px 0;
            color: {text};
        }}
        .action-desc {{
            font-size: 15px;
            color: {subtext};
            line-height: 1.5;
        }}
        .action-q {{
            font-size: 28px;
            font-weight: 900;
            margin-top: 15px;
        }}
        
        /* Form Element Controls */
        [data-testid="stWidgetLabel"] p {{
            font-size: 18px !important;
            font-weight: 800 !important;
            color: {text} !important;
        }}
        div[data-baseweb="select"] {{
            font-size: 16px !important;
        }}
        
        /* Main Sidebar Action Configuration Button */
        div[data-testid="stSidebar"] button {{
            background-color: {VIVID_YELLOW} !important;
            color: #000000 !important;
            font-weight: 900 !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 12px !important;
        }}
        
        .demo-banner {{
            background: rgba(255, 234, 0, 0.15);
            border: 2px solid {VIVID_YELLOW};
            color: {text};
            padding: 16px;
            border-radius: 10px;
            font-size: 16px;
            margin-bottom: 25px;
        }}
        
        .stat-bar {{
            background: {card_bg};
            border: 2px solid {border};
            border-radius: 10px;
            padding: 16px 24px;
            font-size: 16px;
            color: {text};
            margin-top: 15px;
            font-weight: 700;
        }}
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. INITIALIZATION & DATA RECOVERY PIPELINE
# ==============================================================================
_real_artifacts = load_artifacts()
DEMO_MODE = len(_real_artifacts["splits"]) == 0 or _real_artifacts["classifier"] is None

artifacts = build_demo_artifacts() if DEMO_MODE else _real_artifacts
customer_features, n_components = combine_customer_universe(artifacts)

# ==============================================================================
# 5. SIDEBAR CONTROL PANEL ENGINE
# ==============================================================================
st.sidebar.markdown('<div class="sidebar-heading">Control Panel</div>', unsafe_allow_html=True)

st.session_state.dark_mode = st.sidebar.checkbox("Enable Dark Dashboard Theme", value=st.session_state.dark_mode)
inject_ultimate_ui(st.session_state.dark_mode)

st.sidebar.markdown("### Select Customer Target")
selected_id = st.sidebar.selectbox("Customer ID", options=sorted(customer_features.index.tolist()), index=0)

st.sidebar.markdown("### Dimensionality Method")
proj_choice = st.sidebar.radio("Projection", options=["PCA", "LDA"], horizontal=True)

st.sidebar.markdown("### Optimization Engine")
policy_choice = st.sidebar.selectbox("Policy engine", options=["Trained DQN Agent", "Tabular Q-Learning"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Real-Time Pipeline What-If Override")
override_active = st.sidebar.checkbox("Enable Parameters Overrides", value=False)

base_row = customer_features.loc[selected_id].copy()

if override_active:
    override_values = {}
    for col in FEATURE_COLS:
        override_values[col] = st.sidebar.slider(
            col.replace("_Pct", ""), 0.0, 100.0, float(base_row[col]), step=1.0
        )
    row = base_row.copy()
    for col, val in override_values.items():
        row[col] = val
else:
    row = base_row

st.sidebar.markdown("---")
st.sidebar.button("Run Simulation Analysis", use_container_width=True)

# ==============================================================================
# 6. APP MAIN HEADER FRAME
# ==============================================================================
st.markdown('<div class="top-header">Intelligent Customer Behavior &amp; Marketing Strategy Dashboard</div>', unsafe_allow_html=True)

if DEMO_MODE:
    missing_list = ", ".join(_real_artifacts["missing"][:4])
    st.markdown(
        f'<div class="demo-banner">⚠️ <strong>Active Demo Mode Placeholder</strong> — Pipeline components missing: '
        f'({missing_list}). Utilizing synthetic dataset fallback engine.</div>',
        unsafe_allow_html=True,
    )

feature_vector = get_feature_vector(row)
is_high = int(row.get("HighValue", 0)) == 1
segment_str = "High-Value Target" if is_high else "Standard-Value Target"

st.markdown(f'<div class="section-header">Customer Identity Profile Summary: ID {selected_id}</div>', unsafe_allow_html=True)

# Display Row Information Matrices
m1, m2, m3, m4 = st.columns(4)
metrics_payload = [
    ("Recency Index", f"{int(row.get('Recency', 0))} Days Ago"),
    ("Frequency Volume", f"{int(row.get('Frequency', 0))} Orders Placed"),
    ("Monetary Value", f"${row.get('Monetary', 0):,.2f}"),
    ("Pipeline Segment Label", segment_str)
]

for col, (label, value) in zip([m1, m2, m3, m4], metrics_payload):
    col.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div></div>', unsafe_allow_html=True)

# ==============================================================================
# 7. MULTIVARIATE VISUALIZATIONS AND ACCELERATED PREDICTIONS
# ==============================================================================
st.markdown('<div class="section-header">Analytical Projections & Advanced Risk Models</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([7, 5], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{proj_choice} Topological Geometric Feature Map</div>', unsafe_allow_html=True)

    # Apply specialized High-Contrast Plotly configuration schemes
    theme_bg = "rgba(20,20,20,1)" if st.session_state.dark_mode else "rgba(255,255,255,1)"
    theme_grid = "#333333" if st.session_state.dark_mode else "#E0E0E0"
    theme_text = "#FFFFFF" if st.session_state.dark_mode else "#111111"

    if proj_choice == "PCA":
        pt = project_point(artifacts["scaler"], artifacts["pca"], feature_vector, n_dims=2)
        fig = px.scatter(
            customer_features.reset_index(), x="PC1", y="PC2", color="HighValue",
            color_continuous_scale=[RED_PLOT, VIVID_YELLOW],
            labels={"PC1": "Principal Component Vector 1", "PC2": "Principal Component Vector 2"},
        )
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]], mode="markers",
            marker=dict(size=20, color=GREEN_NEON, line=dict(width=4, color="#FFFFFF")),
            name="Selected Position Target",
        ))
    else:
        pt = project_point(artifacts["scaler"], artifacts["lda"], feature_vector, n_dims=1)
        fig = px.strip(
            customer_features.reset_index(), x="LDA_Comp", color="HighValue",
            color_discrete_sequence=[RED_PLOT, VIVID_YELLOW],
        )
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[0], mode="markers",
            marker=dict(size=22, color=GREEN_NEON, symbol="diamond", line=dict(width=4, color="#FFFFFF")),
            name="Selected Position Target",
        ))

    fig.update_layout(
        margin=dict(l=15, r=15, t=15, b=15), 
        height=380, 
        showlegend=False,
        plot_bgcolor=theme_bg, 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme_text, size=13),
        xaxis=dict(gridcolor=theme_grid, zeroline=False),
        yaxis=dict(gridcolor=theme_grid, zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
        <div style="display: flex; gap: 20px; font-weight: bold; font-size: 14px; margin-top: 10px; justify-content: center;">
            <span><span style="color:{VIVID_YELLOW}; font-size:18px;">■</span> Pipeline High-Value</span>
            <span><span style="color:{RED_PLOT}; font-size:18px;">■</span> Pipeline Standard/Low</span>
            <span><span style="color:{GREEN_NEON}; font-size:18px;">▲</span> Currently Selected Target</span>
        </div>
    </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model Diagnostics Outputs</div>', unsafe_allow_html=True)

    proba = classify_customer(artifacts["classifier"], feature_vector)
    row_class = "hi" if proba >= 0.5 else "lo"
    label_text = "High-Value Classification Prediction" if proba >= 0.5 else "Standard/Low-Value Target"
    conf = proba if proba >= 0.5 else (1 - proba)
    
    st.markdown(
        f'<div class="pred-row {row_class}">Classification Variant — (Target Probability)<br/>'
        f'<b>{label_text}</b> &nbsp;({conf*100:.1f}%)</div>', unsafe_allow_html=True,
    )

    pred_spend = predict_spend(artifacts["regressor"], feature_vector)
    reg_meta = artifacts.get("reg_meta") or {}
    rmse = (reg_meta.get("test_metrics") or {}).get("RMSE")
    rmse_str = f" (Error Spread Margin: \u00b1 ${rmse:,.2f} RMSE)" if rmse is not None else ""
    
    st.markdown(
        f'<div class="pred-row neutral">Regression Variant — Predicted Financial Spend<br/>'
        f'<b>${pred_spend:,.2f}</b>{rmse_str}</div>', unsafe_allow_html=True,
    )

    clf_meta = artifacts.get("clf_meta") or {}
    clf_metrics = clf_meta.get("test_metrics") or {}
    f1 = clf_metrics.get("f1")
    r2 = (reg_meta.get("test_metrics") or {}).get("R2")
    
    conf_bits = []
    if f1 is not None: conf_bits.append(f"F1-Score Model Verification Metric: {f1:.2f}")
    if r2 is not None:  conf_bits.append(f"R² Fit Metric: {r2:.2f}")
    
    if conf_bits:
        st.markdown(f'<div class="confidence-line">Pipeline Performance Verification Logs: <br/>{"  ·  ".join(conf_bits)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 8. REINFORCEMENT LEARNING MARKOV RECOMMENDATION MATRIX
# ==============================================================================
st.markdown('<div class="section-header">Reinforcement Learning Strategy Evaluation Engine</div>', unsafe_allow_html=True)

q_values = get_q_values(artifacts, feature_vector, policy_choice)
best_action = int(np.argmax(q_values))

action_cols = st.columns(3)
for i, col in enumerate(action_cols):
    is_best = (i == best_action)
    css_class = "action-card selected" if is_best else "action-card"
    tag_color = GREEN_NEON if is_best else "#8A93A0"
    tag_text = "★ Optimal Policy Objective Selected" if is_best else "Alternative System Pathway"
    with col:
        st.markdown(f"""
            <div class="{css_class}">
                <div>
                    <span class="action-tag" style="color:{tag_color};">{tag_text}</span>
                    <div class="action-title">{ACTION_LABELS[i]}</div>
                    <div class="action-desc">{ACTION_DESCRIPTIONS[i]}</div>
                </div>
                <div class="action-q" style="color:{VIVID_YELLOW if is_best else '#8A93A0'};">Calculated Expected Value Q = {q_values[i]:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

policy_meta = artifacts.get("policy_meta") or {}
comparison_df = get_policy_comparison_table(policy_meta)

if len(comparison_df) > 0:
    best_reward = comparison_df["Avg Reward / Customer"].max()
    random_row = comparison_df[comparison_df["Policy"].str.contains("Random", case=False, na=False)]
    baseline_reward = float(random_row["Avg Reward / Customer"].iloc[0]) if len(random_row) else 40.0
    improvement = ((best_reward - baseline_reward) / baseline_reward) * 100 if baseline_reward else 0.0
    
    st.markdown(
        f'<div class="stat-bar">'
        f'🚀 Optimized Deployment Strategy: <b>{policy_meta.get("best_policy_on_test", "DQN Framework")}</b> Model Variant '
        f'&nbsp;&nbsp;|&nbsp;&nbsp; Simulation Baseline Engine Reward: <b>${baseline_reward:.2f}</b> '
        f'&nbsp;&nbsp;|&nbsp;&nbsp; Pipeline Operational Efficiency Lift vs Random Policy: <span style="color:{GREEN_NEON};"><b>+{improvement:.1f}% Yield Increase</b></span>'
        f'</div>', unsafe_allow_html=True
    )