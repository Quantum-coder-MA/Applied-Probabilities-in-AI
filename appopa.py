# =============================================================================
#  app.py — Application Streamlit Multi-Projets Data Science
#  Projets : Cancer (ML), Blessures Sportives, Simulations Statistiques
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Configuration globale de la page ─────────────────────────────────────────
st.set_page_config(
    page_title="DataScience Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1d27 0%, #0f1117 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #262b40 100%);
        border: 1px solid #3d4266;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c83f5; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 5px; }
    .section-header {
        background: linear-gradient(90deg, #7c83f5 0%, #56cfad 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue   { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .badge-green  { background: #14432a; color: #4ade80; border: 1px solid #16a34a; }
    .badge-purple { background: #2e1065; color: #c084fc; border: 1px solid #9333ea; }
    .badge-red    { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
    hr { border-color: #2d3148; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  COULEURS & THÈME PLOTLY
# =============================================================================

COULEURS = {
    "primary":   "#7c83f5",
    "secondary": "#56cfad",
    "danger":    "#f87171",
    "warning":   "#fbbf24",
    "purple":    "#c084fc",
    "orange":    "#fb923c",
    "bg":        "#1e2130",
    "grid":      "#2d3148",
}

def plotly_theme():
    return dict(
        plot_bgcolor  = COULEURS["bg"],
        paper_bgcolor = COULEURS["bg"],
        font          = dict(color="#c9d1d9", size=12),
        xaxis         = dict(gridcolor=COULEURS["grid"], zerolinecolor=COULEURS["grid"]),
        yaxis         = dict(gridcolor=COULEURS["grid"], zerolinecolor=COULEURS["grid"]),
        margin        = dict(l=40, r=20, t=50, b=40),
    )


# =============================================================================
#  FONCTIONS DE CHARGEMENT DES DONNÉES (mis en cache)
# =============================================================================

@st.cache_data(show_spinner="Chargement des données Cancer…")
def load_cancer():
    try:
        df = pd.read_csv("dataCancer.csv")
    except FileNotFoundError:
        st.error("❌ Fichier 'dataCancer.csv' introuvable.")
        return None, None, None, None, None, None, None

    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["target"] = (df["diagnosis"] == "B").astype(int)
    FEAT_COLS = [c for c in df.columns if c not in ["diagnosis", "target"]]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[FEAT_COLS].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return df, FEAT_COLS, X_train_s, X_test_s, y_train, y_test, scaler


@st.cache_data(show_spinner="Chargement des données Blessures…")
def load_sports():
    try:
        df = pd.read_csv("multimodal_sports_injury_dataset.csv")
    except FileNotFoundError:
        st.error("❌ Fichier 'multimodal_sports_injury_dataset.csv' introuvable.")
        return None

    df.fillna(df.median(numeric_only=True), inplace=True)
    df["injury_occurred"] = df["injury_occurred"].replace({2: 1})
    df = df.drop(columns=["athlete_id", "session_id"], errors="ignore")

    df["fatigue_recovery_ratio"] = df["fatigue_index"] / df["recovery_score"].replace(0, 1e-9)
    df["sleep_stress_index"]     = df["stress_level"] / df["sleep_quality"].replace(0, 1e-9)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer

    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    ohe = OneHotEncoder(sparse_output=False)
    sport_encoded = ohe.fit_transform(df[["sport_type"]])
    sport_cols    = ohe.get_feature_names_out(["sport_type"])
    df = pd.concat([df.drop("sport_type", axis=1),
                    pd.DataFrame(sport_encoded, columns=sport_cols, index=df.index)], axis=1)

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("injury_occurred", errors="ignore")
    pt = PowerTransformer(method="yeo-johnson")
    df[num_cols] = pt.fit_transform(df[num_cols])

    return df


@st.cache_resource(show_spinner="Entraînement des modèles Cancer…")
def train_cancer_models():
    result = load_cancer()
    if result[0] is None:
        return None
    df, FEAT_COLS, X_train_s, X_test_s, y_train, y_test, scaler = result

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

    models = {
        "Naive Bayes":   GaussianNB(),
        "KNN (k=7)":     KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        cm      = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        results[name] = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "auc":         roc_auc_score(y_test, y_proba),
            "f1":          f1_score(y_test, y_pred),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "cm":          cm,
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }
        trained[name] = model

    return results, trained, scaler, FEAT_COLS, y_test


@st.cache_resource(show_spinner="Entraînement des modèles Sports (base)…")
def train_sports_models():
    """Entraîne NB, KNN et RF avec SMOTE sur le dataset Blessures."""
    df = load_sports()
    if df is None:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
    from imblearn.over_sampling import SMOTE

    STRONG_FEATURES = [
        "fatigue_recovery_ratio", "recovery_score", "fatigue_index",
        "training_load", "sleep_quality", "stress_level", "sleep_stress_index",
        "heart_rate", "training_intensity"
    ]
    feat_cols = [c for c in STRONG_FEATURES if c in df.columns]

    X = df[feat_cols]
    y = df["injury_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = {
        "Naive Bayes":   GaussianNB(),
        "KNN (k=15)":    KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cm      = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        results[name] = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "auc":         roc_auc_score(y_test, y_proba),
            "f1":          f1_score(y_test, y_pred),
            "dice":        dice,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "cm":          cm,
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }
        trained[name] = model

    return results, trained, feat_cols, y_test, X_train, X_test, y_train


@st.cache_resource(show_spinner="Entraînement des modèles avancés Sports…")
def train_sports_advanced_models():
    """
    Entraîne :
      - Decision Tree
      - Marbar-RF (Random Forest + pondération Markovienne)
      - Modèle Hybride (Weibull + Benford + RF + NB)
      - GF-MFTL (Gradient-Free Mean-Field-Type Learning)
    """
    df = load_sports()
    if df is None:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                                  roc_auc_score, confusion_matrix, log_loss,
                                  mean_squared_log_error, cohen_kappa_score,
                                  matthews_corrcoef)
    from scipy.stats import weibull_min
    from imblearn.over_sampling import SMOTE

    X_full = df.drop("injury_occurred", axis=1)
    y_full = df["injury_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # ── Helper: advanced metrics ─────────────────────────────────────────────
    def adv_metrics(y_true, y_pred, y_proba_2d, label):
        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred)
        rec   = recall_score(y_true, y_pred)
        cm    = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0
        auc   = roc_auc_score(y_true, y_proba_2d[:, 1])
        ll    = log_loss(y_true, y_proba_2d)
        rmsle = float(np.sqrt(mean_squared_log_error(
                    np.clip(y_true, 0, None), np.clip(y_pred, 0, None))))
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc   = matthews_corrcoef(y_true, y_pred)
        dice  = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {
            "label": label, "accuracy": acc, "f1": f1, "recall": rec,
            "fpr": fpr_v, "auc": auc, "log_loss": ll, "rmsle": rmsle,
            "kappa": kappa, "mcc": mcc, "dice": dice,
            "cm": cm, "y_pred": y_pred, "y_proba": y_proba_2d,
        }

    results = {}

    # ── 1. Decision Tree ─────────────────────────────────────────────────────
    dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_model.fit(X_train_bal, y_train_bal)
    dt_pred   = dt_model.predict(X_test)
    dt_proba  = dt_model.predict_proba(X_test)
    results["Decision Tree"] = adv_metrics(y_test, dt_pred, dt_proba, "Decision Tree")
    results["Decision Tree"]["model"] = dt_model
    results["Decision Tree"]["feat_names"] = list(X_full.columns)

    # ── 2. Marbar-RF (Markovian Random Forest) ───────────────────────────────
    rf_base = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    rf_base.fit(X_train_bal, y_train_bal)

    tree_preds = np.array([tree.predict(X_test) for tree in rf_base.estimators_])
    n_classes = 2
    trans_mat = np.zeros((n_classes, n_classes))
    for i in range(len(rf_base.estimators_) - 1):
        for j in range(len(X_test)):
            s      = int(tree_preds[i, j])
            s_next = int(tree_preds[i + 1, j])
            trans_mat[s, s_next] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_mat /= row_sums

    rf_proba_base  = rf_base.predict_proba(X_test)
    marbar_proba   = rf_proba_base @ trans_mat
    marbar_pred    = np.argmax(marbar_proba, axis=1)
    results["Marbar-RF"] = adv_metrics(y_test, marbar_pred, marbar_proba, "Marbar-RF")
    results["Marbar-RF"]["transition_matrix"] = trans_mat

    # ── 3. Hybrid model (Weibull + Benford features + RF + NB ensemble) ──────
    def weibull_feat(col_vals, c_shape):
        shifted = col_vals - col_vals.min() + 0.01
        return weibull_min.pdf(shifted, c_shape)

    def benford_feat(col_vals):
        out = []
        for v in col_vals:
            v_abs = abs(v)
            s = f"{v_abs:.6g}".replace(".", "").lstrip("0")
            d = int(s[0]) if s else 1
            out.append(np.log10(1 + 1 / d))
        return np.array(out)

    # Fit Weibull on fatigue_index
    fatigue_tr = X_train["fatigue_index"].values if hasattr(X_train, "columns") else X_train[:, 0]
    fatigue_tr_sh = fatigue_tr - fatigue_tr.min() + 0.01
    c_w, _, _ = weibull_min.fit(fatigue_tr_sh, floc=0)

    X_train_h = X_train.copy()
    X_test_h  = X_test.copy()

    X_train_h["weibull_fatigue"]  = weibull_feat(X_train_h["fatigue_index"].values, c_w)
    X_test_h["weibull_fatigue"]   = weibull_feat(X_test_h["fatigue_index"].values, c_w)
    X_train_h["benford_training"] = benford_feat(X_train_h["training_load"].values)
    X_test_h["benford_training"]  = benford_feat(X_test_h["training_load"].values)
    X_train_h["weibull_recovery"] = weibull_feat(X_train_h["recovery_score"].values, c_w)
    X_test_h["weibull_recovery"]  = weibull_feat(X_test_h["recovery_score"].values, c_w)

    X_train_h_bal, y_train_h_bal = smote.fit_resample(X_train_h, y_train)

    rf_hyb = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    nb_hyb = GaussianNB()
    rf_hyb.fit(X_train_h_bal, y_train_h_bal)
    nb_hyb.fit(X_train_h_bal, y_train_h_bal)

    rf_proba_h = rf_hyb.predict_proba(X_test_h)
    nb_proba_h = nb_hyb.predict_proba(X_test_h)
    hybrid_proba = 0.6 * rf_proba_h + 0.4 * nb_proba_h
    hybrid_pred  = np.argmax(hybrid_proba, axis=1)
    results["Hybrid (Weibull+Benford)"] = adv_metrics(
        y_test, hybrid_pred, hybrid_proba, "Hybrid (Weibull+Benford)")
    results["Hybrid (Weibull+Benford)"]["weibull_shape"] = c_w

    # ── 4. GF-MFTL (Gradient-Free Mean-Field-Type Learning) ──────────────────
    class GF_MFTL:
        def __init__(self, n_particles=30, n_iter=80, sigma=0.15, random_state=42):
            self.n_particles = n_particles
            self.n_iter = n_iter
            self.sigma = sigma
            self.rs = random_state
            self.history_ = []

        def _sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

        def _loss(self, w, b, X, y):
            p = np.clip(self._sigmoid(X @ w + b), 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        def fit(self, X, y):
            rng = np.random.RandomState(self.rs)
            nf = X.shape[1]
            W = rng.randn(self.n_particles, nf) * 0.05
            B = rng.randn(self.n_particles) * 0.05
            self.history_ = []
            for _ in range(self.n_iter):
                losses   = np.array([self._loss(W[i], B[i], X, y) for i in range(self.n_particles)])
                best_idx = np.argsort(losses)[:self.n_particles // 2]
                mean_W   = W[best_idx].mean(axis=0)
                mean_B   = B[best_idx].mean()
                W = W + self.sigma * (mean_W - W) + rng.randn(*W.shape) * self.sigma * 0.3
                B = B + self.sigma * (mean_B - B) + rng.randn(self.n_particles) * self.sigma * 0.3
                self.history_.append(float(losses.min()))
            best = np.argmin([self._loss(W[i], B[i], X, y) for i in range(self.n_particles)])
            self.weights_ = W[best]
            self.bias_    = B[best]
            return self

        def predict_proba(self, X):
            p1 = self._sigmoid(X @ self.weights_ + self.bias_)
            return np.column_stack([1 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    from sklearn.preprocessing import StandardScaler as SS
    sc = SS()
    X_train_sc = sc.fit_transform(X_train_bal)
    X_test_sc  = sc.transform(X_test)

    gf = GF_MFTL(n_particles=30, n_iter=80, sigma=0.15, random_state=42)
    gf.fit(X_train_sc, y_train_bal.values)
    gf_proba = gf.predict_proba(X_test_sc)
    gf_pred  = gf.predict(X_test_sc)

    results["GF-MFTL"] = adv_metrics(y_test, gf_pred, gf_proba, "GF-MFTL")
    results["GF-MFTL"]["convergence"] = gf.history_

    return results, y_test


# =============================================================================
#  HELPERS GRAPHIQUES PLOTLY (shared)
# =============================================================================

def fig_confusion_matrix(cm, labels=("Maligne/Inj.", "Bénigne/Sain"), title="Matrice de Confusion"):
    fig = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Compte"),
        x=list(labels), y=list(labels),
        color_continuous_scale="Blues",
        text_auto=True,
        title=title,
    )
    fig.update_layout(**plotly_theme(), height=320)
    fig.update_traces(textfont_size=16)
    return fig


def fig_roc_curves(models_results, y_test_ref, proba_key="y_proba"):
    from sklearn.metrics import roc_curve
    fig = go.Figure()
    palette = [COULEURS["primary"], COULEURS["secondary"], COULEURS["danger"],
               COULEURS["warning"], COULEURS["purple"], COULEURS["orange"]]
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="#4b5563"))
    for i, (name, res) in enumerate(models_results.items()):
        proba = res[proba_key]
        if proba.ndim == 2:
            proba = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_test_ref, proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['auc']:.3f})",
            line=dict(color=palette[i % len(palette)], width=2.5)
        ))
    fig.update_layout(
        **plotly_theme(), title="Courbes ROC — Comparaison des modèles",
        xaxis_title="FPR", yaxis_title="TPR", height=380
    )
    return fig


def fig_metrics_bar(models_results):
    noms = list(models_results.keys())
    accs = [v["accuracy"] for v in models_results.values()]
    aucs = [v["auc"]      for v in models_results.values()]
    f1s  = [v["f1"]       for v in models_results.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=noms, y=accs, marker_color=COULEURS["primary"]))
    fig.add_trace(go.Bar(name="AUC-ROC",  x=noms, y=aucs, marker_color=COULEURS["secondary"]))
    fig.add_trace(go.Bar(name="F1-Score", x=noms, y=f1s,  marker_color=COULEURS["danger"]))
    fig.update_layout(**plotly_theme(), barmode="group",
                      title="Comparaison des métriques", height=350, yaxis_range=[0, 1.05])
    return fig


# =============================================================================
#  PAGE 1 — ACCUEIL
# =============================================================================

def page_accueil():
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px'>
        <h1 style='font-size:3rem; background: linear-gradient(90deg,#7c83f5,#56cfad);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   font-weight:800; letter-spacing:-1px;'>
            🔬 DataScience Dashboard
        </h1>
        <p style='color:#8892b0; font-size:1.1rem; max-width:700px; margin:0 auto;'>
            Plateforme interactive unifiée — Machine Learning, Analyse Médicale &amp; Simulations Statistiques
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #7c83f5;">
            <div style="font-size:3rem; margin-bottom:12px;">🩺</div>
            <h3 style="color:#e2e8f0; margin:0 0 10px;">Diagnostic du Cancer</h3>
            <p style="color:#8892b0; font-size:0.9rem; line-height:1.6;">
                Classification des tumeurs du sein (bénigne / maligne) sur le dataset
                <strong style="color:#c9d1d9;">Wisconsin Breast Cancer</strong> (569 patients, 30 features).
            </p><br>
            <span class="badge badge-blue">Naive Bayes</span>&nbsp;
            <span class="badge badge-blue">KNN</span>&nbsp;
            <span class="badge badge-blue">Decision Tree</span>&nbsp;
            <span class="badge badge-blue">Random Forest</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #56cfad;">
            <div style="font-size:3rem; margin-bottom:12px;">🏃‍♂️</div>
            <h3 style="color:#e2e8f0; margin:0 0 10px;">Blessures Sportives</h3>
            <p style="color:#8892b0; font-size:0.9rem; line-height:1.6;">
                Prédiction de blessures chez 15 420 athlètes. Modèles avancés :
                <strong style="color:#c9d1d9;">Marbar-RF, Hybride Weibull+Benford, GF-MFTL</strong>.
            </p><br>
            <span class="badge badge-green">SMOTE</span>&nbsp;
            <span class="badge badge-green">Markov RF</span>&nbsp;
            <span class="badge badge-green">GF-MFTL</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #f87171;">
            <div style="font-size:3rem; margin-bottom:12px;">🎲</div>
            <h3 style="color:#e2e8f0; margin:0 0 10px;">Simulations Statistiques</h3>
            <p style="color:#8892b0; font-size:0.9rem; line-height:1.6;">
                Comparaison interactive de lois discrètes et continues —
                <strong style="color:#c9d1d9;">Bernoulli, Binomiale, Poisson, Gamma, Beta…</strong>
            </p><br>
            <span class="badge badge-purple">Monte Carlo</span>&nbsp;
            <span class="badge badge-purple">N=10 000</span>&nbsp;
            <span class="badge badge-purple">Temps réel</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    st.markdown("### 📊 Aperçu rapide des datasets")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "569",    "Patients (Cancer)"),
        (c2, "15 420", "Athlètes (Sports)"),
        (c3, "30",     "Features Cancer"),
        (c4, "29",     "Features Sports"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📖 Guide d'utilisation", expanded=False):
        st.markdown("""
        **Comment utiliser cette application :**
        1. **Barre latérale** → Sélectionnez l'un des 4 modules dans le menu de navigation.
        2. **Modules ML** → Les modèles s'entraînent automatiquement au premier chargement (mis en cache).
        3. **Simulateurs** → Ajustez les sliders pour modifier les paramètres en temps réel.
        4. **Fichiers requis** → Placez `dataCancer.csv` et `multimodal_sports_injury_dataset.csv` dans le même dossier que `app.py`.

        **Lancement :**
        ```bash
        pip install streamlit pandas numpy scikit-learn plotly imbalanced-learn scipy
        streamlit run app.py
        ```
        """)


# =============================================================================
#  PAGE 2 — DIAGNOSTIC DU CANCER
# =============================================================================

def page_cancer():
    st.markdown('<p class="section-header">🩺 Diagnostic du Cancer du Sein</p>', unsafe_allow_html=True)
    st.markdown("**Dataset :** Wisconsin Breast Cancer · 569 patients · 30 features morphologiques · Cible : Bénigne (B) / Maligne (M)")
    st.markdown("---")

    data_result  = load_cancer()
    model_result = train_cancer_models()

    if data_result[0] is None or model_result is None:
        st.error("Impossible de charger les données ou d'entraîner les modèles.")
        return

    df, FEAT_COLS, X_train_s, X_test_s, y_train, y_test, scaler = data_result
    models_results, trained_models, _, _, _ = model_result

    tab_eda, tab_bayes, tab_models, tab_sim = st.tabs([
        "📊 Exploration des données",
        "🧮 Analyse Bayésienne",
        "🤖 Modèles ML",
        "🔮 Simulateur",
    ])

    # ── EDA ──────────────────────────────────────────────────────────────────
    with tab_eda:
        st.subheader("Statistiques descriptives")
        c1, c2, c3, c4 = st.columns(4)
        counts = df["diagnosis"].value_counts()
        for col, val, label in [
            (c1, str(len(df)), "Total patients"),
            (c2, str(counts.get("B", 0)), "🟢 Bénignes"),
            (c3, str(counts.get("M", 0)), "🔴 Malignes"),
            (c4, f"{counts.get('B',0)/len(df)*100:.1f}%", "Taux bénignité"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            fig_pie = px.pie(
                values=counts.values, names=["Bénigne (B)", "Maligne (M)"],
                title="Distribution du diagnostic",
                color_discrete_sequence=[COULEURS["secondary"], COULEURS["danger"]],
                hole=0.45
            )
            fig_pie.update_layout(**plotly_theme(), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            feat_choisie = st.selectbox(
                "Feature à visualiser",
                ["radius_mean", "texture_mean", "concavity_mean", "concave points_mean",
                 "area_mean", "perimeter_mean", "radius_worst", "concave points_worst"],
                key="cancer_feat"
            )
            df_m = df[df["diagnosis"] == "M"][feat_choisie]
            df_b = df[df["diagnosis"] == "B"][feat_choisie]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_m, name="Maligne", nbinsx=30,
                                            marker_color=COULEURS["danger"], opacity=0.7))
            fig_hist.add_trace(go.Histogram(x=df_b, name="Bénigne", nbinsx=30,
                                            marker_color=COULEURS["secondary"], opacity=0.7))
            fig_hist.update_layout(**plotly_theme(), barmode="overlay",
                                   title=f"Distribution de {feat_choisie}", height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Boxplots — top 4 features discriminantes
        st.subheader("Boxplots — Top 4 features discriminantes")
        corr_target = df[FEAT_COLS].corrwith(df["target"]).abs().sort_values(ascending=False)
        top4 = list(corr_target.head(4).index)
        box_fig = make_subplots(rows=1, cols=4, subplot_titles=top4)
        for i, feat in enumerate(top4, 1):
            box_fig.add_trace(go.Box(y=df[df["diagnosis"]=="M"][feat].values, name="Maligne",
                                     marker_color=COULEURS["danger"], showlegend=(i==1)), row=1, col=i)
            box_fig.add_trace(go.Box(y=df[df["diagnosis"]=="B"][feat].values, name="Bénigne",
                                     marker_color=COULEURS["secondary"], showlegend=(i==1)), row=1, col=i)
        box_fig.update_layout(**plotly_theme(), height=380, title="Boxplots comparatifs M vs B")
        st.plotly_chart(box_fig, use_container_width=True)

        # Heatmap corrélation top 10
        st.subheader("Corrélations avec la cible (Top 10 features)")
        corr = df[FEAT_COLS].corrwith(df["target"]).abs().sort_values(ascending=False).head(10)
        fig_corr = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker=dict(color=corr.values,
                        colorscale=[[0, COULEURS["bg"]], [1, COULEURS["primary"]]])
        ))
        fig_corr.update_layout(**plotly_theme(), title="Corrélation |r| avec la cible",
                               xaxis_title="|Corrélation|", height=320)
        st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("📋 Aperçu du dataset (10 premières lignes)"):
            st.dataframe(df.head(10), use_container_width=True)

    # ── ANALYSE BAYÉSIENNE ────────────────────────────────────────────────────
    with tab_bayes:
        st.subheader("🧮 Théorème de Bayes — Analyse Manuelle")
        st.markdown("""
        Calcul de la probabilité a posteriori **P(Maligne | radius_mean > seuil)**
        en appliquant le théorème de Bayes directement sur les données.
        """)

        df_m = df[df["diagnosis"] == "M"]
        df_b = df[df["diagnosis"] == "B"]
        P_M = len(df_m) / len(df)
        P_B = len(df_b) / len(df)

        col_s, col_res = st.columns([1, 2])
        with col_s:
            seuil = st.slider(
                "Seuil radius_mean (mm)",
                float(df["radius_mean"].min()),
                float(df["radius_mean"].max()),
                15.0, 0.1, key="bayes_seuil"
            )

        P_R_M = (df_m["radius_mean"] > seuil).mean()
        P_R_B = (df_b["radius_mean"] > seuil).mean()
        P_R   = P_R_M * P_M + P_R_B * P_B
        P_M_R = (P_R_M * P_M) / P_R if P_R > 0 else 0
        P_B_R = (P_R_B * P_B) / P_R if P_R > 0 else 0

        with col_res:
            st.markdown("**Résultats :**")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("P(Maligne)", f"{P_M:.3f}", "Prior")
            bc2.metric(f"P(radius>{seuil:.1f}|M)", f"{P_R_M:.3f}", "Vraisemblance")
            bc3.metric(f"P(M|radius>{seuil:.1f})", f"{P_M_R:.3f}", "Posterior ← Objectif")

        # Courbe de sensibilité bayésienne
        seuils = np.linspace(df["radius_mean"].min(), df["radius_mean"].max(), 200)
        post_M = []
        post_B = []
        for s in seuils:
            pRM = (df_m["radius_mean"] > s).mean()
            pRB = (df_b["radius_mean"] > s).mean()
            ev  = pRM * P_M + pRB * P_B
            if ev > 1e-9:
                post_M.append(pRM * P_M / ev)
                post_B.append(pRB * P_B / ev)
            else:
                post_M.append(None)
                post_B.append(None)

        fig_bayes = go.Figure()
        fig_bayes.add_trace(go.Scatter(
            x=seuils, y=post_M, mode="lines", name="P(Maligne | radius > seuil)",
            line=dict(color=COULEURS["danger"], width=2.5)
        ))
        fig_bayes.add_trace(go.Scatter(
            x=seuils, y=post_B, mode="lines", name="P(Bénigne | radius > seuil)",
            line=dict(color=COULEURS["secondary"], width=2.5)
        ))
        fig_bayes.add_hline(y=P_M, line_dash="dot", line_color=COULEURS["danger"],
                            annotation_text=f"Prior P(M)={P_M:.2f}")
        fig_bayes.add_hline(y=P_B, line_dash="dot", line_color=COULEURS["secondary"],
                            annotation_text=f"Prior P(B)={P_B:.2f}")
        fig_bayes.add_vline(x=seuil, line_dash="dash", line_color="#ffffff",
                            annotation_text=f"Seuil={seuil:.1f}")
        fig_bayes.update_layout(
            **plotly_theme(),
            title="Sensibilité bayésienne — Impact du seuil sur P(Maligne | radius > seuil)",
            xaxis_title="Seuil radius_mean (mm)",
            yaxis_title="Probabilité a posteriori",
            height=400, yaxis_range=[0, 1]
        )
        st.plotly_chart(fig_bayes, use_container_width=True)

        st.info(f"""
        **Interprétation :** Avec un seuil de {seuil:.1f} mm, si radius_mean > {seuil:.1f} mm alors :
        - **P(Maligne)** passe de {P_M:.3f} (prior) → **{P_M_R:.3f}** (posterior)
        - Le ratio de vraisemblance est **{(P_R_M/P_R_B):.2f}x** plus élevé pour les tumeurs malignes
        - Plus le seuil augmente → plus la probabilité de malignité augmente
        """)

    # ── MODÈLES ML ────────────────────────────────────────────────────────────
    with tab_models:
        st.subheader("Comparaison des modèles de Machine Learning")
        st.info("✅ Modèles entraînés sur 75% des données — évalués sur 25% (stratifié, random_state=42).")

        df_metrics = pd.DataFrame({
            name: {
                "Accuracy":    f"{v['accuracy']:.4f}",
                "AUC-ROC":     f"{v['auc']:.4f}",
                "F1-Score":    f"{v['f1']:.4f}",
                "Sensibilité": f"{v['sensitivity']:.4f}",
                "Spécificité": f"{v['specificity']:.4f}",
            }
            for name, v in models_results.items()
        }).T

        best_model = max(models_results, key=lambda k: models_results[k]["auc"])
        st.markdown(f"🏆 **Meilleur modèle (AUC-ROC) :** `{best_model}`")
        st.dataframe(df_metrics, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_metrics_bar(models_results), use_container_width=True)
        with col2:
            st.plotly_chart(fig_roc_curves(models_results, y_test), use_container_width=True)

        st.subheader("Matrices de Confusion")
        cols = st.columns(len(models_results))
        for col_ui, (name, res) in zip(cols, models_results.items()):
            with col_ui:
                st.plotly_chart(
                    fig_confusion_matrix(res["cm"], ("Maligne", "Bénigne"), title=name),
                    use_container_width=True
                )

        # Feature importance (Random Forest)
        if "Random Forest" in trained_models:
            st.subheader("🌲 Importance des Features — Random Forest")
            rf = trained_models["Random Forest"]
            imp_df = pd.DataFrame({
                "Feature": FEAT_COLS,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
                marker=dict(color=imp_df["Importance"],
                            colorscale=[[0, COULEURS["bg"]], [1, COULEURS["primary"]]])
            ))
            fig_imp.update_layout(**plotly_theme(), title="Top 10 Features les plus importantes",
                                   xaxis_title="Importance (Gini)", height=340)
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── SIMULATEUR ────────────────────────────────────────────────────────────
    with tab_sim:
        st.subheader("🔮 Simulateur de Prédiction Interactif")
        st.markdown("Ajustez les valeurs des features clés pour obtenir une prédiction en temps réel.")

        KEY_FEATS = [
            "radius_mean", "texture_mean", "concavity_mean", "concave points_mean",
            "area_mean", "symmetry_mean", "radius_worst", "concave points_worst"
        ]
        KEY_FEATS = [f for f in KEY_FEATS if f in FEAT_COLS]
        patient_vals = {}

        col_sliders, col_result = st.columns([2, 1])

        with col_sliders:
            st.markdown("**Paramètres du patient :**")
            for feat in KEY_FEATS:
                mn, mx, med = float(df[feat].min()), float(df[feat].max()), float(df[feat].median())
                patient_vals[feat] = st.slider(
                    feat, min_value=round(mn, 3), max_value=round(mx, 3),
                    value=round(med, 3), step=round((mx - mn) / 100, 4),
                    key=f"slider_{feat}"
                )

        with col_result:
            st.markdown("**Résultats de prédiction :**")
            input_vec    = np.array([patient_vals.get(f, float(df[f].median())) for f in FEAT_COLS]).reshape(1, -1)
            input_scaled = scaler.transform(input_vec)

            modele_choisi = st.selectbox("Modèle", list(trained_models.keys()), key="sim_model")
            model  = trained_models[modele_choisi]
            pred   = model.predict(input_scaled)[0]
            proba  = model.predict_proba(input_scaled)[0]

            label      = "🟢 BÉNIGNE" if pred == 1 else "🔴 MALIGNE"
            bg_color   = "#14432a" if pred == 1 else "#450a0a"
            text_color = "#4ade80" if pred == 1 else "#f87171"

            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {text_color}; border-radius:12px;
                        padding:20px; text-align:center; margin-top:10px;">
                <div style="font-size:2rem;">{label}</div>
                <div style="color:#8892b0; margin-top:10px;">Confiance</div>
                <div style="font-size:1.5rem; color:{text_color}; font-weight:700;">
                    {max(proba)*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={"text": "P(Bénigne) %", "font": {"color": "#c9d1d9"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": COULEURS["secondary"]},
                    "steps": [
                        {"range": [0, 40],   "color": "#450a0a"},
                        {"range": [40, 60],  "color": "#713f12"},
                        {"range": [60, 100], "color": "#14432a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
                },
                number={"suffix": "%", "font": {"color": COULEURS["secondary"]}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor=COULEURS["bg"], font={"color": "#c9d1d9"},
                height=260, margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# =============================================================================
#  PAGE 3 — BLESSURES SPORTIVES
# =============================================================================

def page_sports():
    st.markdown('<p class="section-header">🏃‍♂️ Analyse des Blessures Sportives</p>', unsafe_allow_html=True)
    st.markdown("**Dataset :** 15 420 sessions d'athlètes · capteurs multimodaux · Cible : Blessure (1) / Sain (0)")
    st.markdown("---")

    df_sports    = load_sports()
    model_result = train_sports_models()
    adv_result   = train_sports_advanced_models()

    if df_sports is None or model_result is None:
        st.error("Impossible de charger les données ou d'entraîner les modèles.")
        return

    models_results, trained_models, feat_cols, y_test, X_train, X_test, y_train = model_result

    tab_eda, tab_models, tab_adv, tab_sim = st.tabs([
        "📊 Exploration",
        "🤖 Modèles (NB / KNN / RF)",
        "🧬 Modèles Avancés",
        "🏋️ Simulateur Athlète",
    ])

    # ── EDA ──────────────────────────────────────────────────────────────────
    with tab_eda:
        st.subheader("Vue d'ensemble du dataset")
        injured = (df_sports["injury_occurred"] == 1).sum()
        not_inj = (df_sports["injury_occurred"] == 0).sum()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{len(df_sports):,}", "Sessions totales"),
            (c2, f"{injured:,}", "🔴 Blessures"),
            (c3, f"{not_inj:,}", "🟢 Sans blessure"),
            (c4, f"{injured/len(df_sports)*100:.1f}%", "Taux blessure"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig_inj = px.pie(
                values=[injured, not_inj], names=["Blessé", "Sain"],
                title="Répartition des blessures",
                color_discrete_sequence=[COULEURS["danger"], COULEURS["secondary"]], hole=0.45
            )
            fig_inj.update_layout(**plotly_theme(), height=320)
            st.plotly_chart(fig_inj, use_container_width=True)

        with col2:
            feat_vis = st.selectbox(
                "Feature à visualiser",
                [f for f in feat_cols if f in df_sports.columns],
                key="sports_feat"
            )
            d0 = df_sports[df_sports["injury_occurred"] == 0][feat_vis]
            d1 = df_sports[df_sports["injury_occurred"] == 1][feat_vis]
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=d0, name="Sain",    nbinsx=40, marker_color=COULEURS["secondary"], opacity=0.7))
            fig_dist.add_trace(go.Histogram(x=d1, name="Blessé",  nbinsx=40, marker_color=COULEURS["danger"],    opacity=0.7))
            fig_dist.update_layout(**plotly_theme(), barmode="overlay",
                                   title=f"Distribution : {feat_vis}", height=320)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Corrélations
        num_df = df_sports.select_dtypes(include="number")
        if "injury_occurred" in num_df.columns:
            corr_s = num_df.corr()["injury_occurred"].drop("injury_occurred").abs().sort_values(ascending=False).head(12)
            fig_corr = go.Figure(go.Bar(
                x=corr_s.values, y=corr_s.index, orientation="h",
                marker=dict(color=corr_s.values,
                            colorscale=[[0, COULEURS["bg"]], [1, COULEURS["secondary"]]])
            ))
            fig_corr.update_layout(**plotly_theme(), title="Top 12 features corrélées à la blessure", height=380)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Bayes manual
        st.subheader("🧮 Analyse Bayésienne — P(Blessure | fatigue_index > seuil)")
        if "fatigue_index" in df_sports.columns:
            threshold = st.slider(
                "Seuil fatigue_index", float(df_sports["fatigue_index"].min()),
                float(df_sports["fatigue_index"].max()),
                float(df_sports["fatigue_index"].mean()), 0.01, key="sports_bayes_thresh"
            )
            df_inj = df_sports[df_sports["injury_occurred"] == 1]
            df_ni  = df_sports[df_sports["injury_occurred"] == 0]
            P_i  = len(df_inj) / len(df_sports)
            P_ni = len(df_ni)  / len(df_sports)
            P_F_I  = (df_inj["fatigue_index"] > threshold).mean()
            P_F_NI = (df_ni["fatigue_index"]  > threshold).mean()
            P_F    = P_F_I * P_i + P_F_NI * P_ni
            P_I_F  = (P_F_I * P_i) / P_F if P_F > 0 else 0

            bb1, bb2, bb3 = st.columns(3)
            bb1.metric("P(Blessure)", f"{P_i:.3f}", "Prior")
            bb2.metric(f"P(fatigue>{threshold:.2f}|Blessure)", f"{P_F_I:.3f}", "Vraisemblance")
            bb3.metric(f"P(Blessure|fatigue>{threshold:.2f})", f"{P_I_F:.3f}", "Posterior")

            st.info(f"Si fatigue_index > {threshold:.2f}, la probabilité de blessure passe de {P_i*100:.1f}% → **{P_I_F*100:.1f}%**")

    # ── MODÈLES BASE ─────────────────────────────────────────────────────────
    with tab_models:
        st.subheader("Modèles de Classification + Métriques Avancées")
        st.info("⚖️ SMOTE appliqué sur le train set · Features sélectionnées par corrélation.")

        df_met = pd.DataFrame({
            name: {
                "Accuracy":    f"{v['accuracy']:.4f}",
                "AUC-ROC":     f"{v['auc']:.4f}",
                "F1-Score":    f"{v['f1']:.4f}",
                "Dice Score":  f"{v['dice']:.4f}",
                "Sensibilité": f"{v['sensitivity']:.4f}",
                "Spécificité": f"{v['specificity']:.4f}",
            }
            for name, v in models_results.items()
        }).T

        best_m = max(models_results, key=lambda k: models_results[k]["auc"])
        st.markdown(f"🏆 **Meilleur modèle (AUC-ROC) :** `{best_m}`")
        st.dataframe(df_met, use_container_width=True)

        with st.expander("ℹ️ Qu'est-ce que le Dice Score ?"):
            st.markdown("""
            Le **Dice Score** (coefficient de Sørensen-Dice) mesure le chevauchement :

            $$\\text{Dice} = \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}$$

            Équivalent au **F1-Score** en classification binaire. **1.0** = prédiction parfaite.
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_metrics_bar(models_results), use_container_width=True)
        with col2:
            st.plotly_chart(fig_roc_curves(models_results, y_test), use_container_width=True)

        st.subheader("Matrices de Confusion")
        cols = st.columns(len(models_results))
        for col_ui, (name, res) in zip(cols, models_results.items()):
            with col_ui:
                st.plotly_chart(
                    fig_confusion_matrix(res["cm"], ("Blessé", "Sain"), title=name),
                    use_container_width=True
                )

        # nDCG
        st.subheader("📈 Discounted Cumulative Gain (nDCG)")
        with st.expander("Calcul du nDCG @ k"):
            def compute_ndcg(y_true, y_scores, k=50):
                order = np.argsort(-y_scores)[:k]
                gains = y_true.values[order] if hasattr(y_true, "values") else y_true[order]
                disc  = np.log2(np.arange(2, len(gains) + 2))
                dcg   = np.sum(gains / disc)
                ideal = np.sort(gains)[::-1]
                idcg  = np.sum(ideal / disc[:len(ideal)])
                return dcg / idcg if idcg > 0 else 0

            k_val = st.slider("Valeur de k", 10, 200, 50, key="ndcg_k")
            ndcg_rows = []
            for name, res in models_results.items():
                proba_1d = res["y_proba"]
                ndcg = compute_ndcg(y_test, proba_1d, k=k_val)
                ndcg_rows.append({"Modèle": name, f"nDCG@{k_val}": f"{ndcg:.4f}"})
            st.dataframe(pd.DataFrame(ndcg_rows), use_container_width=True)

    # ── MODÈLES AVANCÉS ───────────────────────────────────────────────────────
    with tab_adv:
        st.subheader("🧬 Modèles Avancés du Notebook")

        if adv_result is None:
            st.warning("Les modèles avancés n'ont pas pu être chargés.")
        else:
            adv_results, y_test_adv = adv_result

            # ── Tableau métriques avancées ────────────────────────────────────
            st.markdown("#### 📊 Tableau des métriques avancées")
            adv_metric_rows = []
            for name, res in adv_results.items():
                adv_metric_rows.append({
                    "Modèle":       name,
                    "Accuracy":     f"{res['accuracy']:.4f}",
                    "F1-Score":     f"{res['f1']:.4f}",
                    "Recall":       f"{res['recall']:.4f}",
                    "AUC-ROC":      f"{res['auc']:.4f}",
                    "Dice Score":   f"{res['dice']:.4f}",
                    "Log Loss":     f"{res['log_loss']:.4f}",
                    "RMSLE":        f"{res['rmsle']:.4f}",
                    "Kappa (κ)":    f"{res['kappa']:.4f}",
                    "MCC":          f"{res['mcc']:.4f}",
                })
            st.dataframe(pd.DataFrame(adv_metric_rows).set_index("Modèle"), use_container_width=True)

            with st.expander("ℹ️ Description des métriques avancées"):
                st.markdown("""
                | Métrique | Description |
                |----------|-------------|
                | **Recall** | TP / (TP + FN) — Taux de détection des blessures réelles |
                | **Log Loss** | Pénalise les prédictions de probabilité incorrectes |
                | **RMSLE** | Root Mean Squared Log Error — robuste aux outliers |
                | **Kappa (κ)** | Accord au-delà du hasard (0=aléatoire, 1=parfait) |
                | **MCC** | Matthews Correlation Coefficient — métrique équilibrée |
                | **FPR** | False Positive Rate = FP / (FP + TN) |
                """)

            # ── ROC multi-modèles avancés ────────────────────────────────────
            st.markdown("#### 📈 Courbes ROC — Modèles avancés")
            fig_roc_adv = go.Figure()
            palette = [COULEURS["primary"], COULEURS["secondary"],
                       COULEURS["warning"], COULEURS["purple"]]
            fig_roc_adv.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                   line=dict(dash="dash", color="#4b5563"))
            from sklearn.metrics import roc_curve as roc_curve_fn
            for i, (name, res) in enumerate(adv_results.items()):
                proba = res["y_proba"][:, 1]
                fpr_v, tpr_v, _ = roc_curve_fn(y_test_adv, proba)
                fig_roc_adv.add_trace(go.Scatter(
                    x=fpr_v, y=tpr_v, mode="lines",
                    name=f"{name} (AUC={res['auc']:.3f})",
                    line=dict(color=palette[i % len(palette)], width=2.5)
                ))
            fig_roc_adv.update_layout(**plotly_theme(),
                                       title="Courbes ROC — Modèles avancés",
                                       xaxis_title="FPR", yaxis_title="TPR", height=400)
            st.plotly_chart(fig_roc_adv, use_container_width=True)

            # ── Matrices de confusion ────────────────────────────────────────
            st.markdown("#### 🔲 Matrices de Confusion")
            cols_cm = st.columns(len(adv_results))
            for col_ui, (name, res) in zip(cols_cm, adv_results.items()):
                with col_ui:
                    st.plotly_chart(
                        fig_confusion_matrix(res["cm"], ("Blessé", "Sain"), title=name),
                        use_container_width=True
                    )

            # ── Section par modèle ───────────────────────────────────────────
            st.markdown("---")

            # Decision Tree
            with st.expander("🌳 Decision Tree — Détails"):
                dt_res = adv_results["Decision Tree"]
                st.markdown(f"""
                **Accuracy :** {dt_res['accuracy']:.4f} | **AUC :** {dt_res['auc']:.4f} |
                **F1 :** {dt_res['f1']:.4f} | **Recall :** {dt_res['recall']:.4f}

                *Profondeur max = 8, entraîné sur l'ensemble complet des features (post-SMOTE).*
                """)
                if "feat_names" in dt_res and dt_res.get("model"):
                    dt_model = dt_res["model"]
                    feat_names = dt_res["feat_names"]
                    imp = pd.DataFrame({
                        "Feature": feat_names,
                        "Importance": dt_model.feature_importances_
                    }).sort_values("Importance", ascending=False).head(10)
                    fig_dt_imp = go.Figure(go.Bar(
                        x=imp["Importance"], y=imp["Feature"], orientation="h",
                        marker_color=COULEURS["primary"]
                    ))
                    fig_dt_imp.update_layout(**plotly_theme(), title="Top 10 features DT", height=320)
                    st.plotly_chart(fig_dt_imp, use_container_width=True)

            # Marbar-RF
            with st.expander("🔁 Marbar-RF — Matrice de Transition Markovienne"):
                mb_res = adv_results["Marbar-RF"]
                st.markdown("""
                **Principe :** Random Forest classique, puis les probabilités finales sont pondérées
                par une **matrice de transition Markovienne** construite à partir des prédictions
                consécutives inter-arbres. Cela capture les dépendances entre arbres.

                $$P_{marbar} = P_{RF} \\cdot T_{Markov}$$
                """)
                if "transition_matrix" in mb_res:
                    trans = mb_res["transition_matrix"]
                    trans_df = pd.DataFrame(trans,
                                            index=["État: 0 (Sain)", "État: 1 (Blessé)"],
                                            columns=["→ 0 (Sain)", "→ 1 (Blessé)"])
                    st.markdown("**Matrice de Transition Markovienne :**")
                    fig_trans = px.imshow(
                        trans, text_auto=".3f",
                        labels=dict(x="État suivant", y="État courant"),
                        x=["→ Sain (0)", "→ Blessé (1)"],
                        y=["Sain (0)", "Blessé (1)"],
                        color_continuous_scale="Blues",
                        title="Matrice de Transition Markov entre arbres RF"
                    )
                    fig_trans.update_layout(**plotly_theme(), height=300)
                    st.plotly_chart(fig_trans, use_container_width=True)
                    st.dataframe(trans_df.round(4), use_container_width=True)

            # Hybrid
            with st.expander("🔬 Modèle Hybride — Weibull + Benford + RF + NB"):
                hyb_res = adv_results["Hybrid (Weibull+Benford)"]
                c_w = hyb_res.get("weibull_shape", 0)
                st.markdown(f"""
                **Architecture :** Ajout de 3 features probabilistes aux données originales :
                - `weibull_fatigue` — densité Weibull(c={c_w:.2f}) sur `fatigue_index`
                - `benford_training` — conformité Benford sur `training_load`
                - `weibull_recovery` — densité Weibull sur `recovery_score`

                **Fusion probabiliste :** 60% RF + 40% Naive Bayes

                **Accuracy :** {hyb_res['accuracy']:.4f} | **AUC :** {hyb_res['auc']:.4f} | **F1 :** {hyb_res['f1']:.4f}
                """)

                # Weibull distribution visualization
                if df_sports is not None and "fatigue_index" in df_sports.columns:
                    from scipy.stats import weibull_min
                    fatigue_data = df_sports["fatigue_index"].values
                    fatigue_sh   = fatigue_data - fatigue_data.min() + 0.01
                    c_est, _, sc_est = weibull_min.fit(fatigue_sh, floc=0)
                    x_w = np.linspace(fatigue_sh.min(), fatigue_sh.max(), 300)
                    pdf_w = weibull_min.pdf(x_w, c_est, scale=sc_est)

                    fig_weibull = go.Figure()
                    fig_weibull.add_trace(go.Histogram(
                        x=fatigue_sh, name="Données fatigue_index",
                        histnorm="probability density", nbinsx=50,
                        marker_color=COULEURS["primary"], opacity=0.6
                    ))
                    fig_weibull.add_trace(go.Scatter(
                        x=x_w, y=pdf_w, mode="lines", name=f"Weibull(c={c_est:.2f}, η={sc_est:.2f})",
                        line=dict(color=COULEURS["danger"], width=2.5)
                    ))
                    fig_weibull.update_layout(**plotly_theme(),
                                              title="Ajustement Loi de Weibull — fatigue_index",
                                              xaxis_title="fatigue_index (décalé)", height=320)
                    st.plotly_chart(fig_weibull, use_container_width=True)

            # GF-MFTL
            with st.expander("⚡ GF-MFTL — Gradient-Free Mean-Field-Type Learning"):
                gf_res = adv_results["GF-MFTL"]
                st.markdown("""
                **Principe :** Classifieur linéaire (sigmoïde) optimisé par un **essaim de particules**
                sans jamais calculer de gradient. À chaque itération, les particules convergent
                vers le meilleur demi-essaim :

                $$w_i(t+1) = w_i(t) + \\sigma \\cdot (\\bar{w}_{best} - w_i(t)) + \\sigma \\cdot 0.3 \\cdot \\mathcal{N}(0, I)$$

                **Avantage :** robuste aux surfaces de perte non-convexes, pas de divergence de gradient.
                """)

                if "convergence" in gf_res and gf_res["convergence"]:
                    history = np.array(gf_res["convergence"])
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=list(range(len(history))), y=history,
                        mode="lines", fill="tozeroy",
                        name="Log Loss (meilleure particule)",
                        line=dict(color=COULEURS["purple"], width=2.5),
                        fillcolor="rgba(192,132,252,0.15)"
                    ))
                    fig_conv.update_layout(
                        **plotly_theme(),
                        title="Convergence GF-MFTL — Log Loss par itération",
                        xaxis_title="Itération (Mean-Field)",
                        yaxis_title="Log Loss",
                        height=320
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)

                st.markdown(f"""
                **Résultats :** Accuracy={gf_res['accuracy']:.4f} | AUC={gf_res['auc']:.4f} |
                F1={gf_res['f1']:.4f} | Kappa={gf_res['kappa']:.4f}
                """)

            # Bar chart comparatif avancé
            st.markdown("#### 📊 Comparaison visuelle — Modèles avancés")
            adv_names  = list(adv_results.keys())
            adv_accs   = [adv_results[n]["accuracy"] for n in adv_names]
            adv_aucs   = [adv_results[n]["auc"]      for n in adv_names]
            adv_f1s    = [adv_results[n]["f1"]       for n in adv_names]
            adv_kapps  = [adv_results[n]["kappa"]    for n in adv_names]

            fig_adv_bar = go.Figure()
            fig_adv_bar.add_trace(go.Bar(name="Accuracy", x=adv_names, y=adv_accs, marker_color=COULEURS["primary"]))
            fig_adv_bar.add_trace(go.Bar(name="AUC-ROC",  x=adv_names, y=adv_aucs, marker_color=COULEURS["secondary"]))
            fig_adv_bar.add_trace(go.Bar(name="F1-Score", x=adv_names, y=adv_f1s,  marker_color=COULEURS["danger"]))
            fig_adv_bar.add_trace(go.Bar(name="Kappa",    x=adv_names, y=adv_kapps, marker_color=COULEURS["purple"]))
            fig_adv_bar.update_layout(
                **plotly_theme(), barmode="group",
                title="Comparaison Accuracy / AUC / F1 / Kappa — Modèles avancés",
                height=380, yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig_adv_bar, use_container_width=True)

    # ── SIMULATEUR ────────────────────────────────────────────────────────────
    with tab_sim:
        st.subheader("🏋️ Simulateur de Profil d'Athlète")
        st.markdown("Définissez le profil biologique d'un athlète pour estimer son risque de blessure.")

        modele_choisi = st.selectbox("Choisir le modèle", list(trained_models.keys()), key="sport_sim_model")
        model_s = trained_models[modele_choisi]

        col_a, col_b, col_c = st.columns(3)
        inputs = {}

        feat_labels = {
            "fatigue_recovery_ratio": "Ratio Fatigue/Récup.",
            "recovery_score":         "Score de récupération",
            "fatigue_index":          "Index de fatigue",
            "training_load":          "Charge d'entraînement",
            "sleep_quality":          "Qualité du sommeil",
            "stress_level":           "Niveau de stress",
            "sleep_stress_index":     "Index Sommeil/Stress",
            "heart_rate":             "Fréquence cardiaque",
            "training_intensity":     "Intensité entraînement",
        }

        feat_slider_map = {
            "fatigue_recovery_ratio": (0.1, 5.0, 1.0, 0.1),
            "recovery_score":         (0.0, 10.0, 5.0, 0.1),
            "fatigue_index":          (0.0, 10.0, 5.0, 0.1),
            "training_load":          (50.0, 500.0, 200.0, 5.0),
            "sleep_quality":          (1.0, 10.0, 6.0, 0.1),
            "stress_level":           (0.0, 10.0, 5.0, 0.1),
            "sleep_stress_index":     (0.1, 5.0, 1.0, 0.05),
            "heart_rate":             (50.0, 200.0, 120.0, 1.0),
            "training_intensity":     (0.1, 10.0, 5.0, 0.1),
        }

        for i, feat in enumerate(feat_cols):
            target_col = [col_a, col_b, col_c][i % 3]
            rng = feat_slider_map.get(feat, (0.0, 10.0, 5.0, 0.1))
            with target_col:
                label = feat_labels.get(feat, feat)
                inputs[feat] = st.slider(
                    label,
                    min_value=float(rng[0]), max_value=float(rng[1]),
                    value=float(rng[2]), step=float(rng[3]),
                    key=f"sport_{feat}"
                )

        st.markdown("---")
        col_res, col_gauge = st.columns([1, 1])

        input_arr = np.array([[inputs[f] for f in feat_cols]])
        pred_s    = model_s.predict(input_arr)[0]
        proba_s   = model_s.predict_proba(input_arr)[0]
        risk_pct  = proba_s[1] * 100

        with col_res:
            if pred_s == 1:
                st.error(f"⚠️ **RISQUE DE BLESSURE ÉLEVÉ** — Confiance : {risk_pct:.1f}%")
                conseil = "Réduire la charge d'entraînement. Priorité : récupération et sommeil."
            else:
                st.success(f"✅ **FAIBLE RISQUE DE BLESSURE** — Confiance : {100-risk_pct:.1f}%")
                conseil = "Profil favorable. Maintenir l'équilibre charge/récupération."
            st.info(f"💡 **Conseil :** {conseil}")

            categories = ["Fatigue", "Récupération", "Sommeil", "Stress", "Intensité"]
            vals_radar  = [
                inputs.get("fatigue_index", 5) / 10,
                inputs.get("recovery_score", 5) / 10,
                inputs.get("sleep_quality", 5) / 10,
                1 - inputs.get("stress_level", 5) / 10,
                inputs.get("training_intensity", 5) / 10,
            ]
            vals_radar.append(vals_radar[0])
            categories.append(categories[0])

            fig_radar = go.Figure(go.Scatterpolar(
                r=vals_radar, theta=categories, fill="toself",
                fillcolor="rgba(124,131,245,0.2)",
                line=dict(color=COULEURS["primary"], width=2)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor=COULEURS["bg"],
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=COULEURS["grid"]),
                    angularaxis=dict(gridcolor=COULEURS["grid"])
                ),
                paper_bgcolor=COULEURS["bg"], font=dict(color="#c9d1d9"),
                title="Profil de l'athlète", height=300, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                title={"text": "Risque de Blessure (%)", "font": {"color": "#c9d1d9"}},
                delta={"reference": 50, "increasing": {"color": COULEURS["danger"]}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": COULEURS["danger"] if risk_pct > 50 else COULEURS["secondary"]},
                    "steps": [
                        {"range": [0, 33],   "color": "#14432a"},
                        {"range": [33, 66],  "color": "#713f12"},
                        {"range": [66, 100], "color": "#450a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
                },
                number={"suffix": "%", "font": {"color": "#c9d1d9"}}
            ))
            fig_g.update_layout(
                paper_bgcolor=COULEURS["bg"], font={"color": "#c9d1d9"},
                height=350, margin=dict(l=20, r=20, t=60, b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)


# =============================================================================
#  PAGE 4 — SIMULATIONS STATISTIQUES
# =============================================================================

def page_simulations():
    st.markdown('<p class="section-header">🎲 Simulations Statistiques</p>', unsafe_allow_html=True)
    st.markdown("Comparaison interactive de lois de probabilité discrètes et continues — **N = 10 000 tirages**.")
    st.markdown("---")

    N = 10_000
    np.random.seed(42)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔵 Bernoulli / Exponentielle",
        "📊 Binomiale / Normale",
        "⭐ Poisson / Gamma",
        "🔷 Géométrique / Beta",
        "🎯 Uniforme Discrète / Continue",
    ])

    def verification_table(stats_rows):
        """Affiche un tableau comparatif empirique vs théorique."""
        df_v = pd.DataFrame(stats_rows)
        st.markdown("**📋 Vérification empirique vs théorique :**")
        st.dataframe(df_v.style.format({
            "Empirique": "{:.4f}", "Théorique": "{:.4f}", "Écart abs.": "{:.5f}", "Écart %": "{:.3f}%"
        }), use_container_width=True)

    # ── TAB 1: Bernoulli / Exponentielle ────────────────────────────────────
    with tab1:
        st.subheader("Bernoulli (discrète) vs Exponentielle (continue)")
        st.markdown("""
        - **Bernoulli** $B(p)$ : essai binaire — succès avec proba $p$, espérance $E[X]=p$
        - **Exponentielle** $\\mathcal{E}(\\lambda)$ : temps avant un événement, $E[X]=1/\\lambda$
        - **Lien :** Si les événements suivent un processus de Poisson(λ), le temps entre deux suit Exp(λ)
        """)

        col_p, col_l = st.columns(2)
        with col_p:
            p_b  = st.slider("p (Bernoulli)", 0.01, 0.99, 0.30, 0.01, key="p_b")
        with col_l:
            lam = st.slider("λ (Exponentielle)", 0.1, 10.0, 2.0, 0.1, key="lam_exp")

        bern_s = np.random.binomial(1, p_b, N)
        exp_s  = np.random.exponential(1.0 / lam, N)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Bernoulli — Fréquences observées", "Exponentielle — Histogramme"])
        fig.add_trace(go.Bar(x=[0, 1], y=[(bern_s == 0).sum() / N, (bern_s == 1).sum() / N],
                             name="Empirique", marker_color=COULEURS["primary"]), row=1, col=1)
        fig.add_trace(go.Bar(x=[0, 1], y=[1 - p_b, p_b],
                             name="Théorique", marker_color=COULEURS["warning"], opacity=0.6), row=1, col=1)

        from scipy.stats import expon
        x_exp = np.linspace(0, exp_s.max(), 300)
        fig.add_trace(go.Histogram(x=exp_s, nbinsx=60, histnorm="probability density",
                                   name="Empirique", marker_color=COULEURS["secondary"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_exp, y=expon.pdf(x_exp, scale=1/lam),
                                 mode="lines", name="Théorique", line=dict(color=COULEURS["danger"], width=2.5)), row=1, col=2)
        fig.update_layout(**plotly_theme(), height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        verification_table([
            {"Loi": "Bernoulli",     "Stat": "Moyenne",  "Empirique": bern_s.mean(),         "Théorique": p_b,         "Écart abs.": abs(bern_s.mean()-p_b),         "Écart %": abs(bern_s.mean()-p_b)/p_b*100},
            {"Loi": "Bernoulli",     "Stat": "Variance", "Empirique": bern_s.var(),           "Théorique": p_b*(1-p_b), "Écart abs.": abs(bern_s.var()-p_b*(1-p_b)), "Écart %": abs(bern_s.var()-p_b*(1-p_b))/(p_b*(1-p_b))*100},
            {"Loi": "Exponentielle", "Stat": "Moyenne",  "Empirique": exp_s.mean(),           "Théorique": 1/lam,       "Écart abs.": abs(exp_s.mean()-1/lam),        "Écart %": abs(exp_s.mean()-1/lam)/(1/lam)*100},
            {"Loi": "Exponentielle", "Stat": "Variance", "Empirique": exp_s.var(),            "Théorique": 1/lam**2,    "Écart abs.": abs(exp_s.var()-1/lam**2),      "Écart %": abs(exp_s.var()-1/lam**2)/(1/lam**2)*100},
        ])

    # ── TAB 2: Binomiale / Normale ───────────────────────────────────────────
    with tab2:
        st.subheader("Binomiale (discrète) → Approximation Normale (continue)")
        st.markdown("""
        - **Binomiale** $B(n, p)$ : nombre de succès en $n$ essais, $E[X]=np$, $Var=np(1-p)$
        - **Normale** $\\mathcal{N}(\\mu, \\sigma^2)$ : distribution en cloche symétrique
        - **Théorème Central Limite :** $B(n,p) \\approx N(np, np(1-p))$ quand $n$ est grand
        """)

        col_n, col_pp = st.columns(2)
        with col_n:
            n_bin = st.slider("n (Binomiale)", 10, 100, 50, 1, key="n_bin")
        with col_pp:
            p_bin = st.slider("p (Binomiale)", 0.1, 0.9, 0.4, 0.01, key="p_bin")

        mu_n   = n_bin * p_bin
        sig_n  = np.sqrt(n_bin * p_bin * (1 - p_bin))
        binom_s = np.random.binomial(n_bin, p_bin, N)
        norm_s  = np.random.normal(mu_n, sig_n, N)

        from scipy.stats import norm, binom as sp_binom

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Binomiale — Distribution", "Normale — Densité de probabilité"])
        x_b = np.arange(0, n_bin + 1)
        fig.add_trace(go.Bar(x=x_b, y=sp_binom.pmf(x_b, n_bin, p_bin),
                             name="Théorique B(n,p)", marker_color=COULEURS["primary"], opacity=0.8), row=1, col=1)

        x_n = np.linspace(norm_s.min(), norm_s.max(), 300)
        fig.add_trace(go.Histogram(x=norm_s, nbinsx=60, histnorm="probability density",
                                   name="N(μ,σ²) empirique", marker_color=COULEURS["secondary"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_n, y=norm.pdf(x_n, mu_n, sig_n),
                                 mode="lines", name="N(μ,σ²) théorique",
                                 line=dict(color=COULEURS["danger"], width=2.5)), row=1, col=2)
        fig.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"**TCL :** B({n_bin}, {p_bin}) ≈ N(μ={mu_n:.1f}, σ²={sig_n**2:.2f}) "
                f"→ n·p·(1-p) = {sig_n**2:.2f} ≥ 9 {'✅' if sig_n**2 >= 9 else '⚠️'}")

        verification_table([
            {"Loi": "Binomiale", "Stat": "Moyenne",  "Empirique": binom_s.mean(), "Théorique": mu_n,       "Écart abs.": abs(binom_s.mean()-mu_n),       "Écart %": abs(binom_s.mean()-mu_n)/mu_n*100},
            {"Loi": "Binomiale", "Stat": "Variance", "Empirique": binom_s.var(),  "Théorique": sig_n**2,   "Écart abs.": abs(binom_s.var()-sig_n**2),    "Écart %": abs(binom_s.var()-sig_n**2)/sig_n**2*100},
            {"Loi": "Normale",   "Stat": "Moyenne",  "Empirique": norm_s.mean(),  "Théorique": mu_n,       "Écart abs.": abs(norm_s.mean()-mu_n),        "Écart %": abs(norm_s.mean()-mu_n)/mu_n*100},
            {"Loi": "Normale",   "Stat": "Variance", "Empirique": norm_s.var(),   "Théorique": sig_n**2,   "Écart abs.": abs(norm_s.var()-sig_n**2),     "Écart %": abs(norm_s.var()-sig_n**2)/sig_n**2*100},
        ])

    # ── TAB 3: Poisson / Gamma ───────────────────────────────────────────────
    with tab3:
        st.subheader("Poisson (discrète) vs Gamma (continue)")
        st.markdown("""
        - **Poisson** $\\mathcal{P}(\\lambda)$ : nombre d'événements dans un temps fixe, $E[X]=\\lambda$
        - **Gamma** $\\Gamma(\\alpha, \\beta)$ : temps d'attente jusqu'au $k$-ième événement, $E[X]=\\alpha/\\beta$
        - **Lien :** Poisson COMPTE les événements — Gamma MESURE le temps jusqu'au k-ième
        """)

        col_lp, col_ag, col_bg = st.columns(3)
        with col_lp:
            lam_p = st.slider("λ (Poisson)", 0.5, 20.0, 5.0, 0.5, key="lam_pois")
        with col_ag:
            alpha = st.slider("α — forme (Gamma)", 1.0, 15.0, 5.0, 0.5, key="alpha_gam")
        with col_bg:
            beta  = st.slider("β — taux (Gamma)", 0.1, 5.0, 1.0, 0.1, key="beta_gam")

        pois_s = np.random.poisson(lam_p, N)
        gam_s  = np.random.gamma(alpha, 1.0 / beta, N)

        from scipy.stats import poisson, gamma

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Poisson(λ={lam_p})", f"Gamma(α={alpha}, β={beta})"])
        x_p = np.arange(0, pois_s.max() + 1)
        fig.add_trace(go.Bar(x=x_p, y=poisson.pmf(x_p, lam_p),
                             name="Poisson théorique", marker_color=COULEURS["primary"]), row=1, col=1)

        x_g = np.linspace(0.01, gam_s.max(), 300)
        fig.add_trace(go.Histogram(x=gam_s, nbinsx=60, histnorm="probability density",
                                   name="Gamma empirique", marker_color=COULEURS["secondary"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_g, y=gamma.pdf(x_g, a=alpha, scale=1/beta),
                                 mode="lines", name="Gamma théorique",
                                 line=dict(color=COULEURS["danger"], width=2.5)), row=1, col=2)
        fig.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig, use_container_width=True)

        verification_table([
            {"Loi": "Poisson", "Stat": "Moyenne",  "Empirique": pois_s.mean(), "Théorique": lam_p,          "Écart abs.": abs(pois_s.mean()-lam_p),       "Écart %": abs(pois_s.mean()-lam_p)/lam_p*100},
            {"Loi": "Poisson", "Stat": "Variance", "Empirique": pois_s.var(),  "Théorique": lam_p,          "Écart abs.": abs(pois_s.var()-lam_p),        "Écart %": abs(pois_s.var()-lam_p)/lam_p*100},
            {"Loi": "Gamma",   "Stat": "Moyenne",  "Empirique": gam_s.mean(),  "Théorique": alpha/beta,     "Écart abs.": abs(gam_s.mean()-alpha/beta),   "Écart %": abs(gam_s.mean()-alpha/beta)/(alpha/beta)*100},
            {"Loi": "Gamma",   "Stat": "Variance", "Empirique": gam_s.var(),   "Théorique": alpha/beta**2,  "Écart abs.": abs(gam_s.var()-alpha/beta**2), "Écart %": abs(gam_s.var()-alpha/beta**2)/(alpha/beta**2)*100},
        ])

    # ── TAB 4: Géométrique / Beta ────────────────────────────────────────────
    with tab4:
        st.subheader("Géométrique (discrète) vs Beta (continue)")
        st.markdown("""
        - **Géométrique** $G(p)$ : nombre d'essais jusqu'au premier succès, $E[X]=1/p$
        - **Beta** $\\mathcal{B}(\\alpha, \\beta)$ : modélise l'incertitude sur une probabilité $p \\in [0,1]$
        - **Lien bayésien :** Prior Beta + données Bernoulli → Posterior Beta (conjuguée naturelle)
        """)

        col_pg, col_ab, col_bb = st.columns(3)
        with col_pg:
            p_geo  = st.slider("p (Géométrique)", 0.05, 0.9, 0.30, 0.05, key="p_geo")
        with col_ab:
            a_beta = st.slider("α (Beta)", 0.5, 10.0, 3.0, 0.5, key="a_beta")
        with col_bb:
            b_beta = st.slider("β (Beta)", 0.5, 10.0, 7.0, 0.5, key="b_beta")

        geo_s  = np.random.geometric(p_geo, N)
        beta_s = np.random.beta(a_beta, b_beta, N)

        from scipy.stats import geom, beta as sp_beta

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Géométrique(p={p_geo})", f"Beta(α={a_beta}, β={b_beta})"])
        k_max = min(int(geo_s.max()), 30)
        x_geo = np.arange(1, k_max + 1)
        fig.add_trace(go.Bar(x=x_geo, y=geom.pmf(x_geo, p_geo),
                             name="Géom. théorique", marker_color=COULEURS["primary"]), row=1, col=1)

        x_bt = np.linspace(0.001, 0.999, 300)
        fig.add_trace(go.Histogram(x=beta_s, nbinsx=60, histnorm="probability density",
                                   name="Beta empirique", marker_color=COULEURS["secondary"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_bt, y=sp_beta.pdf(x_bt, a_beta, b_beta),
                                 mode="lines", name="Beta théorique",
                                 line=dict(color=COULEURS["danger"], width=2.5)), row=1, col=2)
        fig.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig, use_container_width=True)

        geo_var_theo  = (1 - p_geo) / p_geo**2
        beta_mu_theo  = a_beta / (a_beta + b_beta)
        beta_var_theo = a_beta * b_beta / ((a_beta + b_beta)**2 * (a_beta + b_beta + 1))

        verification_table([
            {"Loi": "Géométrique", "Stat": "Moyenne",  "Empirique": geo_s.mean(),   "Théorique": 1/p_geo,       "Écart abs.": abs(geo_s.mean()-1/p_geo),         "Écart %": abs(geo_s.mean()-1/p_geo)/(1/p_geo)*100},
            {"Loi": "Géométrique", "Stat": "Variance", "Empirique": geo_s.var(),    "Théorique": geo_var_theo,  "Écart abs.": abs(geo_s.var()-geo_var_theo),      "Écart %": abs(geo_s.var()-geo_var_theo)/geo_var_theo*100},
            {"Loi": "Beta",        "Stat": "Moyenne",  "Empirique": beta_s.mean(),  "Théorique": beta_mu_theo,  "Écart abs.": abs(beta_s.mean()-beta_mu_theo),    "Écart %": abs(beta_s.mean()-beta_mu_theo)/beta_mu_theo*100},
            {"Loi": "Beta",        "Stat": "Variance", "Empirique": beta_s.var(),   "Théorique": beta_var_theo, "Écart abs.": abs(beta_s.var()-beta_var_theo),    "Écart %": abs(beta_s.var()-beta_var_theo)/beta_var_theo*100},
        ])

    # ── TAB 5: Uniforme Discrète / Continue ──────────────────────────────────
    with tab5:
        st.subheader("Uniforme Discrète (dé) vs Uniforme Continue")
        st.markdown("""
        - **Uniforme Discrète** $U\\{a,b\\}$ : chaque entier équiprobable, $E[X]=(a+b)/2$
        - **Uniforme Continue** $U[a,b]$ : densité constante $1/(b-a)$ sur $[a,b]$
        - **Lien :** même philosophie d'équiprobabilité — la version continue est la limite discrète avec pas → 0
        """)

        col_d, col_cc = st.columns(2)
        with col_d:
            n_faces = st.slider("Nombre de faces du dé", 2, 20, 6, 1, key="faces")
        with col_cc:
            a_c = st.slider("a (Uniforme Continue)", 0.0, 5.0, 0.0, 0.1, key="a_cont")
            b_c = st.slider("b (Uniforme Continue)", 0.1, 10.0, 1.0, 0.1, key="b_cont")

        if b_c <= a_c:
            st.warning("⚠️ b doit être > a")
            b_c = a_c + 0.1

        disc_s = np.random.randint(1, n_faces + 1, N)
        cont_s = np.random.uniform(a_c, b_c, N)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Uniforme Discrète — Dé à {n_faces} faces",
                                            f"Uniforme Continue [{a_c:.1f}, {b_c:.1f}]"])

        faces     = np.arange(1, n_faces + 1)
        theo_freq = np.full(n_faces, 1 / n_faces)
        emp_freq  = np.array([(disc_s == k).mean() for k in faces])

        fig.add_trace(go.Bar(x=faces, y=emp_freq, name="Empirique",
                             marker_color=COULEURS["primary"], opacity=0.8), row=1, col=1)
        fig.add_trace(go.Scatter(x=faces, y=theo_freq, mode="markers+lines",
                                 name="Théorique 1/n",
                                 marker=dict(color=COULEURS["danger"], size=10),
                                 line=dict(dash="dash", color=COULEURS["danger"])), row=1, col=1)

        x_uc = np.linspace(a_c - 0.1, b_c + 0.1, 300)
        fig.add_trace(go.Histogram(x=cont_s, nbinsx=50, histnorm="probability density",
                                   name="Cont. empirique", marker_color=COULEURS["secondary"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[a_c, a_c, b_c, b_c], y=[0, 1/(b_c-a_c), 1/(b_c-a_c), 0],
            mode="lines", name="Cont. théorique",
            line=dict(color=COULEURS["danger"], width=2.5)
        ), row=1, col=2)
        fig.update_layout(**plotly_theme(), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Test χ²
        from scipy.stats import chisquare
        observed = np.array([(disc_s == k).sum() for k in faces])
        expected = np.full(n_faces, N / n_faces)
        chi2_stat, chi2_p = chisquare(observed, expected)

        disc_mu_theo  = (n_faces + 1) / 2
        disc_var_theo = (n_faces**2 - 1) / 12
        cont_mu_theo  = (a_c + b_c) / 2
        cont_var_theo = (b_c - a_c)**2 / 12

        col_a, col_b, col_cc2 = st.columns(3)
        col_a.metric("χ² stat (adéquation dé)", f"{chi2_stat:.4f}")
        col_b.metric("p-value χ²", f"{chi2_p:.4f}")
        col_cc2.metric("Résultat", "Bonne adéquation ✅" if chi2_p > 0.05 else "Écart significatif ⚠️")

        verification_table([
            {"Loi": "Uniforme Discrète",  "Stat": "Moyenne",  "Empirique": disc_s.mean(), "Théorique": disc_mu_theo,  "Écart abs.": abs(disc_s.mean()-disc_mu_theo), "Écart %": abs(disc_s.mean()-disc_mu_theo)/disc_mu_theo*100},
            {"Loi": "Uniforme Discrète",  "Stat": "Variance", "Empirique": disc_s.var(),  "Théorique": disc_var_theo, "Écart abs.": abs(disc_s.var()-disc_var_theo),  "Écart %": abs(disc_s.var()-disc_var_theo)/disc_var_theo*100},
            {"Loi": "Uniforme Continue",  "Stat": "Moyenne",  "Empirique": cont_s.mean(), "Théorique": cont_mu_theo,  "Écart abs.": abs(cont_s.mean()-cont_mu_theo), "Écart %": abs(cont_s.mean()-cont_mu_theo)/cont_mu_theo*100},
            {"Loi": "Uniforme Continue",  "Stat": "Variance", "Empirique": cont_s.var(),  "Théorique": cont_var_theo, "Écart abs.": abs(cont_s.var()-cont_var_theo),  "Écart %": abs(cont_s.var()-cont_var_theo)/cont_var_theo*100},
        ])


# =============================================================================
#  NAVIGATION SIDEBAR + ROUTEUR PRINCIPAL
# =============================================================================

def main():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 15px 0;'>
            <div style='font-size:2.5rem;'>🔬</div>
            <h2 style='color:#7c83f5; margin:5px 0; font-size:1.1rem;'>DataScience Dashboard</h2>
            <p style='color:#4b5563; font-size:0.75rem; margin:0;'>v2.0 · Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📌 Navigation")
        pages = {
            "🏠  Accueil":                  "accueil",
            "🩺  Diagnostic du Cancer":     "cancer",
            "🏃‍♂️  Blessures Sportives":    "sports",
            "🎲  Simulations Statistiques": "simulations",
        }

        page_choisie = st.radio(
            "Sélectionner un module",
            list(pages.keys()),
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("""
        <div style='color:#4b5563; font-size:0.75rem; line-height:1.8;'>
            <strong style='color:#6b7280;'>Modèles :</strong><br>
            NB · KNN · DT · RF · Marbar-RF<br>
            Hybride (Weibull+Benford) · GF-MFTL<br><br>
            <strong style='color:#6b7280;'>Librairies :</strong><br>
            scikit-learn · imbalanced-learn<br>
            plotly · pandas · numpy · scipy
        </div>
        """, unsafe_allow_html=True)

    route = pages[page_choisie]
    if route == "accueil":
        page_accueil()
    elif route == "cancer":
        page_cancer()
    elif route == "sports":
        page_sports()
    elif route == "simulations":
        page_simulations()


if __name__ == "__main__":
    main()
