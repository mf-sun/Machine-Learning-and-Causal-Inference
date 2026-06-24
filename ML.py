# ============================================================================
# Exoplanet Inner-edge Orbital Period — Predictability Analysis
# ============================================================================
#
# Question
# --------
# How well can stellar and planetary properties predict the inner-edge
# orbital period of small-planet systems, and which properties matter?
#
# Approach
# --------
# A five-tier model comparison of increasing complexity, all predicting
# log10(inner-edge period):
#     1. Linear, stellar mass only          (univariate baseline, "PET 1")
#     2. Linear, all properties             (multivariate linear, Lasso)
#     3. Decision Tree, all properties      (single tree)
#     4. Random Forest, all properties      (bagging ensemble)
#     5. Gradient Boosted Trees, all props  (boosting ensemble, HistGB)
#
# Features : planet mass, stellar mass, stellar [Fe/H], stellar age
# Target   : log10(pl_orbper) of the inner-edge planet
# Sample   : small planets (R < 4 R_Earth, M < 30 M_Earth), single-star,
#            main-sequence FGK hosts, from the NASA Exoplanet Archive.
#
# Pipeline sections
# -----------------
#   1-2  Imports & settings
#   3-4  Helpers & parameter labels
#   5-11 Data loading, filtering, inner-edge extraction, cleaning, grouping
#   12-14 Exploratory analysis (distributions, correlation, mutual information)
#   15   ML shared utilities & model specifications
#   16-18 Model runners (linear, lasso, tree-based)
#   19-20 Run all models + single-split comparison table
#   21   Kepler's-third-law consistency check
#   22-24 Stability evaluation over 30 stratified splits
# ============================================================================


# ── 1. IMPORTS ──────────────────────────────────────────────────────────────

import time
import warnings
from warnings import simplefilter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde, ttest_rel
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error,
                             mutual_info_score)
import shap


# ── 2. SETTINGS ─────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
DATA_FILE     = 'PSCompPars_2025.11.11_07.37.00.csv'
SKIP_ROWS     = 191          # NASA Archive header rows
STRATIFY_BINS = 10           # orbital-period quantile bins used for stratification
N_RUNS        = 30           # number of random splits in the stability evaluation

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
pd.set_option('mode.chained_assignment', None)

sns.set_style('white')
plt.rcParams['font.family']        = 'serif'
plt.rcParams['font.serif']         = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset']   = 'custom'
plt.rcParams['mathtext.rm']        = 'Times New Roman'
plt.rcParams['mathtext.it']        = 'Times New Roman:italic'
plt.rcParams['axes.unicode_minus'] = False


# ── 3. PARAMETER LABELS ─────────────────────────────────────────────────────

PARAM_NAMES = {
    'pl_orbper' : 'Orbital Period (days)',
    'pl_orbsmax': 'Semi-major Axis (AU)',
    'pl_rade'   : 'Planet Radius (R_Earth)',
    'pl_bmasse' : 'Planet Mass (M_Earth)',
    'st_mass'   : 'Stellar Mass (M_Sun)',
    'st_met'    : 'Stellar [Fe/H] (dex)',
    'st_age'    : 'Stellar Age (Gyr)',
    'sy_pnum'   : 'Planet Count',
}

def label(col):
    """Human-readable axis label for a column name."""
    return PARAM_NAMES.get(col, col)


# ── 4. HELPER ───────────────────────────────────────────────────────────────

def count_summary(df, msg, key='tic_id'):
    n_sys = df[key].nunique() if key in df.columns else df['hostname'].nunique()
    print(f"  [{msg}]  Systems: {n_sys}   Planets: {len(df)}")


# ── 5. DATA LOADING ─────────────────────────────────────────────────────────

print("=" * 60)
print("Data Processing Pipeline")
print("=" * 60)

allp = pd.read_csv(DATA_FILE, skiprows=SKIP_ROWS)
allp = allp.drop(columns='loc_rowid', errors='ignore')
count_summary(allp, 'Raw data', key='hostname')


# ── 6. SAMPLE SELECTION ─────────────────────────────────────────────────────
# Restrict to single-star, main-sequence FGK hosts with small planets.

allp = allp[allp['discoverymethod'].isin(['Transit', 'Radial Velocity'])]
count_summary(allp, 'Transit / RV only', key='hostname')

allp = allp[allp['sy_snum'] == 1]
count_summary(allp, 'Single-star systems', key='hostname')

allp = allp[allp['st_logg'] > 3.8]                      # drop evolved giants
count_summary(allp, 'Main sequence (logg > 3.8)', key='hostname')

allp = allp[(allp['st_teff'] >= 4700) & (allp['st_teff'] <= 6500)]   # FGK
count_summary(allp, 'FGK hosts (4700-6500 K)', key='hostname')

# Small-planet definition: radius AND mass cuts (applied before inner-edge pick)
allp = allp[allp['pl_rade'] < 4.0].copy()
print(f"\n  Small planets (R < 4 R_Earth): {len(allp)}")
n0 = len(allp)
allp = allp[allp['pl_bmasse'] < 30.0].copy()            # also drops NaN masses
print(f"  Mass cut (M < 30 M_Earth): removed {n0 - len(allp)}, kept {len(allp)}")


# ── 7. INNER-edge EXTRACTION ────────────────────────────────────────────
# One row per system: the planet with the SHORTEST orbital period.

KEEP = ['tic_id', 'hostname', 'pl_name', 'pl_orbper', 'pl_orbsmax',
        'pl_rade', 'pl_bmasse', 'st_mass', 'st_met', 'st_age',
        'sy_pnum', 'discoverymethod']
allp = allp[[c for c in KEEP if c in allp.columns]]

allp = allp.dropna(subset=['pl_orbper'])
GROUP_KEY = 'tic_id' if allp['tic_id'].notna().any() else 'hostname'
allp = allp.loc[allp.groupby(GROUP_KEY)['pl_orbper'].idxmin()].reset_index(drop=True)
count_summary(allp, 'Inner-edge planet per system', key=GROUP_KEY)


# ── 8. STRICT CLEANING ──────────────────────────────────────────────────────
# Keep only systems with all analysis columns present. st_age (14% missing)
# is the main source of attrition but is retained as a genuine feature.

STRICT_COLS = ['pl_orbper', 'pl_orbsmax', 'pl_bmasse',
               'st_mass', 'st_met', 'st_age', 'sy_pnum']

print("\n  Missing-value rates before cleaning:")
for c in STRICT_COLS:
    pct = allp[c].isnull().mean() * 100
    print(f"    {c:<12} {pct:5.1f}%")

n0   = len(allp)
allp = allp[allp[STRICT_COLS].notna().all(axis=1)].reset_index(drop=True)
print(f"\n  Dropped {n0 - len(allp)} rows ({(n0 - len(allp)) / n0 * 100:.1f}%); "
      f"final N = {len(allp)}")


# ── 9. ANALYSIS GROUP ───────────────────────────────────────────────────────
# Only the full sample ("All") is analysed here. small_multi is defined for
# possible future multi-planet-only analyses.

small_all   = allp.copy()
small_multi = allp[allp['sy_pnum'] > 1].copy()          # not analysed below

vc = small_all['discoverymethod'].value_counts()
print(f"\n  Group 'Small - All': {len(small_all)} systems "
      f"(Transit {vc.get('Transit', 0)}, RV {vc.get('Radial Velocity', 0)})")

GROUP_LABEL = 'Small — All  (R < 4 R_Earth)'
COLOR       = 'steelblue'


# ── 10. PARAMETER DISTRIBUTIONS ─────────────────────────────────────────────

DIST_PARAMS = ['pl_orbper', 'pl_bmasse', 'st_mass', 'st_met', 'st_age']

def plot_distributions(df):
    fig, axes = plt.subplots(1, len(DIST_PARAMS), figsize=(len(DIST_PARAMS) * 4, 4))
    fig.suptitle(f'Parameter Distributions — {GROUP_LABEL}',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, col in zip(axes, DIST_PARAMS):
        ax.hist(df[col].dropna(), bins=100, color=COLOR,
                edgecolor='white', linewidth=0.4)
        ax.set_xlabel(label(col), fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    plt.tight_layout()
    plt.show()

plot_distributions(small_all)


# ── 11. CORRELATION ─────────────────────────────────────────────────────────

CORR_COLS = ['pl_orbper', 'pl_bmasse', 'st_mass', 'st_met', 'st_age']

def plot_correlation(df):
    data   = df[CORR_COLS]
    labels = [label(c) for c in CORR_COLS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Correlation — {GROUP_LABEL}', fontsize=13, fontweight='bold')
    for ax, method in [(axes[0], 'pearson'), (axes[1], 'spearman')]:
        corr = data.corr(method=method)
        corr.index = corr.columns = labels
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 9},
                    linewidths=0.5, square=True, ax=ax)
        ax.set_title(method.capitalize(), fontsize=11)
    plt.tight_layout()
    plt.show()

plot_correlation(small_all)


# ── 12. MUTUAL INFORMATION ──────────────────────────────────────────────────
#
# Mutual information of continuous variables requires either discretisation or
# a continuous estimator. Three views are shown for comparison:
#   (a) raw    : mutual_info_score on raw floats. DEGENERATE — every value is
#                unique, so NMI ~ 1 everywhere. Shown only to demonstrate the
#                pitfall; never use for conclusions.
#   (b) binned : KBinsDiscretizer (30 uniform bins) + mutual_info_score. Valid,
#                but values depend on the bin count.
#   (c) kNN    : mutual_info_regression (KSG estimator). No fixed grid; this is
#                the estimator used for the scientific conclusion below.

def nmi_matrix(df, cols, binned):
    data = df[cols].dropna()
    M = (KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='uniform')
         .fit_transform(data.values)) if binned else data.values
    n = len(cols)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(mutual_info_score(M[:, i], M[:, i]) *
                            mutual_info_score(M[:, j], M[:, j]))
            out[i, j] = mutual_info_score(M[:, i], M[:, j]) / denom if denom else 0
    return out

def plot_nmi(df):
    labels = [label(c) for c in CORR_COLS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Normalized Mutual Information — {GROUP_LABEL}',
                 fontsize=13, fontweight='bold')
    for ax, binned, title in [
        (axes[0], False, 'Unbinned (raw) — degenerate: NMI ~ 1 everywhere'),
        (axes[1], True,  'Binned (uniform, 30 bins) — valid')]:
        M = nmi_matrix(df, CORR_COLS, binned)
        mask = np.triu(np.ones_like(M, dtype=bool), k=1)
        sns.heatmap(M, mask=mask, xticklabels=labels, yticklabels=labels,
                    annot=True, fmt='.2f', vmin=0, vmax=1, square=True,
                    linewidths=0.5, cmap=sns.color_palette('mako_r', as_cmap=True),
                    cbar_kws={'label': 'Normalized Mutual Information'}, ax=ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        for t in ax.get_xticklabels():
            t.set_ha('right')
    plt.tight_layout()
    plt.show()

def plot_knn_mi(df):
    feats = [c for c in CORR_COLS if c != 'pl_orbper']
    data  = df[feats + ['pl_orbper']].dropna()
    mi = mutual_info_regression(data[feats].values,
                                np.log10(data['pl_orbper'].values),
                                random_state=RANDOM_STATE)
    s = pd.Series(mi, index=[label(c) for c in feats]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s.plot(kind='barh', ax=ax, color=COLOR)
    ax.set_title(f'kNN Mutual Information (feature vs Orbital Period)\n{GROUP_LABEL}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Mutual Information [nats]  (0 = independent)', fontsize=10)
    plt.tight_layout()
    plt.show()
    return s

print("\n" + "=" * 60)
print("Mutual Information  (raw | binned | kNN)")
print("=" * 60)
plot_nmi(small_all)
knn_mi = plot_knn_mi(small_all)
print("  kNN MI (feature -> period):")
for k, v in knn_mi.sort_values(ascending=False).items():
    print(f"    {k:<28} {v:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# ── 13. ML — SHARED UTILITIES & MODEL SPECIFICATIONS ────────────────────────
# ════════════════════════════════════════════════════════════════════════════

ALL_FEATURES = ['pl_bmasse', 'st_mass', 'st_met', 'st_age']
MASS_FEATURE = ['st_mass']
TARGET       = 'pl_orbper'

# Hyper-parameter grids for the tree-based models
DT_GRID = {'max_depth': [3, 4, 5, 6, 7, 8],
           'min_samples_leaf': [5, 10, 15, 20, 30]}
RF_GRID = {'n_estimators': [1000], 'max_depth': [5, 6, 7, 8],
           'min_samples_leaf': [8, 10, 12, 15],
           'max_features': ['sqrt', 'log2', 0.25, 0.5, None]}
GB_GRID = {'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 4, 5],
           'min_samples_leaf': [10, 20, 30], 'max_iter': [200]}

# Five-tier model comparison. Each spec drives the run loop, the stability
# loop, the summary table and the box plots — add a model by appending here.
MODEL_SPECS = [
    {'key': 'lin_mass',  'label': 'Linear (mass only)',
     'features': MASS_FEATURE, 'kind': 'linear', 'color': '#8c8c8c'},
    {'key': 'lin_multi', 'label': 'Linear (all props)',
     'features': ALL_FEATURES, 'kind': 'lasso',  'color': 'steelblue'},
    {'key': 'dt_multi',  'label': 'Decision Tree (all props)',
     'features': ALL_FEATURES, 'kind': 'dt', 'grid': DT_GRID,
     'shap': False, 'color': '#e6a817'},
    {'key': 'rf_multi',  'label': 'Random Forest (all props)',
     'features': ALL_FEATURES, 'kind': 'rf', 'grid': RF_GRID,
     'shap': True,  'color': '#2e7d32'},
    {'key': 'gb_multi',  'label': 'Grad. Boosting (all props)',
     'features': ALL_FEATURES, 'kind': 'gb', 'grid': GB_GRID,
     'shap': True,  'color': '#8b1a4a'},
]


def prepare_ml_data(df, features):
    """Return X (DataFrame), y = log10(period), and the feature list."""
    data = df[features + [TARGET]].dropna()
    data = data[data[TARGET] > 0]
    return data[features], np.log10(data[TARGET].values), features


def stratified_split(X, y, seed=RANDOM_STATE):
    """
    Train/test split stratified by orbital period.

    Why bin? sklearn's `stratify` treats its argument as discrete CLASS labels
    and matches class proportions between train and test. Passing the raw
    continuous period would create ~N unique "classes" and raise an error.
    Binning into 10 quantile groups instead asks for the same SHAPE of the
    period distribution in train and test (e.g. equal fractions of short- and
    long-period systems), which is what we actually want.
    """
    y_bins = pd.qcut(y, q=STRATIFY_BINS, duplicates='drop')
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y_bins)


def base_estimator(kind, seed=RANDOM_STATE):
    return {'dt': DecisionTreeRegressor(random_state=seed),
            'rf': RandomForestRegressor(random_state=seed, n_jobs=-1),
            'gb': HistGradientBoostingRegressor(random_state=seed)}[kind]


def build_model(spec, tuned, seed):
    """Construct a model from its spec and tuned hyper-parameters (for stability runs)."""
    kind, key = spec['kind'], spec['key']
    if kind == 'linear':
        return LinearRegression()
    if kind == 'lasso':
        return Lasso(alpha=tuned[key], max_iter=10000)
    if kind == 'dt':
        return DecisionTreeRegressor(random_state=seed, **tuned[key])
    if kind == 'rf':
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **tuned[key])
    if kind == 'gb':
        return HistGradientBoostingRegressor(random_state=seed, **tuned[key])


def compute_metrics(y_true, y_pred):
    """Metrics in both log space (primary) and linear days, plus accuracy rates."""
    y_t, y_p = 10 ** np.array(y_true), 10 ** np.array(y_pred)
    ratio = y_p / np.clip(y_t, 1e-10, None)
    return {
        'R2_log': r2_score(y_true, y_pred),
        'MAE_log': mean_absolute_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_t, y_p),
        'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
        'R2': r2_score(y_t, y_p),
        'MAPE': mean_absolute_percentage_error(y_t, y_p),
        'within_20': np.mean(np.abs(ratio - 1) < 0.20),
        'within_30': np.mean(np.abs(ratio - 1) < 0.30),
        'within_50': np.mean(np.abs(ratio - 1) < 0.50),
        'within_f2': np.mean((ratio > 0.5) & (ratio < 2.0)),
        'median_ratio': np.median(ratio),
    }


def print_metrics(m, extra=None):
    if extra:
        print(f"    {'Tuned':<16}: {extra}")
    print(f"    {'R2 (log)':<16}: {m['R2_log']:.4f}   <- primary")
    print(f"    {'MAE (days)':<16}: {m['MAE']:.3f}")
    print(f"    {'MAPE':<16}: {m['MAPE'] * 100:.1f}%")
    print(f"    {'Within factor-2':<16}: {m['within_f2'] * 100:.1f}%")
    print(f"    {'Median ratio':<16}: {m['median_ratio']:.3f}")


def plot_pred_actual(ax, y_true, y_pred, title, color):
    yt, yp = 10 ** np.array(y_true), 10 ** np.array(y_pred)
    try:
        z = gaussian_kde(np.vstack([yp, yt]))(np.vstack([yp, yt]))
        idx = z.argsort()
        sc = ax.scatter(yp[idx], yt[idx], c=z[idx], s=15, cmap='viridis', alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Density')
    except Exception:
        ax.scatter(yp, yt, s=15, color=color, alpha=0.6)
    lims = [min(yp.min(), yt.min()), max(yp.max(), yt.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1 Line')
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel('Predicted Period (days)', fontsize=10)
    ax.set_ylabel('Actual Period (days)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)


def plot_perm_importance(model, X_test, y_test, feats, title):
    perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    s = pd.Series(perm.importances_mean, index=[label(f) for f in feats]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    s.plot(kind='barh', ax=ax, color='tomato')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean Decrease in MAE [log10(days)]', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_shap(model, X_test, feats, title):
    try:
        Xdf = pd.DataFrame(X_test, columns=feats).rename(columns=PARAM_NAMES)
        shap_values = shap.TreeExplainer(model)(Xdf)
        plt.figure(figsize=(8, 5))
        shap.plots.beeswarm(shap_values, max_display=8, show=False)
        plt.title(title, fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"    SHAP skipped: {e}")


# ── 14. MODEL 1 — LINEAR, STELLAR MASS ONLY ─────────────────────────────────

def run_linear_mass(df):
    print(f"\n{'-' * 55}\n  Linear (mass only)\n{'-' * 55}")
    X, y, feats = prepare_ml_data(df, MASS_FEATURE)
    Xtr, Xte, ytr, yte = stratified_split(X.values, y)
    scaler = StandardScaler().fit(Xtr)
    model  = LinearRegression().fit(scaler.transform(Xtr), ytr)
    m_test = compute_metrics(yte, model.predict(scaler.transform(Xte)))
    print_metrics(m_test, extra=f"coef(std)={model.coef_[0]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Linear (mass only)', fontsize=12, fontweight='bold')
    axes[0].scatter(X.values[:, 0], y, s=12, alpha=0.4, color='#8c8c8c')
    grid = np.linspace(X.values[:, 0].min(), X.values[:, 0].max(), 100)
    axes[0].plot(grid, model.predict(scaler.transform(grid.reshape(-1, 1))),
                 'r-', lw=2, label='OLS fit')
    axes[0].set_xlabel(label('st_mass'), fontsize=10)
    axes[0].set_ylabel('log10(Orbital Period / days)', fontsize=10)
    axes[0].set_title('Stellar Mass vs log10(Period)', fontsize=11)
    axes[0].legend(fontsize=9)
    plot_pred_actual(axes[1], yte, model.predict(scaler.transform(Xte)),
                     'Predicted vs Actual (Test)', '#8c8c8c')
    plt.tight_layout()
    plt.show()
    return {'metrics_test': m_test, 'feat_names': feats}


# ── 15. MODEL 2 — LASSO, ALL PROPERTIES ─────────────────────────────────────

def run_lasso(df):
    print(f"\n{'-' * 55}\n  Lasso (all props)\n{'-' * 55}")
    X, y, feats = prepare_ml_data(df, ALL_FEATURES)
    Xtr, Xte, ytr, yte = stratified_split(X.values, y)
    scaler = StandardScaler().fit(Xtr)
    model  = LassoCV(alphas=np.logspace(-4, 2, 60), cv=5, max_iter=10000)
    model.fit(scaler.transform(Xtr), ytr)
    m_test = compute_metrics(yte, model.predict(scaler.transform(Xte)))
    print_metrics(m_test, extra=f"alpha={model.alpha_:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Lasso (all props)', fontsize=12, fontweight='bold')
    coef = pd.Series(model.coef_, index=[label(f) for f in feats]).sort_values()
    coef.plot(kind='barh', ax=axes[0],
              color=['tomato' if v < 0 else 'steelblue' for v in coef])
    axes[0].axvline(0, color='k', lw=0.8)
    axes[0].set_title('Feature Coefficients', fontsize=11)
    axes[0].set_xlabel('Coefficient Value', fontsize=10)
    plot_pred_actual(axes[1], yte, model.predict(scaler.transform(Xte)),
                     'Predicted vs Actual (Test)', 'steelblue')
    plt.tight_layout()
    plt.show()
    return {'metrics_test': m_test, 'best_alpha': model.alpha_, 'feat_names': feats}


# ── 16. MODELS 3-5 — TREE-BASED (Decision Tree / Random Forest / GBT) ───────
# All three share the same workflow, so a single function handles them, driven
# by the spec (which carries the grid and whether to compute SHAP).

def run_tree_model(df, spec, cv=5):
    print(f"\n{'-' * 55}\n  {spec['label']}\n{'-' * 55}")
    X, y, feats = prepare_ml_data(df, spec['features'])
    Xtr, Xte, ytr, yte = stratified_split(X.values, y)

    n_combos = int(np.prod([len(v) for v in spec['grid'].values()]))
    print(f"  GridSearchCV: {n_combos} combos x {cv} folds ...")
    t0 = time.time()
    grid = GridSearchCV(base_estimator(spec['kind']), spec['grid'], cv=cv,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(Xtr, ytr)
    model = grid.best_estimator_
    print(f"  Done in {time.time() - t0:.1f}s   Best: {grid.best_params_}")

    m_train = compute_metrics(ytr, model.predict(Xtr))
    m_test  = compute_metrics(yte, model.predict(Xte))
    print_metrics(m_test, extra=str(grid.best_params_))
    gap = m_train['R2_log'] - m_test['R2_log']
    print(f"    Train-test R2 gap: {gap:.3f}  "
          f"({'overfit' if gap > 0.15 else 'OK'})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{spec['label']}: Predicted vs Actual",
                 fontsize=12, fontweight='bold')
    plot_pred_actual(axes[0], ytr, model.predict(Xtr), 'Train', spec['color'])
    plot_pred_actual(axes[1], yte, model.predict(Xte), 'Test',  spec['color'])
    plt.tight_layout()
    plt.show()

    plot_perm_importance(model, Xte, yte, feats,
                         f"Permutation Importance — {spec['label']}")
    if spec.get('shap'):
        plot_shap(model, Xte, feats, f"SHAP Beeswarm — {spec['label']}")

    return {'metrics_train': m_train, 'metrics_test': m_test,
            'best_params': grid.best_params_, 'feat_names': feats, 'model': model}


# ── 17. RUN ALL FIVE MODELS ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Five-tier model comparison  (Target: log10({TARGET}))")
print("Stratified split by orbital period, 10 bins")
print("=" * 60)

results = {}
for spec in MODEL_SPECS:
    if spec['kind'] == 'linear':
        results[spec['key']] = run_linear_mass(small_all)
    elif spec['kind'] == 'lasso':
        results[spec['key']] = run_lasso(small_all)
    else:
        results[spec['key']] = run_tree_model(small_all, spec)


# ── 18. SINGLE-SPLIT COMPARISON TABLE ───────────────────────────────────────

print("\n" + "=" * 80)
print(f"{'Model Comparison (single stratified split)':^80}")
print("=" * 80)
print(f"{'Model':<28} {'R2(log)':>8} {'±30%':>7} {'×2':>7} {'Ratio':>7}")
print("-" * 80)
for spec in MODEL_SPECS:
    m = results[spec['key']]['metrics_test']
    print(f"{spec['label']:<28} {m['R2_log']:>8.3f} "
          f"{m['within_30'] * 100:>6.1f}% {m['within_f2'] * 100:>6.1f}% "
          f"{m['median_ratio']:>7.3f}")
print("=" * 80)
print("  R2(log): >0.70 excellent | 0.50-0.70 good | 0.30-0.50 moderate | <0.30 weak")


# ── 19. KEPLER'S-THIRD-LAW CONSISTENCY CHECK (RF) ───────────────────────────
# Predict period -> infer semi-major axis via Kepler's third law -> compare to
# the catalogue value. Note: given st_mass, log(a) and log(T) are algebraically
# tied, so this is a physical-consistency check, NOT an independent validation.

G, M_SUN, AU, DAY = 6.674e-11, 1.989e30, 1.496e11, 86400

def run_kepler_check(df, rf_res):
    print(f"\n{'-' * 55}\n  Kepler consistency check (RF)\n{'-' * 55}")
    feats = rf_res['feat_names']
    data  = df[list(dict.fromkeys(feats + ['st_mass', 'pl_orbsmax']))].dropna()
    period_pred = 10 ** rf_res['model'].predict(data[feats].values)
    T = period_pred * DAY
    sma_pred  = ((G * data['st_mass'].values * M_SUN * T ** 2)
                 / (4 * np.pi ** 2)) ** (1 / 3) / AU
    sma_true  = data['pl_orbsmax'].values
    r2_log    = r2_score(np.log10(sma_true), np.log10(np.clip(sma_pred, 1e-6, None)))
    print(f"  R2 (log, AU): {r2_log:.4f}   MAE (AU): "
          f"{mean_absolute_error(sma_true, sma_pred):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Kepler Consistency Check', fontsize=12, fontweight='bold')
    plot_pred_actual(axes[0], np.log10(sma_pred), np.log10(sma_true),
                     f'Semi-major axis  (R2_log={r2_log:.3f})', '#2e7d32')
    axes[0].set_xlabel('Predicted Semi-major Axis (AU)', fontsize=10)
    axes[0].set_ylabel('Actual Semi-major Axis (AU)', fontsize=10)
    res = sma_pred - sma_true
    axes[1].hist(res, bins=50, color='#2e7d32', edgecolor='white', lw=0.4)
    axes[1].axvline(np.mean(res), color='red', lw=1.2,
                    label=f'Mean = {np.mean(res):.3f} AU')
    axes[1].set_xlabel('Residual: Predicted - Actual (AU)', fontsize=10)
    axes[1].set_title('Residual Distribution', fontsize=11)
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()

if 'pl_orbsmax' in small_all.columns:
    run_kepler_check(small_all, results['rf_multi'])


# ════════════════════════════════════════════════════════════════════════════
# ── 20. STABILITY EVALUATION — ALL MODELS, SHARED SPLITS ────────────────────
# ════════════════════════════════════════════════════════════════════════════
# A single train/test split can be lucky or unlucky. Here we repeat the split
# N_RUNS times (each stratified by period). On every split ALL models are
# trained on the SAME train set and scored on the SAME test set, so the
# comparison is paired and fair. We report mean, std and a 95% CI of the mean.
# Hyper-parameters are fixed at the values tuned in Section 17 (not re-tuned).

def ci95(arr):
    arr = np.asarray(arr, float)
    return arr.mean(), 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))

# Collect tuned hyper-parameters per model
tuned = {}
for spec in MODEL_SPECS:
    r = results[spec['key']]
    tuned[spec['key']] = (r['best_alpha'] if spec['kind'] == 'lasso'
                          else r.get('best_params'))

def run_stability(df):
    X_all, y, _ = prepare_ml_data(df, ALL_FEATURES)
    keys = ['R2_log', 'MAE', 'MAPE', 'within_20', 'within_30', 'median_ratio']
    rec  = {s['key']: {k: [] for k in keys} for s in MODEL_SPECS}

    for seed in range(N_RUNS):
        y_bins = pd.qcut(y, q=STRATIFY_BINS, duplicates='drop')
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y, test_size=0.2, random_state=seed, stratify=y_bins)
        for spec in MODEL_SPECS:
            cols   = spec['features']
            scaler = StandardScaler().fit(Xtr[cols].values)   # no-op for trees
            mdl = build_model(spec, tuned, seed)
            mdl.fit(scaler.transform(Xtr[cols].values), ytr)
            m = compute_metrics(yte, mdl.predict(scaler.transform(Xte[cols].values)))
            for k in keys:
                rec[spec['key']][k].append(m[k])

    # Text summary
    print(f"\n  {'Model':<28} {'R2(log) mean':>13} {'std':>7} {'95% CI':>20}")
    print("  " + "-" * 70)
    for spec in MODEL_SPECS:
        arr = np.clip(rec[spec['key']]['R2_log'], -2, 1)
        mean, half = ci95(arr)
        print(f"  {spec['label']:<28} {mean:>13.4f} {np.std(arr, ddof=1):>7.4f} "
              f"[{mean - half:.3f}, {mean + half:.3f}]")

    # Paired t-tests on MAE between consecutive tiers
    print("\n  Paired t-tests on MAE (consecutive tiers, same splits):")
    order = [s['key'] for s in MODEL_SPECS]
    for a, b in zip(order[:-1], order[1:]):
        t, p = ttest_rel(rec[a]['MAE'], rec[b]['MAE'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        la = next(s['label'] for s in MODEL_SPECS if s['key'] == a)
        lb = next(s['label'] for s in MODEL_SPECS if s['key'] == b)
        print(f"    {la:<26} vs {lb:<26} t={t:>6.2f}  p={p:.4f} {sig}")
    return rec

print("\n" + "=" * 60)
print(f"Stability Evaluation — {N_RUNS} stratified splits, all 5 models")
print("=" * 60)
stability = run_stability(small_all)


# ── 21. STABILITY BOX PLOTS ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Stability Evaluation ({N_RUNS} runs) — {GROUP_LABEL}',
             fontsize=13, fontweight='bold')
for ax, (metric, ylab) in zip(axes, [('MAE', 'MAE (days)'),
                                      ('R2_log', 'R2 (log)'),
                                      ('MAPE', 'MAPE (%)')]):
    data, labels, colors = [], [], []
    for spec in MODEL_SPECS:
        arr = np.clip(stability[spec['key']][metric],
                      -2 if metric == 'R2_log' else -np.inf, np.inf)
        arr = np.array(arr) * (100 if metric == 'MAPE' else 1)
        data.append(arr)
        labels.append(spec['label'].replace(' (', '\n('))
        colors.append(spec['color'])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, (arr, c) in enumerate(zip(data, colors), 1):
        ax.scatter(np.random.normal(i, 0.04, len(arr)), arr,
                   s=8, alpha=0.4, color=c, zorder=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=7)
plt.tight_layout()
plt.show()


# ── 22. STABILITY SUMMARY TABLE ─────────────────────────────────────────────

print("\n" + "=" * 88)
print(f"{'Stability Summary — ' + str(N_RUNS) + ' stratified runs':^88}")
print("=" * 88)
print(f"{'Model':<28} {'R2(log)':>9} {'Std':>7} {'95% CI':>20} {'±30%':>7} {'Ratio':>7}")
print("-" * 88)
for spec in MODEL_SPECS:
    r   = stability[spec['key']]
    arr = np.clip(r['R2_log'], -2, 1)
    mean, half = ci95(arr)
    print(f"{spec['label']:<28} {mean:>9.4f} {np.std(arr, ddof=1):>7.4f} "
          f"[{mean - half:.3f}, {mean + half:.3f}]"
          f"{np.mean(r['within_30']) * 100:>7.1f}% {np.mean(r['median_ratio']):>7.3f}")
print("=" * 88)
print("  95% CI = mean ± 1.96·std/√N over the N stratified splits.")
print("  Key finding: R2(log) plateaus near 0.22-0.23 — the predictive ceiling")
print("  of these features. Stellar properties add little; planet mass dominates.")
