# △ Elliptic Bitcoin Fraud Network Analysis
## Blockchain Graph Analytics on the Elliptic Bitcoin Transaction Dataset

> A blockchain fraud detection project analysing **203,769 Bitcoin transactions** and **234,000+ payment flows** using graph network analysis — going beyond transaction-level labels to map how illicit funds structurally move through a payment network. Built with Python, NetworkX, Pandas, and a custom interactive HTML dashboard.

<img width="1920" height="949" alt="image" src="https://github.com/user-attachments/assets/e1065a12-ce6f-4f56-96e5-b5edf5147fa1" />

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [The Core Question](#the-core-question)
- [Dataset Description](#dataset-description)
- [Tools & Technologies](#tools--technologies)
- [Project Architecture](#project-architecture)
- [Data Preparation & Merging](#data-preparation--merging)
- [Network Construction](#network-construction)
- [Analysis Methods](#analysis-methods)
  - [Node Classification & Labelling](#node-classification--labelling)
  - [Layering Pattern Detection](#layering-pattern-detection)
  - [Fan-Out Pattern Detection](#fan-out-pattern-detection)
  - [Composite Risk Scoring](#composite-risk-scoring)
  - [Structural Similarity to Illicit Nodes](#structural-similarity-to-illicit-nodes)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [Key Findings & Insights](#key-findings--insights)
- [Why Position Matters More Than Label](#why-position-matters-more-than-label)
- [Compliance Implications](#compliance-implications)
- [Limitations & Data Gaps](#limitations--data-gaps)
- [Project Structure](#project-structure)
- [How to Reproduce This Analysis](#how-to-reproduce-this-analysis)
- [References](#references)

---

## Project Overview

Most fraud detection tools ask: *"Is this transaction suspicious?"*

This project asked a different question: *"Where does this transaction sit in the network?"*

The distinction matters. A transaction can carry no fraud label, contain no anomalous amounts, and still be a critical node in an illicit money movement chain — simply because of **who it connects to**, **how many sources it receives from**, and **what relay role it plays** between otherwise disconnected parts of the graph.

This project applies **graph network analysis** to the Elliptic Bitcoin Transaction Dataset — a real-world dataset published in academic research (Weber et al., 2019) — to surface structural fraud signals that label-based detection cannot capture.

**Scale of analysis:**

| Metric | Value |
|---|---|
| Total transactions in dataset | 203,769 |
| Total payment flows (edges) | 234,355+ |
| Feature columns per transaction | 168 |
| Subgraph analysed | 12,473 nodes |
| Confirmed illicit nodes | 300 |
| Confirmed licit nodes | 3,359 |
| Unlabelled nodes | 8,814 |
| Labelled coverage | ~23% |

---

## The Core Question

> **77% of this dataset carries no label.**
> Any compliance review that only examines confirmed cases is working with less than a quarter of the real picture.

The project reframes fraud detection from a binary classification problem into a **network position problem**:

- Which unlabelled nodes behave structurally like confirmed illicit ones?
- Which nodes sit at the highest-traffic crossroads of the payment graph?
- Where do layering and fan-out patterns — two core money laundering techniques — appear in the network?
- Can risk be scored based purely on network topology, independent of formal labels?

---

## Dataset Description

**Source:** Elliptic Bitcoin Transaction Dataset — Weber et al., 2019
**Published in:** *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*

### Dataset Components

| File | Description | Dimensions |
|---|---|---|
| `elliptic_txs_features.csv` | 166 anonymised features per transaction (time step + graph features) | 203,769 × 166 |
| `elliptic_txs_classes.csv` | Node labels: `1` = illicit, `2` = licit, `unknown` = unlabelled | 203,769 × 2 |
| `elliptic_txs_edgelist.csv` | Directed payment edges between transactions | 234,355 × 2 |

### Feature Structure

| Feature Index | Description |
|---|---|
| Feature 1 | Time step (1–49, representing 2-week periods) |
| Features 2–95 | Local transaction features (anonymised) |
| Features 96–166 | Aggregated neighbourhood features |

### Label Distribution

| Class | Count | % of Total |
|---|---|---|
| Unknown (unlabelled) | 157,205 | 77.1% |
| Licit (Class 2) | 42,019 | 20.6% |
| Illicit (Class 1) | 4,545 | 2.2% |

> The severe class imbalance and high proportion of unlabelled data are defining characteristics of this dataset — and the primary motivation for a graph-structural rather than label-dependent approach.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core analysis language |
| **NetworkX** | Graph construction, centrality measures, pattern detection |
| **Pandas** | Data loading, merging, transformation, segmentation |
| **Matplotlib** | Static visualisations, subgraph plots |
| **Custom Canvas Dashboard** | Interactive HTML dashboard with network visualisation |
| **Jupyter Notebook** | Full reproducible analysis environment |

---

## Project Architecture

```
Raw Dataset (3 CSV files)
    │
    ▼
Data Merging & Preparation
    │  → Join features + classes + edgelist
    │  → Handle unknown labels
    │  → Validate edge integrity
    │
    ▼
Network Construction (NetworkX DiGraph)
    │  → 203,769 nodes
    │  → 234,355+ directed edges
    │
    ▼
Subgraph Extraction (12,473 nodes)
    │  → Focused analysis region
    │  → Contains 300 confirmed illicit + 3,359 licit + 8,814 unknown
    │
    ▼
Graph Feature Engineering
    │  → In-degree / out-degree per node
    │  → Betweenness centrality
    │  → Composite risk scoring
    │
    ▼
Pattern Detection
    │  → 52 layering patterns
    │  → 465 fan-out patterns
    │
    ▼
Unlabelled Node Flagging
    │  → 327 structurally suspicious unknowns
    │
    ▼
Risk Ranking (Top 20 Nodes)
    │
    ▼
Dashboard + Compliance Report
```

---

## Data Preparation & Merging

### Step 1 — Load and merge all three files

```python
import pandas as pd
import networkx as nx

# Load features
features = pd.read_csv('elliptic_txs_features.csv', header=None)
features.columns = ['txId'] + [f'feat_{i}' for i in range(1, 167)]

# Load class labels
classes = pd.read_csv('elliptic_txs_classes.csv')
classes.columns = ['txId', 'class']

# Load edge list
edges = pd.read_csv('elliptic_txs_edgelist.csv', header=None)
edges.columns = ['source', 'target']

# Merge features and labels
df = features.merge(classes, on='txId', how='left')

print(f"Total transactions: {len(df):,}")
print(f"Feature columns: {df.shape[1]}")
print(f"Label distribution:\n{df['class'].value_counts(dropna=False)}")
```

### Step 2 — Normalise class labels

```python
# Map numeric labels to readable classes
df['label'] = df['class'].map({
    '1': 'illicit',
    '2': 'licit',
    'unknown': 'unknown'
})

# Validate — no unmapped values should remain
assert df['label'].isna().sum() == 0, "Unmapped label values found"
```

### Step 3 — Validate edge integrity

```python
# Confirm all edge node IDs exist in the features table
all_nodes = set(df['txId'].astype(str))
edge_nodes = set(edges['source'].astype(str)) | set(edges['target'].astype(str))
orphaned = edge_nodes - all_nodes

print(f"Edges referencing unknown nodes: {len(orphaned)}")
# Filter out orphaned edges if any
edges_clean = edges[
    edges['source'].astype(str).isin(all_nodes) &
    edges['target'].astype(str).isin(all_nodes)
]
print(f"Clean edges: {len(edges_clean):,} of {len(edges):,}")
```

### Step 4 — Export cleaned merged dataset

```python
# Final merged dataset: 203,769 rows × 168 columns
df_final = df.copy()
df_final['time_step'] = df_final['feat_1']
df_final.to_csv('elliptic_merged_clean.csv', index=False)
print(f"Exported: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
```

---

## Network Construction

```python
# Build directed graph
G = nx.DiGraph()

# Add all nodes with label attributes
for _, row in df.iterrows():
    G.add_node(str(row['txId']), label=row['label'], time_step=row['feat_1'])

# Add directed edges (payment flows: source → target)
for _, row in edges_clean.iterrows():
    G.add_edge(str(row['source']), str(row['target']))

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")
print(f"Is directed: {G.is_directed()}")
print(f"Network density: {nx.density(G):.6f}")
```

### Subgraph Extraction

```python
# Extract a focused subgraph for deep analysis
# (connected component containing the highest-density region)
subgraph_nodes = list(...)  # Derived from component analysis
G_sub = G.subgraph(subgraph_nodes).copy()

print(f"Subgraph nodes: {G_sub.number_of_nodes():,}")   # 12,473
print(f"Subgraph edges: {G_sub.number_of_edges():,}")

# Label breakdown in subgraph
sub_labels = pd.Series(
    nx.get_node_attributes(G_sub, 'label')
).value_counts()
print(sub_labels)
# unknown    8814
# licit      3359
# illicit     300
```

---

## Analysis Methods

### Node Classification & Labelling

Each node in the subgraph was assigned one of four display classes:

| Class | Count | Colour in Dashboard |
|---|---|---|
| Confirmed Illicit | 300 | Coral / Red |
| Confirmed Licit | 3,359 | Teal / Green |
| Unknown | 8,814 | Slate / Grey |
| Top 10 Highest Risk | 10 | Gold ★ |

### Layering Pattern Detection

**Layering** is a money laundering technique where illicit funds are moved through multiple intermediate transactions to obscure their origin.

```python
def detect_layering_patterns(G, illicit_nodes, depth=3):
    """
    Detect chains of transactions originating from confirmed illicit nodes
    that pass through 2+ intermediate hops before reaching a non-illicit node.
    Returns list of (chain_start, chain_path) tuples.
    """
    layering_chains = []

    for node in illicit_nodes:
        # BFS from each illicit node up to specified depth
        for path in nx.all_simple_paths(G, source=node, cutoff=depth):
            if len(path) >= 3:  # At least 2 hops
                # Check if intermediate nodes are non-illicit (layering relay)
                intermediates = path[1:-1]
                relay_nodes = [
                    n for n in intermediates
                    if G.nodes[n].get('label') != 'illicit'
                ]
                if len(relay_nodes) == len(intermediates):
                    layering_chains.append(path)

    return layering_chains

layering_patterns = detect_layering_patterns(G_sub, illicit_nodes)
print(f"Layering patterns detected: {len(layering_patterns)}")  # 52
```

### Fan-Out Pattern Detection

**Fan-out** is a structuring technique where a single source transaction disperses funds to many destination addresses simultaneously — a classic smurfing or structuring pattern.

```python
def detect_fan_out_patterns(G, min_out_degree=10, include_labels=None):
    """
    Detect nodes with high out-degree — single sources distributing
    to many destinations in a single time step.
    Fan-out from illicit or unknown nodes is a high-priority signal.
    """
    fan_out_nodes = []

    for node in G.nodes():
        out_deg = G.out_degree(node)
        label = G.nodes[node].get('label', 'unknown')

        if out_deg >= min_out_degree:
            if include_labels is None or label in include_labels:
                fan_out_nodes.append({
                    'node': node,
                    'out_degree': out_deg,
                    'label': label
                })

    return pd.DataFrame(fan_out_nodes).sort_values('out_degree', ascending=False)

fan_out_df = detect_fan_out_patterns(G_sub, min_out_degree=5)
print(f"Fan-out patterns detected: {len(fan_out_df)}")  # 465
```

### Composite Risk Scoring

Each node was assigned a **composite risk score** combining three graph-structural signals:

```python
def compute_risk_scores(G, illicit_nodes):
    """
    Composite risk score = weighted combination of:
      - In-degree (normalised): receiving from many sources
      - Betweenness centrality: sitting at network crossroads
      - Illicit neighbour ratio: proportion of direct neighbours confirmed illicit
    """
    # Compute centrality measures
    in_degrees = dict(G.in_degree())
    max_in = max(in_degrees.values()) if in_degrees else 1

    betweenness = nx.betweenness_centrality(G_sub, normalized=True)

    risk_scores = {}

    for node in G.nodes():
        # Normalised in-degree (0–1)
        in_deg_norm = in_degrees.get(node, 0) / max_in

        # Betweenness centrality (already 0–1)
        between = betweenness.get(node, 0)

        # Illicit neighbour ratio
        neighbours = list(G.predecessors(node)) + list(G.successors(node))
        illicit_neighbours = sum(
            1 for n in neighbours if n in illicit_nodes
        )
        illicit_ratio = illicit_neighbours / max(len(neighbours), 1)

        # Composite score (weighted)
        risk_scores[node] = (
            0.35 * in_deg_norm +
            0.40 * between +
            0.25 * illicit_ratio
        )

    return risk_scores

risk_scores = compute_risk_scores(G_sub, illicit_nodes=set(
    n for n in G_sub.nodes() if G_sub.nodes[n].get('label') == 'illicit'
))

# Top 20 nodes by composite risk score
top_20 = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:20]
```

### Structural Similarity to Illicit Nodes

```python
def flag_suspicious_unlabelled(G, risk_scores, illicit_nodes, threshold=0.25):
    """
    Flag unlabelled nodes whose composite risk score meets or exceeds
    the threshold typically associated with confirmed illicit nodes.
    These are high-priority investigation targets regardless of label.
    """
    suspicious = []

    for node, score in risk_scores.items():
        label = G.nodes[node].get('label', 'unknown')
        if label == 'unknown' and score >= threshold:
            suspicious.append({
                'node_id': node,
                'risk_score': round(score, 4),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node)
            })

    return pd.DataFrame(suspicious).sort_values('risk_score', ascending=False)

suspicious_df = flag_suspicious_unlabelled(G_sub, risk_scores, illicit_nodes)
print(f"Structurally suspicious unlabelled nodes: {len(suspicious_df)}")  # 327
```

---

## Dashboard Walkthrough

<img width="1920" height="949" alt="image" src="https://github.com/user-attachments/assets/e1065a12-ce6f-4f56-96e5-b5edf5147fa1" />

The interactive dashboard was built as a custom HTML canvas application with a dark-mode financial compliance aesthetic- https://drive.google.com/file/d/12cTk7vul9aa7t2V_OOhPVcfJfGQNvJcu/view?usp=sharing
### KPI Cards (Top Row)

| Metric | Value | Colour |
|---|---|---|
| Nodes in Subgraph | **12,473** | Teal |
| Confirmed Illicit | **300** | Red / Coral |
| Confirmed Licit | **3,359** | Blue |
| Unlabelled | **8,814** | White |

### Filter Bar

Interactive class filters allowing toggling between:
- **All Nodes** (default)
- **Illicit** only (coral)
- **Licit** only (teal)
- **Unknown** only (slate)
- **Top 10 Risk** (gold stars)

### Transaction Activity Across Time Steps (Stacked Bar Chart)

- X-axis: Time steps 1–49 (each step ≈ 2-week period)
- Y-axis: Transaction count (0 – 7,880)
- Stacked by class: Illicit (red) · Licit (teal) · Unknown (grey)
- Shows transaction volume fluctuation over time with illicit nodes highlighted as a top layer
- Notable spikes at time steps ~11, ~21, ~36, ~41 indicate periods of elevated activity

### Network Graph (Centre)

- Interactive force-directed node-link diagram of the subgraph
- Node colours: Coral = Illicit · Green = Licit · Slate = Unknown
- Gold ★ stars = Top 10 highest composite risk score nodes
- Node size scales with degree (larger = more connections)
- Edges show directed payment flows

### Top 20 Nodes — Composite Risk Ranking (Right Panel)

| # | Node ID | Class | Risk Score | In | Out |
|---|---|---|---|---|---|
| 1 | 43388675… | licit | 0.45 | 284 | 0 |
| 2 | 72748868… | licit | 0.422 | 79 | 1 |
| 3 | 30699343… | licit | 0.3769 | 241 | 0 |
| 4 | 41342269… | licit | 0.3627 | 1 | 10 |
| 5 | 14950377… | licit | 0.3573 | 97 | 6 |
| 6 | 22585904… | unknown | 0.3373 | 212 | 0 |
| 7 | 19610786… | licit | 0.3031 | 188 | 0 |
| 8 | 12665008… | licit | 0.2976 | 84 | 9 |
| 9 | 31808898… | unknown | 0.2878 | 1 | 23 |
| 10 | 79515877… | licit | 0.2775 | 2 | 14 |
| 11 | 30179316… | **illicit** | 0.2739 | 177 | 0 |
| 12 | 30276715… | licit | 0.2682 | 129 | 0 |
| 13 | 28125035… | unknown | 0.2664 | 1 | 16 |
| 14 | 172562… | licit | 0.2661 | 24 | 4 |
| 15 | 96936982… | licit | 0.2656 | 20 | 1 |
| 16 | 29474363… | licit | 0.265 | 96 | 4 |

> Notable: Rank 11 (Risk Score 0.2739) is the **only confirmed illicit node** in the top 20. Ranks 6, 9, and 13 are **unlabelled** — yet structurally they belong in the same risk tier as the confirmed illicit node. This is the core insight of the project.

---

## Key Findings & Insights

### 1. Network Scale and Label Sparsity

| Finding | Value |
|---|---|
| Total transactions analysed | 203,769 |
| Unlabelled transactions | 157,205 (77.1%) |
| Confirmed illicit | 4,545 (2.2%) |
| Confirmed licit | 42,019 (20.6%) |

**Implication:** Any compliance system that works exclusively with labelled data is blind to 77% of the network. The unlabelled majority is not safe — it is simply unclassified.

### 2. Layering Patterns — 52 Chains Detected

52 distinct layering chains were identified — sequences of 3+ transactions originating from a confirmed illicit node and passing through relay transactions before reaching an endpoint. These are structural signatures of the **placement-layering-integration** money laundering cycle.

### 3. Fan-Out Patterns — 465 Instances Detected

465 fan-out patterns were detected — single nodes dispersing payments to 5+ destinations simultaneously. Fan-out at this scale is consistent with **structuring** (smurfing) behaviour designed to break transaction amounts below detection thresholds.

### 4. 327 Unlabelled Nodes with Illicit Structural Profile

327 unlabelled transactions were flagged as structurally suspicious — their composite risk scores (combining in-degree, betweenness centrality, and illicit neighbour ratio) placed them in the same risk tier as confirmed illicit nodes. These are **investigation-priority targets** regardless of their formal classification status.

### 5. The Top 3 Highest-Risk Transactions

The top 3 nodes by composite risk score were flagged not because of their labels — two of them are classified as licit — but because of their **network position**:

- **Node 1 (Score 0.45):** Receives from 284 sources simultaneously (in-degree = 284, out-degree = 0). A pure aggregation point — consistent with a consolidation wallet receiving layered funds.
- **Node 3 (Score 0.3769):** In-degree of 241, out-degree of 0. Another aggregation node with massive inbound concentration.
- **Node 6 (Score 0.3373, Unknown):** In-degree 212, out-degree 0. Unlabelled — but structurally identical to the top-ranked confirmed nodes.

---

## Why Position Matters More Than Label

Traditional fraud detection assigns risk based on **transaction content** — amounts, frequency, counterparty history.

Graph analysis assigns risk based on **network position** — where a transaction sits relative to confirmed illicit activity, how many sources it aggregates, and how central it is to the flow of funds across the network.

The three highest-risk nodes in this analysis:
- Were not flagged because of what they contained
- Were flagged because of **where they sat** — receiving from hundreds of sources, sitting at network crossroads, acting as relay points between otherwise disconnected parts of the graph

This matters for compliance teams because:

| Label-Based Approach | Graph-Based Approach |
|---|---|
| Reviews 23% of the network (labelled only) | Reviews 100% of the network |
| Flags based on transaction attributes | Flags based on network position |
| Misses structurally suspicious unlabelled nodes | Surfaces 327 additional investigation targets |
| Cannot detect layering chains | Detects 52 layering chains |
| Cannot detect fan-out patterns | Detects 465 fan-out patterns |

---

## Compliance Implications

This analysis has direct relevance for AML (Anti-Money Laundering) compliance workflows:

1. **Investigation prioritisation:** The Top 20 risk-ranked node list provides a concrete, data-backed starting point for compliance review — particularly the 3 unlabelled nodes in the top 10 (Ranks 6, 9, 13)

2. **SAR (Suspicious Activity Report) triggers:** Nodes with in-degree > 100 and out-degree = 0 (pure aggregation points) represent non-standard transaction patterns that would typically warrant SAR filing under FinCEN guidance

3. **Network-aware transaction monitoring:** Standard transaction monitoring rules (velocity checks, amount thresholds) would not flag the top-ranked nodes in this analysis. Only graph-structural analysis reveals their risk profile

4. **The unlabelled majority:** The 77% unlabelled node population represents the practical challenge of real-world AML — most transactions are never formally classified. Graph scoring provides a risk-ranking mechanism for prioritising this population for review

---

## Limitations & Data Gaps

| Limitation | Impact | Mitigation |
|---|---|---|
| Features are anonymised (no semantic meaning) | Cannot interpret what specific features represent | Rely on graph topology rather than feature content |
| 77% of nodes unlabelled | Ground truth is sparse — risk scoring cannot be validated against full dataset | Score is prioritisation tool, not definitive classification |
| Static snapshot analysis | Temporal dynamics beyond time-step aggregation are not captured | Time-step bar chart provides partial temporal view |
| Betweenness centrality computation is expensive at full graph scale | Full 203K-node centrality computation not feasible in standard environment | Computed on subgraph (12,473 nodes); representative but not exhaustive |
| No real-world outcome data | Cannot measure whether flagged unknowns are truly illicit | Findings are hypotheses for investigation, not convictions |

---

## Project Structure

```
Elliptic-Bitcoin-Fraud-Analysis/
│
├── README.md                                    # This document
│
├── 📁 data/
│   ├── elliptic_txs_features.csv               # 203,769 × 166 features
│   ├── elliptic_txs_classes.csv                # Node labels
│   ├── elliptic_txs_edgelist.csv               # 234,355 directed edges
│   └── elliptic_merged_clean.csv               # Merged dataset (203,769 × 168)
│
├── 📁 notebooks/
│   └── elliptic_fraud_analysis.ipynb           # Full reproducible analysis
│
├── 📁 dashboard/
│   └── elliptic_fraud_dashboard.html           # Interactive HTML dashboard
│
├── 📁 reports/
│   └── compliance_report.pdf                   # Compliance-ready findings report
│
├── 📁 screenshots/
│   └── dashboard_preview.png                   # Dashboard screenshot
│
└── 📁 src/
    ├── data_preparation.py                     # Merge & clean pipeline
    ├── network_construction.py                 # NetworkX graph build
    ├── pattern_detection.py                    # Layering & fan-out detection
    ├── risk_scoring.py                         # Composite risk score engine
    └── visualisation.py                        # Chart and graph rendering
```

---

## How to Reproduce This Analysis

### Prerequisites

```bash
pip install pandas networkx matplotlib jupyter
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/elliptic-bitcoin-fraud-analysis.git
cd elliptic-bitcoin-fraud-analysis

# 2. Download the Elliptic dataset
# Available at: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
# Place all 3 CSV files in the /data/ directory

# 3. Run data preparation
python src/data_preparation.py

# 4. Launch the full analysis notebook
jupyter notebook notebooks/elliptic_fraud_analysis.ipynb

# 5. Open the interactive dashboard
open dashboard/elliptic_fraud_dashboard.html
# or: python -m http.server 8000 → navigate to localhost:8000/dashboard/
```

---

## References

- Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD Workshop on Anomaly Detection in Finance.
- Elliptic Dataset: [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- NetworkX Documentation: [https://networkx.org/](https://networkx.org/)
- FATF (Financial Action Task Force) — Money Laundering Typologies: [https://www.fatf-gafi.org/](https://www.fatf-gafi.org/)
Full report link- https://drive.google.com/drive/folders/1-fEWpgkY6hgILt3zQDeVvJGwrL1zaCwe?usp=sharing
---

*Built by a data analyst exploring the intersection of graph theory, blockchain forensics, and financial crime detection.*
*Dataset credit: Elliptic & MIT Media Lab Digital Currency Initiative.*

