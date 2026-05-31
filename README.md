# Uncovering Fraud Networks in a Blockchain Payment System
Most fraud detection tools ask "is this transaction suspicious?" I asked a different question — "where does this transaction sit in the network?" The answer changed everything.
I just completed a blockchain fraud analysis project using the Elliptic Bitcoin Transaction Dataset — a real-world dataset used in published academic research — where I analysed over 203,000 Bitcoin transactions and 234,000 payment flows to map exactly how illicit funds move through a network, not just which transactions carry a fraud label.
What I found:
Working with a focused subgraph of 12,473 transactions, the analysis surfaced 300 confirmed illicit nodes, 52 layering patterns and 465 fan-out patterns — two of the core movement techniques associated with money laundering. More importantly, 327 unlabelled transactions showed identical structural behaviour to confirmed illicit ones, making them high-priority targets for further investigation regardless of their formal classification status.
The three highest-risk transactions were not flagged because of what they contained. They were flagged because of where they sat — receiving from hundreds of sources simultaneously, sitting at the busiest crossroads in the payment graph, acting as relay points between otherwise disconnected parts of the network.
The key takeaway:
In fraud detection, position matters as much as label. A transaction with no formal classification can still be deeply suspicious based purely on its network behaviour. 77% of this dataset carries no label — which means any compliance review that only looks at confirmed cases is working with less than a quarter of the real picture.
Deliverables:
🔗 Jupyter Notebook (full analysis)
🔗 Interactive HTML Dashboard
🔗 Compliance Report
🔗 Cleaned and Merged Dataset — 203,769 rows × 168 columns
Tools used: Python · NetworkX · Pandas · Matplotlib · Custom Canvas Dashboard
Dataset: Elliptic Bitcoin Transaction Dataset — Weber et al., 2019
HTML Interactive Dashboard Link: https://drive.google.com/file/d/12cTk7vul9aa7t2V_OOhPVcfJfGQNvJcu/view?usp=sharing
<img width="1920" height="949" alt="image" src="https://github.com/user-attachments/assets/e1065a12-ce6f-4f56-96e5-b5edf5147fa1" />
