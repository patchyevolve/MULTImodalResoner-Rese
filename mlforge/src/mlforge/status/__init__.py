"""Status & telemetry plane — build step 7 (pending).

Normative: 13_product_specification.md §9.

Core invariant: STATUS DOWN → TRAINING CONTINUES.
Liveness rule: state.json never proves a live process — only a fresh
supervisor heartbeat does (12 §16.1 LIVE STATE).
Layered status: L1 overview / L2 training detail / L3 hardware telemetry.
"""
