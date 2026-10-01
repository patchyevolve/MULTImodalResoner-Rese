"""Validation gate + preflight — build step 3 (pending).

Normative: 12_training_system.md §7 (BLOCK vs WARN, fail-closed),
§18 (resume flow, 19 steps), §8.4 (semantic-invariant matrix),
§17 (invariant table).

Rules to implement:
  * any required check fails → NO EXECUTION (exit 1), always
  * unknown compatibility → BLOCK, never GUESS
  * steps 15–16: acquire run lease, revalidate volatile subset (§23.3)
"""
