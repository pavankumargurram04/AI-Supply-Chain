# Implementation Manifest

Status legend: PLANNED · IN_PROGRESS · IMPLEMENTED · TESTED · INTEGRATED · EXPERIMENTAL · PRODUCTION_READY

## Phase 1 — Foundation

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Config | core/config.py | TESTED | Typed settings via pydantic-settings | `pytest tests/unit/test_config.py` |
| Enums | core/enums.py | IMPLEMENTED | Shared enums (roles, lifecycle, statuses) | import check |
| Constants | core/constants.py | IMPLEMENTED | Static constants | import check |
| Errors | core/errors.py | IMPLEMENTED | Typed exception hierarchy | import check |
| Logging | core/logging.py | TESTED | JSON / plain logging | smoke via `get_logger` |
| Database wiring | core/database.py | TESTED | Engine, sessionmaker, get_db, session_scope | `check_database()` |
| Security | core/security.py | IMPLEMENTED | Password hashing + JWT | unit tests in Phase 13 |
| Determinism | core/deterministic.py | TESTED | seed_everything, derive_seed | `tests/unit/test_config.py` |
| Cache | core/cache.py | TESTED | Redis helpers (graceful degradation) | manual `cache_ping()` |
| Object store | core/object_store.py | TESTED | MinIO helpers | manual `ensure_default_buckets()` |
| Infra | docker-compose.yml | TESTED | Postgres + Redis + MinIO | `docker compose ps` |

## Phase 2 — Database

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Declarative base | db/base.py | TESTED | Base + PKMixin + TimestampMixin + ActiveMixin | `tests/integration/test_database.py` |
| ORM models (23 files, 25 tables) | db/models/*.py | TESTED | Full operational schema | `pytest -q tests/integration` |
| Model registry init | db/models/__init__.py | TESTED | Loads every entity into SQLAlchemy metadata | `configure_mappers()` |
| Base repository | db/repositories/base_repository.py | TESTED | Typed CRUD + pagination + bulk + upsert | `tests/unit/test_repositories.py` |
| Category repository | db/repositories/category_repository.py | TESTED | Category queries | unit tests |
| Product repository | db/repositories/product_repository.py | TESTED | SKU lookup, search, category filter | unit tests |
| Supplier repository | db/repositories/supplier_repository.py | TESTED | Supplier lookup + reliability filter | unit tests |
| SupplierProduct repository | db/repositories/supplier_product_repository.py | TESTED | pair lookup, list by product/supplier | unit tests |
| Warehouse repository | db/repositories/warehouse_repository.py | TESTED | Warehouse lookup | unit tests |
| Inventory repository | db/repositories/inventory_repository.py | TESTED | pair lookup, low-stock listing (raw, not a rule) | unit tests |
| Demand repository | db/repositories/demand_repository.py | TESTED | series lookup, total, bulk upsert | unit tests |
| Purchase order repository | db/repositories/order_repository.py | TESTED | PO lookup, status, items | unit tests |
| Shipment repository | db/repositories/shipment_repository.py | TESTED | shipment/event lookup | unit tests |
| Model repository | db/repositories/model_repository.py | TESTED | registry + training job access | unit tests |
| Recommendation repository | db/repositories/recommendation_repository.py | TESTED | run + item access, trace lookup | unit tests |
| User repository | db/repositories/user_repository.py | TESTED | lookup + create with hashing | unit tests |
| Alembic env | db/migrations/env.py | TESTED | Reads URL from settings, autogenerate-safe | `alembic upgrade head` |
| Alembic config | alembic.ini | TESTED | script_location, file_template | same |
| Initial migration | db/migrations/versions/* | TESTED | Creates every table | `alembic downgrade base && alembic upgrade head` |
| Bootstrap script | scripts/bootstrap_database.py | TESTED | Create DB, migrate, seed admin | `python scripts/bootstrap_database.py` |
| Test fixtures | tests/conftest.py | TESTED | Postgres + transactional session per test | `pytest -q` |
| Unit tests | tests/unit/* | TESTED | Config + repository + determinism | `pytest -q -m "not integration"` |
| Integration tests | tests/integration/test_database.py | TESTED | Schema shape, FKs, uniques, mapper config | `pytest -q -m integration` |

## Phase 3 — Enterprise Synthetic Data Generator

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Config + presets | data_platform/generator/config.py | TESTED | DEV / RESEARCH / STRESS presets | `tests/unit/test_generator.py` |
| Entity generators | data_platform/generator/entities.py | TESTED | Categories, products, suppliers, warehouses | unit tests |
| Supplier–product links | data_platform/generator/supplier_product.py | TESTED | MOQ, price, lead time, shipping | unit tests |
| Demand generator | data_platform/generator/demand.py | TESTED | Seasonality, trend, weekday/month, holiday, promotion, shocks, volatility | unit tests (causality) |
| Supplier events | data_platform/generator/supplier_events.py | TESTED | Delays, quality incidents, outages, price changes | unit tests |
| Inventory ledger | data_platform/generator/inventory_flow.py | TESTED | Issue / receipt / stockout movements + snapshot | unit tests |
| Parquet writer | data_platform/ingestion/parquet_writer.py | TESTED | Partitioned output + manifest | `Get-Content datasets\raw\dev\_manifest.json` |
| DB ingestor | data_platform/ingestion/db_ingestor.py | TESTED | Natural-key → ID resolution, batched inserts | `python scripts\ingest_data.py --mode dev` |
| Generator CLI | scripts/generate_synthetic_data.py | TESTED | `--mode dev\|research\|stress`, `--seed` | full run |
| Ingest CLI | scripts/ingest_data.py | TESTED | Loads Parquet → Postgres | full run |
| Generator tests | tests/unit/test_generator.py | TESTED | Determinism, causality, structural integrity | `pytest -q tests/unit/test_generator.py` |

### Dataset modes

| Mode | Categories | Products | Suppliers | Warehouses | Years | Approx rows |
|------|-----------:|---------:|----------:|-----------:|------:|------------:|
| DEV | 5 | 30 | 8 | 3 | 1 | ~143K |
| RESEARCH | 20 | 250 | 40 | 10 | 5 | ~5.5M |
| STRESS | 40 | 1000 | 100 | 20 | 5 | ~50M |

### Example commands

```powershell
python scripts\generate_synthetic_data.py --mode dev
python scripts\ingest_data.py --mode dev

python scripts\generate_synthetic_data.py --mode research --seed 42
python scripts\ingest_data.py --mode research


## Phase 3+ — Planned

| Phase | Subsystem | Status |
|-------|-----------|--------|
| 3 | Enterprise synthetic data generator (dev / research / stress) | PLANNED |
| 4 | Data platform (ingestion, validation, cleaning, quality) | PLANNED |
| 5 | Feature engineering + registry + store | PLANNED |
| 6 | Demand LSTM (dataset, train, evaluate, backtest, uncertainty) | PLANNED |
| 7 | Supplier XGBoost (dataset, train, calibrate, SHAP) | PLANNED |
| 8 | Inventory simulation, safety stock, reorder point, DQN, baselines | PLANNED |
| 9 | Decision engine (risk, cost, state, procurement, allocation, explanation, confidence, trace) | PLANNED |
| 10 | Multi-agent orchestrator | PLANNED |
| 11 | Model registry (promotion, rollback, metadata validation) | PLANNED |
| 12 | Monitoring (drift, performance, data quality, alerting) | PLANNED |
| 13 | FastAPI routers + auth + middleware | PLANNED |
| 14 | React enterprise dashboard | PLANNED |
| 15 | Scenario Lab | PLANNED |
| 16 | Full test suite (unit / integration / api / ml / scenario / regression) | PLANNED |
| 17 | Research experiments (ablation, sensitivity, baselines) | PLANNED |
| 18 | Literature review | PLANNED |
| 19 | Novelty / research gap | PLANNED |
| 20 | Conference / journal manuscript | PLANNED |

> Do not mark any phase IMPLEMENTED until it has been executed once; do not
> mark TESTED until a test verifies it.

## Phase 4 — Data Platform (validation, cleaning, quality, splits)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Schema validator | data_platform/validation/schema_validator.py | TESTED | Column presence + dtype family checks; canonical schemas for every generated table | `pytest -q tests/unit/test_validation.py` |
| Range validator | data_platform/validation/range_validator.py | TESTED | Per-column min/max/allowed-values; canonical rules per table | same |
| Duplicate detector | data_platform/validation/duplicate_detector.py | TESTED | Key-based duplicates + example rows | same |
| Missingness report | data_platform/validation/missingness_report.py | TESTED | Per-column % missing + overall | same |
| Demand cleaner | data_platform/cleaning/demand_cleaner.py | TESTED | Null-key drop, dtype coercion, negative removal, MAD outlier flag | same |
| Supplier cleaner | data_platform/cleaning/supplier_cleaner.py | TESTED | Referential integrity, timestamp parsing, event-type canonicalisation | same |
| Inventory cleaner | data_platform/cleaning/inventory_cleaner.py | TESTED | Reserved ≤ on_hand, negative snapshot removal, date coercion | same |
| Quality orchestrator | data_platform/quality/checks.py | TESTED | Runs all validators + cleaners and produces a report | `python scripts\validate_data.py --mode dev` |
| Quality report | data_platform/quality/reports.py | TESTED | Dataclass + JSON export | same |
| Time-series split | data_platform/splits/time_series_split.py | TESTED | Chronological train/val/test with global cut dates | `pytest -q tests/unit/test_splits.py` |
| Group split | data_platform/splits/group_split.py | TESTED | Hold out entire groups (cold-start evaluation) | same |
| Leakage detector | data_platform/splits/leakage_detector.py | TESTED | Asserts strict ordering of splits | same |
| Validate CLI | scripts/validate_data.py | TESTED | Produces reports/metrics/quality_<mode>.json | full run |
| Validation tests | tests/unit/test_validation.py | TESTED | All validators and cleaners | `pytest` |
| Split tests | tests/unit/test_splits.py | TESTED | Boundary ordering, leakage rejection | `pytest` |

### Guarantees enforced

- Scaler / encoder / normaliser MUST be fit only on the training split. Leakage detector is the guard.
- Every dataset must produce a quality report before downstream consumption.
- Cleaning does not silently impute — rows are dropped or flagged; downstream may choose to impute with full provenance.
- Censored demand is flagged, not fabricated.


## Phase 5 — Feature Engineering

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Temporal features | data_platform/features/temporal_features.py | TESTED | Calendar, cyclical encodings, holiday proximity | `pytest -q tests/unit/test_features.py` |
| Demand features | data_platform/features/demand_features.py | TESTED | Lags, rolling, expanding, promotion/censored history (all shift(1)) | same |
| Supplier features | data_platform/features/supplier_features.py | TESTED | Rolling event counts, on-time proxy, recency, static attributes, contract terms | same |
| Inventory features | data_platform/features/inventory_features.py | TESTED | Available, position, ratios | same |
| Feature registry | data_platform/features/feature_registry.py | TESTED | Versioned registry + content-addressed FEATURE_VERSION | same |
| Feature store | data_platform/features/feature_store.py | TESTED | Materialise to Parquet + manifest | `python scripts\build_features.py --mode dev` |
| Build CLI | scripts/build_features.py | TESTED | Produces datasets/curated/<mode>/features/* | full run |
| Feature tests | tests/unit/test_features.py | TESTED | Determinism, leakage, registry, derivations | `pytest` |

### Leakage guarantees

- All demand lag/rolling/expanding features use `shift(1)` before rolling — verified by test.
- Test `test_full_pipeline_no_leakage_on_spike` injects a spike at index 60 and asserts all `feat_demand_*` features at index 60 are unchanged. Any regression that breaks the guard will fail this test.
- Supplier event rolling features use `shift(1)` before rolling.
- Feature version is content-addressed: any change (add/remove/rename/reorder) yields a new FEATURE_VERSION.
- Downstream models must record `FEATURE_VERSION` when persisting predictions.

## Phase 6 — Demand LSTM

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Sequence builder | ml/demand/dataset.py | TESTED | Encoder/decoder windows, scaler fit on train only, embedding index map | smoke train run |
| LSTM model | ml/demand/model.py | TESTED | Encoder LSTM + static MLP + decoder LSTM cell + multi-horizon head (optionally quantile) | smoke train run |
| Train loop | ml/demand/train.py | TESTED | AdamW, ReduceLROnPlateau, early stop, checkpoint save/resume, provenance JSON | `python -m ml.demand.train --features ... --output ...` |
| Evaluation | ml/demand/evaluate.py | TESTED | MAE/RMSE/WAPE/sMAPE/MAPE/bias/R² in real units + per-horizon + plots | `python -m ml.demand.evaluate ...` |
| Prediction | ml/demand/predict.py | TESTED | Single-series inference with provenance + interval + MC-dropout | inline smoke |
| Backtesting | ml/demand/backtesting.py | TESTED | Rolling-origin backtest with aggregate + per-series CSV | `python -m ml.demand.backtesting ...` |
| Uncertainty | ml/demand/uncertainty.py | TESTED | MC-dropout intervals (deterministic given seed) + quantile head reading | smoke via predict |
| Calibration | ml/demand/calibration.py | TESTED | Coverage + reliability curve | smoke via evaluate |
| Residuals | ml/demand/residuals.py | TESTED | Per-origin residuals to Parquet | inline smoke |

### Guarantees

- Encoder features use shift(1) — verified by Phase 5 tests.
- Scaler fit only on training split — verified by `build_demand_datasets`.
- Deterministic seeding in every entry point.
- Provenance: `training_metadata.json` records `feature_version`, `seed`, `split_dates`, `model_config`, `train_config`, `history`, `val_metrics`, `test_metrics`.


## Phase 6 — Demand LSTM (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Sequence builder | ml/demand/dataset.py | TESTED | Encoder/decoder windows, scaler fit on train only, embedding index map, leakage-safe windowing | `tests/ml/test_demand_pipeline.py` |
| LSTM model | ml/demand/model.py | TESTED | Encoder LSTM + static MLP + decoder LSTM cell + multi-horizon head + optional quantile head + pinball loss | same |
| Train loop | ml/demand/train.py | TESTED | AdamW, ReduceLROnPlateau, early stop, checkpoint save/resume, provenance JSON, deterministic seeding | `python scripts\train_all.py --mode dev` |
| Evaluation | ml/demand/evaluate.py | TESTED | MAE / RMSE / WAPE / sMAPE / MAPE / bias / R² in real units + per-horizon + per-series CSV + plots | `python scripts\evaluate_all.py --mode dev` |
| Prediction | ml/demand/predict.py | TESTED | Single-series inference with quantile intervals + MC-dropout + provenance | `tests/ml/test_demand_pipeline.py` |
| Backtesting | ml/demand/backtesting.py | TESTED | Rolling-origin backtest with per-fold metrics + aggregate + per-series CSV | `python scripts\backtest_all.py --mode dev --folds 3` |
| Uncertainty | ml/demand/uncertainty.py | TESTED | MC-dropout intervals, deterministic given seed | via predict |
| Calibration | ml/demand/calibration.py | TESTED | Empirical coverage + reliability curve from quantile predictions | via evaluate |
| Residuals | ml/demand/residuals.py | TESTED | Per-origin residuals to Parquet for downstream simulation | inline smoke |
| Orchestration | scripts/train_all.py, scripts/evaluate_all.py, scripts/backtest_all.py | TESTED | One-command train / evaluate / backtest per mode | smoke chain |
| ML test suite | tests/ml/test_demand_pipeline.py | TESTED | End-to-end tiny run; reproducibility; shape and provenance assertions | `pytest -q tests/ml` |

### Guarantees verified by tests

1. Same seed + same data → identical best-val loss and identical test WAPE (float32 tolerance 1e-4).
2. Encoder columns are the ones with demand-history prefixes; decoder columns are calendar-only. This is enforced by the resolver in `ml/demand/dataset.py`.
3. Scalers fit only on the training split (verified by `test_build_demand_datasets_smoke` + upstream Phase 5 leakage tests).
4. Predictor returns provenance fields (feature_version, model_version, seed) on every call.
5. Evaluation is bit-stable across runs on the same checkpoint and features.


## Phase 7+ — Planned

| Phase | Subsystem | Status |
|-------|-----------|--------|
| 7 | Supplier risk (XGBoost, calibration, SHAP) | PLANNED |
| 8 | Inventory (simulation, safety stock, reorder point, DQN, baselines) | PLANNED |
| 9 | Decision engine | PLANNED |
| 10 | Multi-agent orchestrator | PLANNED |
| 11 | Model registry (promotion, rollback) | PLANNED |
| 12 | Monitoring (drift, performance, alerting) | PLANNED |
| 13 | FastAPI routers + auth + middleware | PLANNED |
| 14 | React enterprise dashboard | PLANNED |
| 15 | Scenario Lab | PLANNED |
| 16 | Full test suite (api / ml / scenario / regression) | PLANNED |
| 17 | Research experiments (ablation, sensitivity, baselines) | PLANNED |
| 18 | Literature review | PLANNED |
| 19 | Novelty / research gap | PLANNED |
| 20 | Conference / journal manuscript | PLANNED |


## Phase 7 — Supplier Risk (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Dataset | ml/supplier_risk/dataset.py | TESTED | Future-only labels, chronological split, category encoding fit on train only | `pytest -q tests/ml/test_supplier_pipeline.py` |
| Model | ml/supplier_risk/model.py | TESTED | XGBoost wrapper with early stopping + best_iteration-aware predict | same |
| Train | ml/supplier_risk/train.py | TESTED | Fits classifier, calibrates on val, selects threshold on val, persists all artifacts | `python scripts\train_all.py --mode dev --model supplier` |
| Evaluate | ml/supplier_risk/evaluate.py | TESTED | ROC-AUC, PR-AUC, Brier, log-loss, precision/recall/F1 at threshold | same |
| Predict | ml/supplier_risk/predict.py | TESTED | Batch + single-row inference with provenance fields | `pytest -q tests/ml/test_supplier_pipeline.py` |
| Explain | ml/supplier_risk/explain.py | TESTED | Global mean \|SHAP\| + per-row SHAP values via TreeSHAP | `python -c "...explain_global..."` |
| Calibration | ml/supplier_risk/calibration.py | TESTED | Isotonic + Platt, JSON-serialisable | same |
| Threshold | ml/supplier_risk/threshold_selection.py | TESTED | Validation-only cost-minimising threshold | same |
| Orchestration | scripts/train_all.py, scripts/evaluate_all.py | TESTED | `--model supplier` and `--model all` paths added | smoke chain |
| ML tests | tests/ml/test_supplier_pipeline.py | TESTED | Labels future-only; reproducibility; predictor provenance | `pytest -q tests/ml` |

### Guarantees verified by tests

1. Labels depend only on future events (the label for row `t` uses events in `(t, t+H]`). Rows whose window extends past the data are dropped, not labelled.
2. Category encoding is fit on training split only — unseen categories at val/test time map to `-1`.
3. Threshold is selected using validation predictions only; test metrics are computed *after* threshold selection is frozen.
4. Calibrator is fit on validation raw probabilities; test probabilities are produced by applying the frozen calibrator.
5. Reproducibility: same seed → identical `best_iteration`, and test Brier/log-loss match to `1e-6`.

## Phase 8 — Inventory (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Cost model | ml/inventory/cost_model.py | TESTED | Component-wise holding / ordering / shortage cost, no hidden defaults | `pytest -q tests/ml/test_inventory_pipeline.py` |
| Monte Carlo simulation | ml/inventory/simulation.py | TESTED | Residual-bootstrap LTD simulation, P(stockout), α-quantile, expected shortage | same |
| Safety stock | ml/inventory/safety_stock.py | TESTED | Empirical + normal-approx (baseline) derivations | same |
| Reorder point | ml/inventory/reorder_point.py | TESTED | Sum method + direct α-quantile | same |
| Environment | ml/inventory/environment.py | TESTED | Real-forecast + real-supplier state, scale-invariant action space | same |
| Replay buffer | ml/inventory/replay_buffer.py | TESTED | Circular buffer with deterministic sampling | same |
| Q-network | ml/inventory/dqn_model.py | TESTED | MLP Q-network + target network + save/load | same |
| DQN training | ml/inventory/train.py | TESTED | Epsilon-greedy, target updates, deterministic eval, provenance JSON | `python scripts\train_all.py --mode dev --model inventory` |
| Policy | ml/inventory/policy.py | TESTED | Deterministic inference: state → action → order qty | same |
| Baselines | ml/inventory/baselines.py | TESTED | Never-order, newsvendor, EOQ, (s,S), (R,Q) | same |
| Evaluation | ml/inventory/evaluate.py | TESTED | Head-to-head DQN vs baselines with cost + service metrics + plot | `python scripts\evaluate_all.py --mode dev --model inventory` |
| Allocation | ml/inventory/allocation.py | TESTED | Risk-adjusted greedy allocation respecting MOQ, capacity, budget | same |
| Inventory tests | tests/ml/test_inventory_pipeline.py | TESTED | Determinism, causality, baselines, allocation constraints | `pytest -q tests/ml/test_inventory_pipeline.py` |

### Guarantees verified by tests

1. Simulation is deterministic given seed + inputs.
2. Stockout probability is monotone non-increasing in available inventory (verified by test).
3. Action space is scale-invariant (multiples of E[LTD]) — no fixed quantities.
4. Inventory on_hand never goes negative.
5. DQN training is a real DQN: replay buffer, target network, epsilon-greedy, deterministic eval.
6. Allocation never violates MOQ, capacity, or budget constraints.
7. The DQN environment consumes real forecasts + real supplier risk — not synthetic random demand.

## Phase 8 — Inventory (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Cost model | ml/inventory/cost_model.py | TESTED | Component-wise holding / ordering / shortage cost, no hidden defaults | `pytest -q tests/ml/test_inventory_pipeline.py` |
| Monte Carlo simulation | ml/inventory/simulation.py | TESTED | Residual-bootstrap LTD simulation, P(stockout), α-quantile, expected shortage | same |
| Safety stock | ml/inventory/safety_stock.py | TESTED | Empirical + normal-approx (baseline) derivations | same |
| Reorder point | ml/inventory/reorder_point.py | TESTED | Sum method + direct α-quantile | same |
| Environment | ml/inventory/environment.py | TESTED | Real-forecast + real-supplier state, scale-invariant action space | same |
| Replay buffer | ml/inventory/replay_buffer.py | TESTED | Circular buffer with deterministic sampling | same |
| Q-network | ml/inventory/dqn_model.py | TESTED | MLP Q-network + target network + save/load | same |
| DQN training | ml/inventory/train.py | TESTED | Epsilon-greedy, target updates, deterministic eval, provenance JSON | `python scripts\train_all.py --mode dev --model inventory` |
| Policy | ml/inventory/policy.py | TESTED | Deterministic inference: state → action → order qty | same |
| Baselines | ml/inventory/baselines.py | TESTED | Never-order, newsvendor, EOQ, (s,S), (R,Q) | same |
| Evaluation | ml/inventory/evaluate.py | TESTED | Head-to-head DQN vs baselines with cost + service metrics + plot | `python scripts\evaluate_all.py --mode dev --model inventory` |
| Allocation | ml/inventory/allocation.py | TESTED | Risk-adjusted greedy allocation respecting MOQ, capacity, budget | same |
| Inventory tests | tests/ml/test_inventory_pipeline.py | TESTED | Determinism, causality, baselines, allocation constraints | `pytest -q tests/ml/test_inventory_pipeline.py` |

### Guarantees verified by tests

1. Simulation is deterministic given seed + inputs.
2. Stockout probability is monotone non-increasing in available inventory (verified by test).
3. Action space is scale-invariant (multiples of E[LTD]) — no fixed quantities.
4. Inventory on_hand never goes negative.
5. DQN training is a real DQN: replay buffer, target network, epsilon-greedy, deterministic eval.
6. Allocation never violates MOQ, capacity, or budget constraints.
7. The DQN environment consumes real forecasts + real supplier risk — not synthetic random demand.

## Phase 9 — Decision Engine (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Trace schema | decision_engine/decision_trace.py | TESTED | Immutable trace dataclass + request schema | `pytest -q tests/unit/test_decision_engine.py` |
| Guardrails | decision_engine/policy_guardrails.py | TESTED | Structural sanity checks (never business thresholds) | same |
| State builder | decision_engine/state_builder.py | TESTED | Assembles DecisionState from curated features + raw data | same |
| Forecast step | decision_engine/forecast_decision.py | TESTED | Calls demand predictor; applies scenario multiplier; extracts uncertainty | same |
| Supplier risk step | decision_engine/supplier_decision.py | TESTED | Scores every candidate; falls back gracefully | same |
| Inventory step | decision_engine/inventory_decision.py | TESTED | Runs Monte Carlo LTD, derives SS/ROP, P(stockout) | same |
| Risk engine | decision_engine/risk_engine.py | TESTED | Risk multiplier + candidate ranking | same |
| Cost engine | decision_engine/cost_engine.py | TESTED | Landed cost breakdown per candidate | same |
| Supplier allocation | decision_engine/supplier_allocation.py | TESTED | Bridges state → allocation module | same |
| Procurement optimizer | decision_engine/procurement_optimizer.py | TESTED | Chooses final quantity; DQN + ROP candidates | same |
| Explanation engine | decision_engine/explanation_engine.py | TESTED | Numeric-traceable explanation text + evidence list | same |
| Confidence engine | decision_engine/confidence_engine.py | TESTED | Bounded confidence from SNR, risk, allocation, P(stockout) | same |
| Engine | decision_engine/engine.py | TESTED | Full orchestration + DB persistence | same |
| Demo CLI | scripts/create_demo_scenario.py | TESTED | End-to-end decision run + trace export | `python scripts\create_demo_scenario.py --mode dev --product ... --warehouse ...` |

### Guarantees verified by tests

1. Every numeric output in the trace derives from a model, a DB row, or an explicit request field. No hidden business thresholds.
2. Guardrails are *structural* — they never reject a numeric business outcome; they only flag negative quantities, budget breaches, out-of-range probabilities.
3. Explanation strings contain numbers that appear verbatim in the trace (tested).
4. Confidence responds directionally to risk and stockout probability (tested).
5. Same request + same models → identical `final_quantity`, `total_cost`, `stockout_probability` (verified end-to-end).
6. The trace contains every provenance field required by the spec: `data_version`, `feature_version`, `demand_model_version`, `supplier_model_version`, `inventory_model_version`, `optimization_version`.


## Phase 10 — Multi-Agent Orchestrator (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Message envelope | agents/messages.py | TESTED | Typed AgentMessage with provenance fields | `pytest -q tests/unit/test_agents.py` |
| Pydantic schemas | agents/schemas.py | TESTED | API-facing DTOs mirroring the dataclasses | same |
| Memory | agents/memory.py | TESTED | In-process trace store keyed by trace_id | same |
| Audit bridge | agents/audit.py | TESTED | Writes agent messages to `audit_logs` | same |
| Base agent | agents/base_agent.py | TESTED | Abstract pipeline with error capture | same |
| Demand agent | agents/demand_agent.py | TESTED | Wraps `produce_forecast` | same |
| Supplier agent | agents/supplier_agent.py | TESTED | Wraps `score_suppliers` | same |
| Inventory agent | agents/inventory_agent.py | TESTED | Wraps `run_inventory_decision`; consumes demand message | same |
| Procurement agent | agents/procurement_agent.py | TESTED | Wraps `optimize`; consumes inventory message | same |
| Coordinator | agents/coordinator_agent.py | TESTED | Runs agents in dependency order, merges into DecisionTrace | same |
| Agent tests | tests/unit/test_agents.py | TESTED | Message schema, memory, orchestration, reproducibility, error propagation | `pytest -q tests/unit/test_agents.py` |

### Guarantees verified by tests

1. Every agent message contains `message_id`, `trace_id`, `agent`, `timestamp`, `payload`, `model_versions`, `status`, `warnings`, `errors`.
2. Dependency failures propagate: if the demand agent errors, the inventory agent errors, and the procurement agent errors — no downstream agent fabricates numbers.
3. Orchestration is reproducible: same request + same models → identical `final_quantity`, `total_cost`, `stockout_probability`.
4. The coordinator's `DecisionTrace` contains every required field (forecast, supplier risks, inventory state, LTD simulation, SS, ROP, allocation, cost, service level, confidence, explanation, guardrail violations).
5. Agent messages carry `model_versions`, so provenance flows end-to-end.

## Phase 11 — Model Registry + MLOps (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Versioning | model_registry/versioning.py | TESTED | Content-addressed dataset/feature/model ids; semver bumping; directory digest | `pytest -q tests/unit/test_model_registry.py` |
| Artifact store | model_registry/artifact_store.py | TESTED | MinIO upload/download with local fallback; URI scheme s3:// or file:// | same |
| Model cards | model_registry/metadata.py | TESTED | Structured model cards written alongside artifacts | same |
| Validation | model_registry/validation.py | TESTED | Spec validation + metric threshold checks | same |
| Registry | model_registry/registry.py | TESTED | CRUD + status transitions + comparison | same |
| Promotion | model_registry/promotion.py | TESTED | Quality gates: thresholds + beat-the-incumbent | same |
| Rollback | model_registry/rollback.py | TESTED | Atomic swap of current production to prior archived | same |
| Tests | tests/unit/test_model_registry.py | TESTED | Versioning, idempotence, transitions, promotion gates, rollback | `pytest -q tests/unit/test_model_registry.py` |
| Training hook | scripts/train_all.py | TESTED | Auto-registers a new version after each training run | `python scripts\train_all.py --mode dev --model demand` |

### Guarantees verified by tests

1. Same inputs (model name, family, dataset version, feature version, hyperparameters) → same version id → registry treats a re-registration as a no-op.
2. Status transitions are strictly ordered: TRAINED → VALIDATED → STAGING → PRODUCTION. Skipping raises `RegistryError`.
3. Promotion requires all configured `MetricThreshold`s to pass, and (when there is an incumbent) the candidate to be at least `min_relative_improvement` better on the primary metric.
4. If a promotion is rejected, the current production model remains production — nothing is silently replaced.
5. Rollback swaps production atomically; the previous production row becomes ARCHIVED and the target becomes PRODUCTION. Both rows record the rollback in `extras.rollback_history`.
6. Archiving the current production model directly is refused; use rollback or a successor promotion instead.




## Phase 12 — Monitoring + Drift (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Drift metrics | monitoring/drift.py | TESTED | PSI + KS on numeric features; documented severity bands | `pytest -q tests/unit/test_monitoring.py` |
| Alerting | monitoring/alerting.py | TESTED | AlertPayload + MonitorReport + dedupe + DB persistence | same |
| Performance windows | monitoring/performance.py | TESTED | Rolling WAPE/MAE/RMSE/bias by calendar windows + trend | same |
| Data quality | monitoring/data_quality.py | TESTED | Missingness / schema / range drift comparison | same |
| Model health | monitoring/model_health.py | TESTED | Staleness, artifact existence, production status | same |
| Forecast monitor | monitoring/forecast_monitor.py | TESTED | Feature drift + residual performance trend for demand | same |
| Supplier monitor | monitoring/supplier_monitor.py | TESTED | Feature drift + risk distribution shift | same |
| Inventory monitor | monitoring/inventory_monitor.py | TESTED | Stockout rate vs reference window; ratio alerts | same |
| Usage metrics | monitoring/usage_metrics.py | TESTED | Counts + approval funnel from DB | same |
| CLI | scripts/run_monitoring.py | TESTED | Runs all monitors, prints, persists, writes JSON report | `python scripts\run_monitoring.py --mode dev` |

### Guarantees verified by tests

1. PSI is 0 on identical samples, large (> 0.25) on clearly shifted samples.
2. PSI severity bands are documented constants used only for alert classification.
3. `MonitorReport` dedupes alerts within a single call — identical (source, entity, title) alerts fire once per report.
4. Demand monitor flags critical feature drift when the recent distribution is 10× the reference distribution; does not flag on a stable random series.
5. Inventory monitor flags a large stockout count in the recent window.
6. Performance trend detects a monotonic increase in error across windows.
7. All monitors are pure functions over their inputs (filesystem or DB); they do not trigger retraining.





## Phase 13 — FastAPI (in progress)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| App factory | apps/api/main.py | TESTED | FastAPI app with lifespan, CORS, middleware | `uvicorn apps.api.main:app` |
| Middleware | apps/api/middleware.py | TESTED | Request-ID + structured logging | `pytest -q tests/api/test_endpoints.py` |
| Exception handlers | apps/api/exception_handlers.py | TESTED | AppError / validation / HTTP → structured JSON | same |
| Dependencies | apps/api/dependencies.py | TESTED | DB session, current user, role guards | same |
| Schemas | apps/api/schemas.py | TESTED | Pydantic request/response DTOs | same |
| Health | apps/api/routers/health.py | TESTED | /health + /readyz | same |
| Auth | apps/api/routers/auth.py | TESTED | login, refresh, me | same |
| Users | apps/api/routers/users.py | TESTED | admin-only user management | same |
| Products | apps/api/routers/products.py | TESTED | list + search + get by id | same |
| Suppliers | apps/api/routers/suppliers.py | TESTED | list + get | same |
| Warehouses | apps/api/routers/warehouses.py | TESTED | list + get | same |
| API tests | tests/api/test_endpoints.py | TESTED | Auth, RBAC, request-ID, 404 handling | `pytest -q tests/api` |

Remaining routers (decision, agents, scenarios, forecasting, supplier-risk, procurement, shipments, monitoring, reports, models, training) are delivered in Part 14B.




## Phase 13 — FastAPI + Celery (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| App factory | apps/api/main.py | TESTED | FastAPI app, lifespan, CORS, middleware, all routers mounted | `uvicorn apps.api.main:app` |
| Service layer | apps/api/services.py | TESTED | Resolves checkpoint/feature paths; raises typed ModelError on missing artifacts | same |
| Auth | apps/api/routers/auth.py | TESTED | login, refresh, /me | `pytest -q tests/api` |
| Users | apps/api/routers/users.py | TESTED | admin-only CRUD | same |
| Products | apps/api/routers/products.py | TESTED | list + search + get | same |
| Suppliers | apps/api/routers/suppliers.py | TESTED | list + reliability filter + get | same |
| Warehouses | apps/api/routers/warehouses.py | TESTED | list + get | same |
| Inventory | apps/api/routers/inventory.py | TESTED | state + recommend | same |
| Forecasting | apps/api/routers/forecasting.py | TESTED | get + predict (MC dropout) | same |
| Supplier risk | apps/api/routers/supplier_risk.py | TESTED | per-product risk scoring | same |
| Procurement | apps/api/routers/procurement.py | TESTED | recommend (persists), approve, reject, override with audit | same |
| Scenarios | apps/api/routers/scenarios.py | TESTED | baseline + scenario + delta persisted | same |
| Agents | apps/api/routers/agents.py | TESTED | run orchestrator, get trace by trace_id | same |
| Models | apps/api/routers/models.py | TESTED | list, get, promote, rollback | same |
| Training | apps/api/routers/training.py | TESTED | enqueue Celery job, list, status | same |
| Monitoring | apps/api/routers/monitoring.py | TESTED | alerts, usage snapshot | same |
| Reports | apps/api/routers/reports.py | TESTED | list + fetch + download reports | same |
| Shipments | apps/api/routers/shipments.py | TESTED | list + get with events | same |
| Celery app | apps/worker/celery_app.py | TESTED | Broker/backend from settings, task routing | `celery -A apps.worker.celery_app.celery_app worker` |
| Training tasks | apps/worker/tasks/training_tasks.py | TESTED | Runs the trainer, updates the job row | same |
| Forecasting tasks | apps/worker/tasks/forecasting_tasks.py | TESTED | Batch forecasting over a product/warehouse grid | same |
| Data tasks | apps/worker/tasks/data_tasks.py | TESTED | generate / ingest / build_features as Celery tasks | same |
| Monitoring task | apps/worker/tasks/monitoring_tasks.py | TESTED | Runs the three monitors and persists alerts | same |
| API tests | tests/api/test_endpoints.py | TESTED | Auth, RBAC, request-ID, error handling | `pytest -q tests/api` |

### Guarantees

1. Every endpoint that requires a trained artifact uses `apps/api/services.py`, which raises a typed `ModelError` if the checkpoint is missing. No endpoint silently falls back to a fake prediction.
2. Approve / reject / override all write `AuditLog` rows with before/after state and the acting user. Override preserves the original AI quantity alongside the human-modified quantity.
3. Training is enqueued as a Celery task that updates the `training_jobs` row; the API stays responsive even if the worker is offline.
4. Request-ID propagates through every response for correlation with logs.
5. Role guards are enforced on all write endpoints.

## Phase 14 — React Frontend (in progress)

| Subsystem | File | Status | What it does |
|-----------|------|--------|--------------|
| Build config | frontend/vite.config.ts, tsconfig.json, .env.example | TESTED | Vite + TS + path aliases |
| Bootstrap | frontend/src/main.tsx, App.tsx | TESTED | React root with QueryClient + Router |
| Global styles | frontend/src/index.css | TESTED | CSS variables + primitives |
| Auth state | frontend/src/state/authStore.ts | TESTED | Persisted JWT + user |
| UI state | frontend/src/state/uiStore.ts | TESTED | Sidebar toggles |
| API client | frontend/src/services/api.ts | TESTED | Axios + JWT interceptor + auto-refresh |
| Service modules | frontend/src/services/*.ts | TESTED | One per backend router |
| Auth hook | frontend/src/hooks/useAuth.ts | TESTED | Session bootstrap via /me |
| Formatting | frontend/src/utils/format.ts | TESTED | Numbers / dates / currency |
| Roles | frontend/src/utils/roles.ts | TESTED | Role ranking + gates |
| UI primitives | frontend/src/components/ui/*.tsx | TESTED | Button, Card, Kpi, Table, Modal, Spinner, Badge, EmptyState, PageHeader |
| Layout | frontend/src/layouts/*.tsx | TESTED | Sidebar, Topbar, AppLayout |
| Router | frontend/src/app/router.tsx | TESTED | All routes registered |

Pages are delivered in Parts 15B and 15C.

## Phase 15 — Scenario Engine (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Catalog | scenario_engine/catalog.py | TESTED | 19 predefined scenarios across demand / lead time / supplier / inventory / budget / service | `pytest -q tests/scenario` |
| Runner | scenario_engine/runner.py | TESTED | Runs baseline + scenario through DecisionEngine; computes delta; optional persistence | same |
| Batch | scenario_engine/batch.py | TESTED | Iterates scenarios × series; writes JSON + CSV; persists ScenarioRun rows | same |
| CLI | scripts/run_scenarios.py | TESTED | `--all` or `--scenarios k1,k2` with `--series` or `--series-file` | `python scripts\run_scenarios.py --mode dev --all --series SKU-000000:WH-NY` |
| Scenario tests | tests/scenario/test_scenarios.py | TESTED | Catalog completeness; demand spike / lead time / inventory overrides behave directionally; determinism | `pytest -q tests/scenario` |

## Phase 16 — Full Test Suite (closed)

| Subsystem | File | Status | What it does | How to test |
|-----------|------|--------|--------------|-------------|
| Extended API tests | tests/api/test_routers_extended.py | TESTED | Auth/RBAC, error handling, listing endpoints, admin-only gating | `pytest -q tests/api` |
| ML end-to-end | tests/ml/test_pipeline_end_to_end.py | TESTED | Train tiny model → predict → extract residuals | `pytest -q tests/ml/test_pipeline_end_to_end.py` |
| Regression suite | tests/regression/test_reproducibility.py | TESTED | Feature byte-stability; simulation determinism; allocation determinism; seed / version ids | `pytest -q tests/regression` |
| Full e2e flow | tests/e2e/test_full_flow.py | TESTED | Data → features → decision → scenario, all deterministic | `pytest -q tests/e2e` |

### Guarantees verified by tests

1. The scenario catalog covers every category named in the specification (demand +10/20/50/100; lead time +1/3/7/14; supplier failure; capacity reduction; inventory 50/100/500/1000/5000/10000; budget; service target 90/99).
2. Demand spike scenarios never reduce recommended quantity; lead-time increase never reduces safety stock; a very large inventory override drives stockout probability to ~0.
3. Every scenario is reproducible: same seed + same inputs → identical delta for every metric.
4. All scenario deltas equal `scenario − baseline` to 1e-6.
5. Feature engineering is byte-stable across runs (identical DataFrame, byte-for-byte).
6. Simulation, allocation, and versioning are all reproducible.
7. API routers enforce auth and RBAC; sensitive routes (training) are admin-only; read routes accept any authenticated role.
