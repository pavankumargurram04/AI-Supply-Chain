# AI Supply Chain Enterprise

An integrated, reproducible, explainable **AI supply-chain decision-intelligence platform**:

- Demand forecasting (global LSTM)
- Supplier risk (XGBoost + SHAP)
- Probabilistic inventory / stockout simulation
- Deep Q-Network inventory policy
- Procurement optimisation (MILP)
- Multi-agent orchestration
- Model registry, monitoring, drift detection
- Scenario lab, human approval, full audit trail
- Enterprise React frontend

This project is the master repository for a final-year B.Tech project and is designed
to run for a full academic year, producing a portfolio-grade system and a
conference / journal manuscript backed by real experiments.

---

## 1. Quick start (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env

docker compose up -d

python scripts/bootstrap_database.py

pytest -q