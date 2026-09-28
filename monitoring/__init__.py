"""Monitoring subsystem.

Public API:
    from monitoring.alerting import AlertPayload, write_alert
    from monitoring.drift import psi, ks_statistic, drift_report
    from monitoring.forecast_monitor import monitor_demand
    from monitoring.supplier_monitor import monitor_supplier
    from monitoring.inventory_monitor import monitor_inventory

All monitors return a `MonitorReport` and optionally write AlertPayloads.
"""

from monitoring.alerting import AlertPayload, MonitorReport, write_alert, write_alerts
from monitoring.drift import drift_report, ks_statistic, psi, psi_severity
from monitoring.forecast_monitor import monitor_demand
from monitoring.inventory_monitor import monitor_inventory
from monitoring.supplier_monitor import monitor_supplier

__all__ = [
    "AlertPayload",
    "MonitorReport",
    "write_alert",
    "write_alerts",
    "psi",
    "psi_severity",
    "ks_statistic",
    "drift_report",
    "monitor_demand",
    "monitor_supplier",
    "monitor_inventory",
]