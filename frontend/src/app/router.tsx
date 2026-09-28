import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "@/layouts/AppLayout";
import Login from "@/pages/Login";
import Dashboard from "@/pages/Dashboard";
import Products from "@/pages/Products";
import ProductDetail from "@/pages/ProductDetail";
import DemandForecast from "@/pages/DemandForecast";
import SupplierRisk from "@/pages/SupplierRisk";
import Inventory from "@/pages/Inventory";
import Procurement from "@/pages/Procurement";
import Orders from "@/pages/Orders";
import Shipments from "@/pages/Shipments";
import Warehouses from "@/pages/Warehouses";
import ScenarioLab from "@/pages/ScenarioLab";
import MultiAgent from "@/pages/MultiAgent";
import ModelCenter from "@/pages/ModelCenter";
import TrainingCenter from "@/pages/TrainingCenter";
import Monitoring from "@/pages/Monitoring";
import Reports from "@/pages/Reports";
import AuditTrail from "@/pages/AuditTrail";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<AppLayout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/products" element={<Products />} />
        <Route path="/products/:sku" element={<ProductDetail />} />
        <Route path="/demand" element={<DemandForecast />} />
        <Route path="/supplier-risk" element={<SupplierRisk />} />
        <Route path="/inventory" element={<Inventory />} />
        <Route path="/procurement" element={<Procurement />} />
        <Route path="/orders" element={<Orders />} />
        <Route path="/shipments" element={<Shipments />} />
        <Route path="/warehouses" element={<Warehouses />} />
        <Route path="/scenarios" element={<ScenarioLab />} />
        <Route path="/agents" element={<MultiAgent />} />
        <Route path="/models" element={<ModelCenter />} />
        <Route path="/training" element={<TrainingCenter />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/audit" element={<AuditTrail />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}