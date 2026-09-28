import { useEffect, useMemo, useState } from "react";

import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { PageHeader } from "@/components/ui/PageHeader";

import {
  getOrder,
  listOrders,
  type OrderDetails,
  type OrderListItem,
} from "@/services/orders";
import { listProducts, type Product } from "@/services/products";
import { listSuppliers, type Supplier } from "@/services/suppliers";
import { listWarehouses, type Warehouse } from "@/services/warehouses";

function formatDate(value: string | null) {
  if (!value) return "—";

  return new Date(value).toLocaleDateString();
}

function formatAmount(amount: number, currency: string) {
  return `${currency} ${amount.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function statusClass(status: string) {
  switch (status.toUpperCase()) {
    case "ORDERED":
      return "status-success";
    case "DRAFT":
      return "status-neutral";
    case "CANCELLED":
      return "status-danger";
    case "RECEIVED":
      return "status-success";
    default:
      return "status-neutral";
  }
}

export default function Orders() {
  const [orders, setOrders] = useState<OrderListItem[]>([]);
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [warehouses, setWarehouses] = useState<Warehouse[]>([]);
  const [products, setProducts] = useState<Product[]>([]);

  const [selectedOrder, setSelectedOrder] = useState<OrderDetails | null>(
    null,
  );

  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [orderRows, supplierRows, warehouseRows, productRows] =
          await Promise.all([
            listOrders(),
            listSuppliers(),
            listWarehouses(),
            listProducts(undefined, 500),
          ]);

        if (cancelled) return;

        setOrders(orderRows);
        setSuppliers(supplierRows);
        setWarehouses(warehouseRows);
        setProducts(productRows);
      } catch (err) {
        if (cancelled) return;

        console.error("Failed to load orders:", err);
        setError("Unable to load purchase orders.");
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const supplierMap = useMemo(
    () => new Map(suppliers.map((supplier) => [supplier.id, supplier])),
    [suppliers],
  );

  const warehouseMap = useMemo(
    () => new Map(warehouses.map((warehouse) => [warehouse.id, warehouse])),
    [warehouses],
  );

  const productMap = useMemo(
    () => new Map(products.map((product) => [product.id, product])),
    [products],
  );

  async function openOrder(orderId: number) {
    try {
      setDetailsLoading(true);
      setError(null);

      const details = await getOrder(orderId);
      setSelectedOrder(details);
    } catch (err) {
      console.error("Failed to load order details:", err);
      setError("Unable to load order details.");
    } finally {
      setDetailsLoading(false);
    }
  }

  function closeOrder() {
    setSelectedOrder(null);
  }

  if (loading) {
    return (
      <>
        <PageHeader
          title="Orders"
          subtitle="Purchase orders created from approved recommendations."
        />

        <Card>
          <div className="empty-state">
            <div className="empty-title">Loading purchase orders...</div>
            <div className="empty-hint">
              Fetching orders and related master data.
            </div>
          </div>
        </Card>
      </>
    );
  }

  if (error && orders.length === 0) {
    return (
      <>
        <PageHeader
          title="Orders"
          subtitle="Purchase orders created from approved recommendations."
        />

        <Card>
          <EmptyState
            title="Unable to load purchase orders."
            hint={error}
          />
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Orders"
        subtitle="Purchase orders created from approved recommendations."
      />

      <Card
        title="Purchase Orders"
        actions={
          <span className="text-sm text-muted-foreground">
            {orders.length} order{orders.length === 1 ? "" : "s"}
          </span>
        }
      >
        {orders.length === 0 ? (
          <EmptyState
            title="No purchase orders yet."
            hint="Purchase orders will appear here after an approved or overridden recommendation is converted into an order."
          />
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>PO Number</th>
                  <th>Supplier</th>
                  <th>Warehouse</th>
                  <th>Status</th>
                  <th>Total</th>
                  <th>Expected Delivery</th>
                  <th>Created</th>
                  <th />
                </tr>
              </thead>

              <tbody>
                {orders.map((order) => {
                  const supplier = supplierMap.get(order.supplier_id);
                  const warehouse = warehouseMap.get(order.warehouse_id);

                  return (
                    <tr key={order.id}>
                      <td>
                        <button
                          type="button"
                          className="link-button"
                          onClick={() => openOrder(order.id)}
                        >
                          {order.po_number}
                        </button>
                      </td>

                      <td>
                        {supplier ? (
                          <>
                            <div>{supplier.name}</div>
                            <div className="text-muted">
                              {supplier.code}
                            </div>
                          </>
                        ) : (
                          `Supplier #${order.supplier_id}`
                        )}
                      </td>

                      <td>
                        {warehouse ? (
                          <>
                            <div>{warehouse.name}</div>
                            <div className="text-muted">
                              {warehouse.code}
                            </div>
                          </>
                        ) : (
                          `Warehouse #${order.warehouse_id}`
                        )}
                      </td>

                      <td>
                        <span className={`status ${statusClass(order.status)}`}>
                          {order.status}
                        </span>
                      </td>

                      <td>
                        {formatAmount(
                          order.total_amount,
                          order.currency,
                        )}
                      </td>

                      <td>{formatDate(order.expected_delivery_date)}</td>

                      <td>{formatDate(order.created_at)}</td>

                      <td>
                        <button
                          type="button"
                          className="button button-secondary"
                          onClick={() => openOrder(order.id)}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {error && orders.length > 0 && (
        <Card>
          <div className="empty-hint">{error}</div>
        </Card>
      )}

      {selectedOrder && (
        <Card
          title={`Order ${selectedOrder.po_number}`}
          actions={
            <button
              type="button"
              className="button button-secondary"
              onClick={closeOrder}
            >
              Close
            </button>
          }
        >
          {detailsLoading ? (
            <div className="empty-state">
              <div className="empty-title">
                Loading order details...
              </div>
            </div>
          ) : (
            <>
              <div className="order-summary">
                <div>
                  <strong>Status</strong>
                  <div>{selectedOrder.status}</div>
                </div>

                <div>
                  <strong>Supplier</strong>
                  <div>
                    {supplierMap.get(selectedOrder.supplier_id)?.name ??
                      `Supplier #${selectedOrder.supplier_id}`}
                  </div>
                </div>

                <div>
                  <strong>Warehouse</strong>
                  <div>
                    {warehouseMap.get(selectedOrder.warehouse_id)?.name ??
                      `Warehouse #${selectedOrder.warehouse_id}`}
                  </div>
                </div>

                <div>
                  <strong>Total</strong>
                  <div>
                    {formatAmount(
                      selectedOrder.total_amount,
                      selectedOrder.currency,
                    )}
                  </div>
                </div>

                <div>
                  <strong>Expected Delivery</strong>
                  <div>
                    {formatDate(
                      selectedOrder.expected_delivery_date,
                    )}
                  </div>
                </div>

                <div>
                  <strong>Recommendation</strong>
                  <div>
                    {selectedOrder.recommendation_id ?? "—"}
                  </div>
                </div>
              </div>

              <h3 className="section-title">Order Items</h3>

              {selectedOrder.items.length === 0 ? (
                <EmptyState
                  title="No order items found."
                  hint="This purchase order currently has no stored line items."
                />
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Product</th>
                        <th>Quantity</th>
                        <th>Received</th>
                        <th>Unit Price</th>
                        <th>Line Total</th>
                      </tr>
                    </thead>

                    <tbody>
                      {selectedOrder.items.map((item) => {
                        const product = productMap.get(item.product_id);

                        return (
                          <tr key={item.id}>
                            <td>
                              {product ? (
                                <>
                                  <div>{product.name}</div>
                                  <div className="text-muted">
                                    {product.sku}
                                  </div>
                                </>
                              ) : (
                                `Product #${item.product_id}`
                              )}
                            </td>

                            <td>{item.quantity.toLocaleString()}</td>

                            <td>
                              {item.received_quantity.toLocaleString()}
                            </td>

                            <td>
                              {formatAmount(
                                item.unit_price,
                                selectedOrder.currency,
                              )}
                            </td>

                            <td>
                              {formatAmount(
                                item.line_total,
                                selectedOrder.currency,
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              <h3 className="section-title">Shipments</h3>

              {selectedOrder.shipments.length === 0 ? (
                <EmptyState
                  title="No shipments found."
                  hint="This purchase order does not currently have a shipment record."
                />
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Shipment</th>
                        <th>Carrier</th>
                        <th>Tracking</th>
                        <th>Status</th>
                        <th>Expected Arrival</th>
                      </tr>
                    </thead>

                    <tbody>
                      {selectedOrder.shipments.map((shipment) => (
                        <tr key={shipment.id}>
                          <td>#{shipment.id}</td>
                          <td>{shipment.carrier ?? "—"}</td>
                          <td>{shipment.tracking_number ?? "—"}</td>
                          <td>{shipment.status}</td>
                          <td>
                            {formatDate(shipment.expected_arrival)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedOrder.notes && (
                <>
                  <h3 className="section-title">Notes</h3>
                  <p>{selectedOrder.notes}</p>
                </>
              )}
            </>
          )}
        </Card>
      )}
    </>
  );
}