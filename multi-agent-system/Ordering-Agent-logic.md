Given: requested item, quantity, customer delivery deadline, request date

1. Check current stock (Calculate Stock Tool)
   │
   ├── Stock >= requested quantity?
   │     └── YES → Place Order Tool (record sale) → done ✅
   │
   └── NO (stock insufficient)
         │
         └── Check delivery timeline (Delivery Timeline Tool)
               │   → get_supplier_delivery_date(request_date, restock_qty)
               │
               ├── Supplier can deliver BEFORE customer deadline?
               │     └── YES → Restock Tool first (record stock_order)
               │                 → then Place Order Tool (record sale) → done ✅
               │
               └── NO → Reply: "Cannot fulfill order, restock arrives too late" ❌

# Edge Case 1
Stock: 300, Requested: 500, Shortage: 200

├── Can restock 200 arrive before deadline?
│     └── YES → Place sale for 300 (existing stock)
│                + Restock 200
│                + Place sale for 200 (restocked units)
│                → "Fulfilled 500 units (300 from stock + 200 restocked)" ✅
│
└── NO  → Place sale for 300 only (what we have)
           → "Partially fulfilled: 300 of 500 units. 
              Remaining 200 unavailable before your deadline." ⚠️


# Edge Case 2
Item A: 500 requested, 600 in stock       → Full fulfillment ✅
Item B: 300 requested, 100 in stock       → Partial + restock check
Item C: 200 requested, 0 in stock         → Restock check only

Each item is processed independently.
Order response summarizes per-item outcome:
  - Item A: Fulfilled ✅
  - Item B: 100 from stock + 200 restocked ✅ (if timeline ok) or 100 only ⚠️
  - Item C: 200 restocked ✅ (if timeline ok) or unavailable ❌