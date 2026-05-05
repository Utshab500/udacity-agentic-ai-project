For each item:
│
├── status == "unresolved_item"
│     → quote_status = "skipped_unresolved", no pricing
│
├── deliverable_quantity == 0  (out of stock)
│     → quote_status = "skipped_no_stock", no pricing
│
└── deliverable_quantity > 0
      │
      ├── historical_quote_count >= 2
      │     → pricing_method = "historical_avg"
      │     → historical_avg_price = mean of all historical quotes
      │     → discount_pct = null
      │
      └── historical_quote_count < 2
            → pricing_method = "bulk_discount"
            → discount_pct based on tiers (5/10/15%)
            → historical_avg_price = null