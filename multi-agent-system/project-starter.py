import pandas as pd
import numpy as np
import os
import time
import re
import hashlib
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────

dotenv.load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)

# ─────────────────────────────────────────────
# INTENT CLASSIFIER — TOOL
# ─────────────────────────────────────────────

@tool
def get_product_catalog() -> str:
    """
    Returns the full list of valid product names in the Munder Difflin catalog.
    Use this to resolve customer-mentioned product names to exact catalog names
    before classifying the request.

    Returns:
        str: All valid product names, one per line.
    """
    names = [item["item_name"] for item in paper_supplies]
    return "Valid product names in catalog:\n" + "\n".join(f"- {n}" for n in names)


# ─────────────────────────────────────────────
# PIPELINE LOGGING — Python helper (no LLM involved)
# ─────────────────────────────────────────────

# In-memory dedupe cache: (file_path, content_hash) -> last save timestamp
_pipeline_save_cache: Dict[tuple, datetime] = {}


def save_pipeline_section(
    response: str,
    section_name: str,
    file_path: str,
    metadata: dict = None,
    dedupe_window_seconds: int = 30,
) -> None:
    """
    Extract a labelled section from the orchestrator's response and append it
    to a per-stage audit CSV. Deterministic — no LLM, no duplicates.

    The orchestrator emits responses with markers like:
        === CLASSIFICATION ===
        ...content...
        === INVENTORY ===
        ...content...

    This helper grabs the chunk under one marker, up to the next marker (or end),
    and appends it with a timestamp + optional metadata to the given CSV file.

    Dedupe: if the same content was saved to the same file within
    `dedupe_window_seconds`, skip the write.

    Args:
        response: The full text response from orchestrator_agent.run().
        section_name: Marker label without the equals signs, e.g. "CLASSIFICATION".
        file_path: Path to the CSV file to append to (created if missing).
        metadata: Optional dict of extra columns (e.g. request_id, request_date).
        dedupe_window_seconds: Skip write if same content was saved within this window.
    """
    pattern = rf"=== {re.escape(section_name)} ===\s*(.*?)(?=\n=== |\Z)"
    match = re.search(pattern, response, flags=re.DOTALL)
    if not match:
        print(f"[save_pipeline_section] No '{section_name}' section found in response.")
        return

    extracted = match.group(1).strip()
    now = datetime.now()

    # ── Dedupe check ──
    cache_key = (file_path, hashlib.md5(extracted.encode("utf-8")).hexdigest())
    last_save = _pipeline_save_cache.get(cache_key)
    if last_save and (now - last_save).total_seconds() < dedupe_window_seconds:
        print(f"[save_pipeline_section] Duplicate '{section_name}' skipped (saved {(now - last_save).total_seconds():.1f}s ago).")
        return
    _pipeline_save_cache[cache_key] = now

    row = {"timestamp": now.isoformat(), **(metadata or {}), section_name.lower(): extracted}
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    pd.DataFrame([row]).to_csv(file_path, mode="a", header=not file_exists, index=False)


# ─────────────────────────────────────────────
# RESPONSE VERIFICATION LAYER
# ─────────────────────────────────────────────

def _try_parse_json_lenient(text: str):
    """
    Attempt to parse JSON from orchestrator output. Try strict first.
    If that fails, try a simple repair for the most common LLM mistake:
    unescaped double-quotes inside string values like
        "300 poster boards (24" x 36")"
    Returns the parsed dict, or None if all attempts fail.
    """
    import json as _json

    # Strict parse first
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # Heuristic repair: common case where the LLM forgot to escape "
    # inside an "original_name" value. Replace stray inch-marks with single quotes.
    repaired = re.sub(
        r'(\"original_name\":\s*\")(.*?)(\")(\s*,\s*\"resolved_name\")',
        lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3) + m.group(4),
        text,
    )
    try:
        return _json.loads(repaired)
    except _json.JSONDecodeError:
        return None


def _extract_section(response: str, section_name: str) -> str:
    """
    Extract the content under a "=== SECTION_NAME ===" marker from the
    orchestrator's response.

    Handles two formats:

    1. Standard text format (preferred):
         === CLASSIFICATION ===
         {"...": ...}
         === INVENTORY ===
         ...

    2. JSON-of-strings drift (LLM sometimes does this):
         {"=== CLASSIFICATION ===": "{...}", "=== INVENTORY ===": "...", ...}

    Returns the inner content as a string (with escapes resolved if needed),
    or empty string if not found.
    """
    if not isinstance(response, str):
        return ""

    # Format 1: standard markers
    m = re.search(
        rf"=== {re.escape(section_name)} ===\s*(.*?)(?=\n=== |\Z)",
        response,
        flags=re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip()
        # Sanity check: if the captured chunk starts with `":"` it's probably
        # the JSON-of-strings format leaking through — fall through to format 2.
        if not candidate.startswith('":"'):
            return candidate

    # Format 2: JSON-of-strings (the LLM drift case)
    try:
        import json as _json
        # Strip any leading/trailing whitespace and try to parse the whole thing
        parsed = _json.loads(response.strip())
        if isinstance(parsed, dict):
            key = f"=== {section_name} ==="
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    except Exception:
        pass

    return ""


def _classification_from_raw_request(raw_request: str, request_date: str) -> dict:
    """
    Last-resort rescue: if the orchestrator's output is so mangled that we
    can't extract any section, run the intent_classifier_agent ourselves
    against the raw request to get a clean classification.

    This makes the system fully resilient to orchestrator-format failures.
    """
    try:
        # Build a "clean" request for just the classifier (no need for cash etc.)
        result = intent_classifier_agent.run(raw_request)
    except Exception as e:
        print(f"  [rescue] Classifier rescue call failed: {e}")
        return None

    # Result may be a JSON string or already a dict, depending on smolagents version
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return _try_parse_json_lenient(result)
    return None


def verify_and_fix_response(
    response: str,
    request_date: str,
    raw_request: str = "",
) -> str:
    """
    Post-process the orchestrator's response so the inventory, quote, AND
    order sections are GUARANTEED CORRECT, regardless of what the LLM wrote.

    This function is also the SOLE WRITER of order transactions to the
    database. The ordering_agent's tool computes a plan but does not persist;
    persistence happens here so we have one and only one source of writes.

    Robustness layers (tried in order):
      A. Standard text-format extraction.
      B. JSON-of-strings drift extraction (LLM sometimes wraps the response
         as {"=== CLASSIFICATION ===": "{...}", ...}).
      C. RESCUE: re-run intent_classifier directly on the raw_request.

    Steps:
      1. Get a parseable CLASSIFICATION (via A, B, or C).
      2. Recompute INVENTORY deterministically.
      3. Recompute QUOTE deterministically.
      4. Recompute ORDER deterministically AND commit transactions.
      5. Recompute SUMMARY counts.
      6. Reassemble the response with clean text-format sections.
    """
    import json as _json

    # ── Try to get CLASSIFICATION from response (A or B) ──
    clf_text = _extract_section(response, "CLASSIFICATION")
    clf_json = _try_parse_json_lenient(clf_text) if clf_text else None

    # ── Rescue path (C): re-run classifier on raw request ──
    if clf_json is None and raw_request:
        print("  [verify] CLASSIFICATION unparseable — invoking rescue path.")
        clf_json = _classification_from_raw_request(raw_request, request_date)

    # If still no classification, give up gracefully and emit an error response
    # (prevents a malformed CSV row).
    if clf_json is None:
        print("  [verify] All extraction paths failed — emitting error response.")
        return (
            f"=== CLASSIFICATION ===\n{{}}\n\n"
            f"=== INVENTORY ===\n{{\"as_of_date\": \"{request_date}\", \"items\": []}}\n\n"
            f"=== QUOTE ===\n{{\"as_of_date\": \"{request_date}\", \"items\": []}}\n\n"
            f"=== ORDER ===\n{{\"as_of_date\": \"{request_date}\", \"items\": []}}\n\n"
            f"=== SUMMARY ===\n"
            f"request_type: unknown\n"
            f"delivery_deadline: null\n"
            f"total_items: 0\n"
            f"unresolved_items: 0\n"
            f"items_with_shortage: 0\n"
            f"items_quoted: 0\n"
            f"items_fulfilled: 0\n"
            f"items_skipped: 0\n"
            f"total_revenue: 0\n"
            f"verifier_status: extraction_failed"
        )

    items = clf_json.get("items", [])
    delivery_deadline = clf_json.get("delivery_deadline")  # request-level deadline

    # ── Recompute INVENTORY deterministically (no LLM, no DB writes) ──
    inv_items = [
        _compute_inventory_record(
            item_name=item.get("resolved_name", "unresolved"),
            requested_quantity=item.get("quantity", 0),
            as_of_date=request_date,
            original_name=item.get("original_name", ""),
            delivery_deadline=delivery_deadline,
        )
        for item in items
    ]
    inv_json = {"as_of_date": request_date, "items": inv_items}

    # ── Recompute QUOTE deterministically (no LLM, no DB writes) ──
    quote_items = [
        _compute_quote_record(inv_item, request_date)
        for inv_item in inv_items
    ]
    quote_json = {"as_of_date": request_date, "items": quote_items}

    # ── Recompute ORDER deterministically AND commit transactions to DB.
    #    This is the SOLE writer — the agent's tool produces a plan but
    #    does not persist. That guarantees no double-spending and ensures
    #    the saved state matches the saved CSV.
    available_cash = get_cash_balance(request_date)
    order_items = []
    for q_item in quote_items:
        outcome = _compute_order_outcome(q_item, request_date, available_cash)
        _execute_order_transactions(outcome, request_date)
        # Decrement running cash by what this item spent
        available_cash -= float(outcome.get("cash_spent_now") or 0.0)
        order_items.append(outcome)
    order_json = {"as_of_date": request_date, "items": order_items}

    # ── Recompute SUMMARY ──
    total_items = len(inv_items)
    unresolved_count = sum(1 for i in inv_items if i["status"] == "unresolved_item")
    shortage_count = sum(1 for i in inv_items if i["shortage_quantity"] > 0)
    quoted_count = sum(1 for i in quote_items if i["quote_status"] == "quoted")
    fulfilled_count = sum(
        1 for i in order_items if str(i.get("order_status", "")).startswith("fulfilled")
    )
    skipped_count = sum(
        1 for i in order_items if str(i.get("order_status", "")).startswith("skipped")
    )
    total_revenue = round(
        sum(float(i.get("sale_total_revenue") or 0.0) for i in order_items), 2
    )

    # ── Reassemble response ──
    new_response = (
        f"=== CLASSIFICATION ===\n"
        f"{_json.dumps(clf_json, indent=2)}\n\n"
        f"=== INVENTORY ===\n"
        f"{_json.dumps(inv_json, indent=2)}\n\n"
        f"=== QUOTE ===\n"
        f"{_json.dumps(quote_json, indent=2)}\n\n"
        f"=== ORDER ===\n"
        f"{_json.dumps(order_json, indent=2)}\n\n"
        f"=== SUMMARY ===\n"
        f"request_type: {clf_json.get('request_type')}\n"
        f"delivery_deadline: {clf_json.get('delivery_deadline')}\n"
        f"total_items: {total_items}\n"
        f"unresolved_items: {unresolved_count}\n"
        f"items_with_shortage: {shortage_count}\n"
        f"items_quoted: {quoted_count}\n"
        f"items_fulfilled: {fulfilled_count}\n"
        f"items_skipped: {skipped_count}\n"
        f"total_revenue: {total_revenue}"
    )
    return new_response


# ─────────────────────────────────────────────
# FINALIZER TOOL — Customer-facing response composer
# ─────────────────────────────────────────────

def finalize_customer_response(verified_response: str) -> str:
    """
    Deterministic Python helper (the "Finalizer Tool" in our diagram) that
    converts the verified structured response into a natural-language,
    customer-facing message suitable for test_results.csv.

    Inputs:
        verified_response: The post-verifier text containing
        === CLASSIFICATION ===, === INVENTORY ===, === QUOTE ===,
        === ORDER ===, === SUMMARY === sections (already deterministically
        produced by verify_and_fix_response).

    Output:
        A natural-language string addressed to the customer, summarising
        what was fulfilled, what was skipped, the order total, and the
        discount/savings the customer received.
    """
    import json as _json

    # Pull structured data out of the verified response
    clf_text = _extract_section(verified_response, "CLASSIFICATION")
    order_text = _extract_section(verified_response, "ORDER")

    clf_json = _try_parse_json_lenient(clf_text) if clf_text else None
    order_json = _try_parse_json_lenient(order_text) if order_text else None

    if not order_json or not isinstance(order_json.get("items"), list):
        return (
            "Dear Customer,\n\n"
            "Thank you for your inquiry. Unfortunately we encountered an "
            "issue processing your request. Please resubmit or contact us "
            "directly.\n\n"
            "Best regards,\nMunder Difflin Paper Co."
        )

    items = order_json["items"]
    deadline = (clf_json or {}).get("delivery_deadline")

    # Categorise items
    fulfilled = []
    partial = []
    skipped_no_stock = []
    skipped_unresolved = []

    total_revenue = 0.0
    total_retail_value = 0.0   # total at full catalog price (no discount)
    earliest_arrival = None    # latest restock arrival across fulfilled items

    def _retail_value_for(item):
        """Catalog price × total units sold for this item (full retail)."""
        catalog_price = _get_unit_price(item.get("requested_item", ""))
        total_units = (
            int(item.get("sale_units_immediate") or 0)
            + int(item.get("sale_units_restocked") or 0)
        )
        return round(catalog_price * total_units, 4)

    for item in items:
        status = item.get("order_status", "")
        revenue = float(item.get("sale_total_revenue") or 0.0)
        retail = _retail_value_for(item)
        total_revenue += revenue
        total_retail_value += retail

        if item.get("restock_units", 0):
            arr = item.get("restock_arrival_date")
            if arr and (earliest_arrival is None or arr > earliest_arrival):
                earliest_arrival = arr

        if status in ("fulfilled_full", "fulfilled_stock_only"):
            fulfilled.append(item)
        elif status == "fulfilled_partial":
            partial.append(item)
        elif status == "skipped_no_stock":
            skipped_no_stock.append(item)
        elif status == "skipped_unresolved":
            skipped_unresolved.append(item)

    def _per_item_discount_phrase(item):
        """
        Produce a short phrase describing the discount/saving for one item,
        e.g. ' — 5% bulk discount applied (you save $2.00)' or empty string.
        """
        retail = _retail_value_for(item)
        revenue = float(item.get("sale_total_revenue") or 0.0)
        savings = round(retail - revenue, 2)
        if savings <= 0.005:
            return ""

        method = item.get("pricing_method")
        if method == "bulk_discount" and item.get("discount_pct"):
            return (
                f" — {int(item['discount_pct'])}% bulk discount applied "
                f"(you save ${savings:.2f})"
            )
        if method == "historical_avg":
            return f" — competitive pricing applied (you save ${savings:.2f})"
        # Fallback (e.g. method=null but somehow priced under retail)
        return f" — discount applied (you save ${savings:.2f})"

    # ── Build the customer-facing message ──
    lines = ["Dear Customer,", ""]

    if fulfilled or partial:
        lines.append(
            "Thank you for your order with Munder Difflin Paper Co. "
            "Here is the status of your requested items:"
        )
        lines.append("")
    else:
        lines.append(
            "Thank you for reaching out to Munder Difflin Paper Co. "
            "Unfortunately we were unable to fulfil any of the requested items. "
            "Please see the details below:"
        )
        lines.append("")

    # Fulfilled items
    for item in fulfilled:
        orig = item.get("original_name") or item.get("requested_item", "this item")
        catalog = item.get("requested_item", "")
        qty_imm = int(item.get("sale_units_immediate") or 0)
        qty_re = int(item.get("sale_units_restocked") or 0)
        revenue = float(item.get("sale_total_revenue") or 0.0)
        arrival = item.get("restock_arrival_date")
        disc_phrase = _per_item_discount_phrase(item)

        if qty_re > 0 and qty_imm > 0:
            base = (
                f"  • {orig} — fulfilled {qty_imm + qty_re} units of {catalog} "
                f"({qty_imm} from existing stock, {qty_re} via restock"
                f"{f' arriving {arrival}' if arrival else ''}). "
                f"Subtotal: ${revenue:.2f}"
            )
        elif qty_re > 0:
            base = (
                f"  • {orig} — fulfilled {qty_re} units of {catalog} via restock"
                f"{f' arriving {arrival}' if arrival else ''}. "
                f"Subtotal: ${revenue:.2f}"
            )
        else:
            base = (
                f"  • {orig} — fulfilled {qty_imm} units of {catalog} "
                f"from existing stock. Subtotal: ${revenue:.2f}"
            )
        lines.append(base + disc_phrase)

    # Partially fulfilled items
    for item in partial:
        orig = item.get("original_name") or item.get("requested_item", "this item")
        catalog = item.get("requested_item", "")
        qty_imm = int(item.get("sale_units_immediate") or 0)
        requested = int(item.get("requested_quantity") or 0)
        shortage = int(item.get("shortage_quantity") or 0)
        revenue = float(item.get("sale_total_revenue") or 0.0)
        disc_phrase = _per_item_discount_phrase(item)

        lines.append(
            f"  • {orig} — partially fulfilled: {qty_imm} of {requested} units "
            f"of {catalog} from existing stock. The remaining {shortage} units "
            f"could not be sourced before your delivery deadline. "
            f"Subtotal: ${revenue:.2f}" + disc_phrase
        )

    # Out-of-stock items (couldn't restock either)
    if skipped_no_stock:
        if fulfilled or partial:
            lines.append("")
            lines.append("The following items could not be fulfilled at this time:")
        for item in skipped_no_stock:
            orig = item.get("original_name") or item.get("requested_item", "this item")
            catalog = item.get("requested_item", "")
            requested = int(item.get("requested_quantity") or 0)
            lines.append(
                f"  • {orig} — {requested} units of {catalog} are currently "
                f"unavailable and could not be restocked in time."
            )

    # Items not in catalog
    if skipped_unresolved:
        lines.append("")
        lines.append(
            "We could not match the following items to our product catalog. "
            "Please contact us if you believe this is an error:"
        )
        for item in skipped_unresolved:
            orig = item.get("original_name") or "(unspecified item)"
            lines.append(f"  • {orig}")

    # Totals + savings highlight
    if total_revenue > 0:
        lines.append("")
        lines.append(f"Order Total: ${total_revenue:.2f}")

        total_savings = round(total_retail_value - total_revenue, 2)
        if total_savings >= 0.01 and total_retail_value > 0:
            effective_pct = total_savings / total_retail_value * 100
            lines.append(
                f"Total Savings: ${total_savings:.2f} "
                f"({effective_pct:.1f}% off catalog price)"
            )

        if deadline:
            lines.append(f"Requested Delivery Deadline: {deadline}")
        if earliest_arrival:
            lines.append(f"Restock Arrival (latest): {earliest_arrival}")

    lines.append("")
    lines.append("Thank you for choosing Munder Difflin Paper Co.")
    lines.append("")
    lines.append("Best regards,")
    lines.append("Munder Difflin Sales Team")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# INTENT CLASSIFIER — AGENT
# ─────────────────────────────────────────────

INTENT_CLASSIFIER_PROMPT = """
You are the Intent Classifier Agent for Munder Difflin, a paper supply company.

YOUR ONLY JOB: analyse a customer request and return a structured JSON object.

YOU MUST NOT:
- Place orders, fulfil orders, or claim any action was taken.
- Provide pricing, availability, or inventory information.
- Add commentary, explanations, markdown, code fences, or any prose.
- Paraphrase or summarise. Output JSON only.

═══════════════════════════════════════════════════════
WORKFLOW (follow exactly, every time):
═══════════════════════════════════════════════════════

STEP 1 — Call get_product_catalog ONCE to get the list of valid catalog names.
         These names are CASE-SENSITIVE. You must preserve exact capitalisation.

STEP 2 — Determine request_type:
   • "order"   → customer wants to purchase / place an order / buy something
   • "inquiry" → customer is asking about availability, pricing, or info only

STEP 3 — Extract every item the customer mentions and its quantity (integer).

STEP 4 — Resolve each item to the EXACT catalog name from get_product_catalog.
         CASE-SENSITIVE — copy the name character-for-character.
         "poster paper" is WRONG. "Poster paper" is correct.
         "party streamers" is WRONG. "Party streamers" is correct.

         Resolution rules:
         • "glossy A4 paper" / "glossy paper" / "A3 glossy paper"
                                                        → "Glossy paper"
         • "matte paper" / "A4 matte paper" / "A3 matte paper"
                                                        → "Matte paper"
         • "heavy cardstock" / "white cardstock" / "colorful cardstock" /
           "cardstock in assorted colors" / "heavyweight cardstock"
                                                        → "Cardstock"
         • "colored paper" / "assorted colored paper" / "A3 colored paper"
                                                        → "Colored paper"
         • "construction paper" / "colorful construction paper"
                                                        → "Construction paper"
         • "recycled paper" / "recycled cardstock" / "kraft paper envelopes"
                                                        → match the closest
                                                          paper category in catalog
         • "biodegradable cups" / "paper cups"          → "Paper cups"
         • "paper plates" / "biodegradable plates"      → "Paper plates"
         • "table napkins" / "napkins" / "paper napkins" → "Paper napkins"
         • "poster board" / "poster boards 24x36"        → "Large poster paper (24x36 inches)"
         • "washi tape" / "decorative washi tape"        → "Decorative adhesive tape (washi tape)"
         • "streamers" / "rolls of streamers"            → "Party streamers"
         • "printer paper" / "copy paper" / "standard paper" /
           "reams of printer paper" / "standard printer paper" /
           "standard printing paper"                     → "Standard copy paper"
         • "A4 paper" (plain, no qualifier) / "A4 white paper" /
           "A4 printing paper"                           → "A4 paper"
         • "flyers"                                      → "Flyers"
         • "posters" / "poster paper" (no size)          → "Poster paper"
         • "balloons", "ribbons", "tickets", "cardboard"
                                                        → "unresolved"

📌 IMPORTANT — paper SIZE alone does NOT make an item unresolved:
   - "A3 matte paper" → use "Matte paper" (size doesn't matter, we don't
     differentiate; the catalog just has "Matte paper")
   - "A3 glossy paper" → "Glossy paper"
   - "A3 colored paper" → "Colored paper"
   - "A5 paper" (PLAIN, no other qualifier) → "unresolved" (no plain A5 in catalog)
   - But "A5 glossy paper" → "Glossy paper"

📌 IMPORTANT — preserve catalog capitalisation EXACTLY:
   "Paper napkins" not "paper napkins"
   "Cardstock" not "cardstock"
   "Glossy paper" not "glossy paper"

STEP 5 — Extract delivery_deadline:
         If the customer mentions a delivery date, format as YYYY-MM-DD.
         Otherwise: null.

STEP 6 — Output ONLY a single JSON object exactly matching this schema.
         No prose before, no prose after, no markdown fences, no comments.

{
  "request_type": "order" | "inquiry",
  "delivery_deadline": "YYYY-MM-DD" | null,
  "items": [
    {
      "original_name": "<the exact phrase the customer used>",
      "resolved_name": "<exact catalog name OR 'unresolved'>",
      "quantity": <integer>
    }
  ]
}

═══════════════════════════════════════════════════════
EXAMPLE INPUT/OUTPUT
═══════════════════════════════════════════════════════

INPUT: "I'd like 500 sheets of colorful poster paper and 200 balloons. Deliver by April 15, 2025."

OUTPUT:
{
  "request_type": "order",
  "delivery_deadline": "2025-04-15",
  "items": [
    {"original_name": "500 sheets of colorful poster paper", "resolved_name": "Poster paper", "quantity": 500},
    {"original_name": "200 balloons", "resolved_name": "unresolved", "quantity": 200}
  ]
}
"""

intent_classifier_agent = ToolCallingAgent(
    tools=[get_product_catalog],
    model=model,
    max_steps=3,
    name="intent_classifier",
    description=(
        "Pass the full raw customer request text. "
        "Returns JSON with: request_type (order/inquiry), delivery_deadline, "
        "and items list — each with original_name, resolved catalog name, and quantity. "
        "Always call this FIRST for every incoming request."
    ),
    instructions=INTENT_CLASSIFIER_PROMPT,
)


# ─────────────────────────────────────────────
# INVENTORY AGENT — TOOL
# ─────────────────────────────────────────────

def _compute_inventory_record(
    item_name: str,
    requested_quantity: int,
    as_of_date: str,
    original_name: str = "",
    delivery_deadline: str = None,
) -> dict:
    """
    Deterministic computation of an inventory record for one item.
    Plain Python — callable directly (used by the verification layer)
    AND by the @tool wrapper below (used by the LLM agent).

    This is the SINGLE SOURCE OF TRUTH for inventory data.
    Both code paths (agent and verifier) produce identical results.

    The original_name (the customer's exact phrase) and delivery_deadline
    are preserved through the pipeline so downstream agents can produce
    customer-friendly responses and make timeline-aware decisions.
    """
    requested = int(requested_quantity) if requested_quantity else 0

    # Validate against the actual catalog. Any name not in paper_supplies
    # (or empty / "unresolved") is treated as an unresolved item.
    catalog_names = {item["item_name"] for item in paper_supplies}
    is_unresolved = (
        not item_name
        or item_name == "unresolved"
        or item_name not in catalog_names
    )

    if is_unresolved:
        return {
            "original_name": original_name,
            "requested_item": "unresolved",
            "delivery_deadline": delivery_deadline,
            "requested_quantity": requested,
            "stock_quantity": 0,
            "deliverable_quantity": 0,
            "shortage_quantity": requested,
            "status": "unresolved_item",
        }

    # Real catalog item — query the DB
    try:
        stock_df = get_stock_level(item_name, as_of_date)
        stock_qty = int(stock_df.iloc[0]["current_stock"]) if not stock_df.empty else 0
    except Exception:
        stock_qty = 0

    return {
        "original_name": original_name,
        "requested_item": item_name,
        "delivery_deadline": delivery_deadline,
        "requested_quantity": requested,
        "stock_quantity": stock_qty,
        "deliverable_quantity": min(requested, stock_qty),
        "shortage_quantity": max(0, requested - stock_qty),
        "status": "ok",
    }


@tool
def check_inventory_for_item(
    item_name: str,
    requested_quantity: int,
    as_of_date: str,
    original_name: str = "",
    delivery_deadline: str = None,
) -> str:
    """
    Compute the complete inventory record for ONE requested item.
    Performs the stock lookup AND all math (deliverable, shortage) deterministically.
    The agent should NOT do any arithmetic — just call this tool once per item
    and collect the JSON results.

    Args:
        item_name: The exact catalog name from the intent classifier (e.g.
                   "Glossy paper", "Cardstock"). Pass "unresolved" for items
                   not in the catalog.
        requested_quantity: How many units the customer wants (positive integer).
        as_of_date: Date in YYYY-MM-DD format (use the request date).
        original_name: The customer's original phrasing (e.g. "200 sheets of
                       A4 glossy paper") — passed through for downstream agents.
        delivery_deadline: When the customer wants the items delivered
                           (YYYY-MM-DD or null) — passed through for the
                           ordering agent's restock-feasibility decisions.

    Returns:
        str: A JSON string with EXACTLY these 8 keys:
        {
          "original_name": "<customer's original phrasing>",
          "requested_item": "<item_name passed in or 'unresolved'>",
          "delivery_deadline": "<YYYY-MM-DD or null>",
          "requested_quantity": <int>,
          "stock_quantity": <int>,
          "deliverable_quantity": <int>,
          "shortage_quantity": <int>,
          "status": "ok" | "unresolved_item"
        }
    """
    import json as _json
    record = _compute_inventory_record(
        item_name, requested_quantity, as_of_date, original_name, delivery_deadline
    )
    return _json.dumps(record)


# ─────────────────────────────────────────────
# INVENTORY AGENT
# ─────────────────────────────────────────────

INVENTORY_AGENT_PROMPT = """
You are the Inventory Agent for Munder Difflin.

═══════════════════════════════════════════════════════
THE MOST IMPORTANT RULE — READ THIS FIRST
═══════════════════════════════════════════════════════

YOU HAVE ZERO KNOWLEDGE OF STOCK LEVELS. ZERO.
You do NOT know what is in stock. You do NOT know catalog membership.
You CANNOT guess, infer, or assume any stock value.

The ONLY way to get stock data is to call check_inventory_for_item.
Hallucinating stock numbers is the worst thing you can do — it produces
wrong business decisions downstream.

EVERY item in the input list MUST result in EXACTLY ONE call to
check_inventory_for_item. No exceptions. No shortcuts.

This applies even if:
- The item looks "obviously out of stock"
- The resolved_name is "unresolved"
- You think you "already know" the answer
- Other items in the same request had stock=0
- The customer asked for "a lot" of something

Skip the tool call → wrong answer → forbidden.

═══════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════

A JSON object like:
{
  "as_of_date": "YYYY-MM-DD",
  "delivery_deadline": "YYYY-MM-DD" or null,
  "items": [
    {"original_name": "<customer's phrasing>",
     "resolved_name": "<exact catalog name or 'unresolved'>",
     "quantity": <int>}
  ]
}

═══════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════

For EACH item in the input list (in order, NO skipping):

  1. Call check_inventory_for_item(
         item_name=<resolved_name>,
         requested_quantity=<quantity>,
         as_of_date=<as_of_date>,
         original_name=<original_name>,
         delivery_deadline=<top-level delivery_deadline>
     )

  2. Take the JSON the tool returns. Add it as-is to your output list.

DO NOT modify the tool's output.
DO NOT compute anything yourself.
DO NOT change the "status" field — the tool decides "ok" vs "unresolved_item".
DO NOT skip a tool call because you "know" the answer.

═══════════════════════════════════════════════════════
OUTPUT — JSON ONLY, NO PROSE, NO MARKDOWN
═══════════════════════════════════════════════════════

{
  "as_of_date": "YYYY-MM-DD",
  "items": [
    <exact JSON record returned by check_inventory_for_item, one per input item>
  ]
}

The "status" field MUST be either "ok" or "unresolved_item" — those are
the only two values the tool returns. If you see any other status value
in your output, you have hallucinated and failed.

═══════════════════════════════════════════════════════
EXAMPLE — 3 items means 3 tool calls
═══════════════════════════════════════════════════════

INPUT:
{
  "as_of_date": "2025-04-01",
  "delivery_deadline": "2025-04-15",
  "items": [
    {"original_name": "200 sheets of glossy paper", "resolved_name": "Glossy paper", "quantity": 200},
    {"original_name": "500 sheets of cardstock",    "resolved_name": "Cardstock",    "quantity": 500},
    {"original_name": "100 balloons",               "resolved_name": "unresolved",   "quantity": 100}
  ]
}

You MUST call check_inventory_for_item exactly 3 times:
  Call 1: check_inventory_for_item("Glossy paper", 200, "2025-04-01",
                                    "200 sheets of glossy paper", "2025-04-15")
  Call 2: check_inventory_for_item("Cardstock",    500, "2025-04-01",
                                    "500 sheets of cardstock",    "2025-04-15")
  Call 3: check_inventory_for_item("unresolved",   100, "2025-04-01",
                                    "100 balloons",               "2025-04-15")

Then assemble the three returned JSON records into:

OUTPUT:
{
  "as_of_date": "2025-04-01",
  "items": [
    {"original_name": "200 sheets of glossy paper", "requested_item": "Glossy paper", "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 587, "deliverable_quantity": 200, "shortage_quantity": 0, "status": "ok"},
    {"original_name": "500 sheets of cardstock", "requested_item": "Cardstock", "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 595, "deliverable_quantity": 500, "shortage_quantity": 0, "status": "ok"},
    {"original_name": "100 balloons", "requested_item": "unresolved", "delivery_deadline": "2025-04-15", "requested_quantity": 100, "stock_quantity": 0, "deliverable_quantity": 0, "shortage_quantity": 100, "status": "unresolved_item"}
  ]
}
"""

inventory_agent = ToolCallingAgent(
    tools=[check_inventory_for_item],
    model=model,
    max_steps=15,  # request 17 had 5 items; need headroom for tool-calling overhead
    name="inventory_agent",
    description=(
        "Looks up REAL stock data via tool calls. Pass a JSON string with "
        "as_of_date and a list of items (each with resolved_name and quantity). "
        "Returns JSON with stock figures: requested_quantity, stock_quantity, "
        "deliverable_quantity, shortage_quantity, status per item. "
        "This agent has no inherent stock knowledge — every item triggers a "
        "tool call to fetch current data. Call this AFTER intent_classifier."
    ),
    instructions=INVENTORY_AGENT_PROMPT,
)


# ─────────────────────────────────────────────
# QUOTING AGENT — TOOL & DETERMINISTIC LOGIC
# ─────────────────────────────────────────────

# Bulk discount tiers (applied when there's not enough historical pricing data).
# Quantity is matched against deliverable_quantity (what we can actually ship).
BULK_DISCOUNT_TIERS = [
    (1000, 15),  # 1,000+ units → 15% off
    (500,  10),  # 500–999  → 10% off
    (100,   5),  # 100–499  → 5% off
    (0,     0),  # 1–99     → no discount
]

# Minimum number of historical quotes required before we trust the average.
# If fewer than this exist, we fall back to bulk discount.
HISTORICAL_QUOTE_MIN_COUNT = 2


def _bulk_discount_for_quantity(quantity: int) -> int:
    """Return the discount percentage for a given deliverable quantity."""
    for threshold, pct in BULK_DISCOUNT_TIERS:
        if quantity >= threshold:
            return pct
    return 0


def _compute_quote_record(item_record: dict, as_of_date: str) -> dict:
    """
    Deterministic computation of a quote record for ONE inventory item.
    Takes the item dict produced by _compute_inventory_record (which already
    contains original_name, requested_item, deliverable_quantity, status, etc.)
    and ENRICHES it with quoting fields.

    Plain Python — single source of truth for both the agent tool and the
    verification layer. Never hallucinates.
    """
    # Start with all the fields the inventory agent provided
    enriched = dict(item_record)

    deliverable = int(enriched.get("deliverable_quantity", 0) or 0)
    status = enriched.get("status", "ok")

    # Default quoting fields (set explicitly so every record has them)
    enriched["pricing_method"] = None
    enriched["historical_avg_price"] = None
    enriched["discount_pct"] = None
    enriched["quote_status"] = None

    # ── Skip cases ──
    if status == "unresolved_item":
        enriched["quote_status"] = "skipped_unresolved"
        return enriched

    if deliverable <= 0:
        enriched["quote_status"] = "skipped_no_stock"
        return enriched

    # ── Quoting cases (deliverable_quantity > 0, valid catalog item) ──
    item_name = enriched.get("requested_item", "")

    # Try historical pricing first
    historical_avg = None
    try:
        history = search_quote_history([item_name], limit=20)
        # search_quote_history returns a list of dicts with "explanation"+"original_request",
        # not raw prices. We need to extract numeric quotes from the explanations.
        # For robustness, we instead look at the financial transactions table for
        # past sales of this exact item.
        prices = _extract_historical_unit_prices(item_name, as_of_date)
        if len(prices) >= HISTORICAL_QUOTE_MIN_COUNT:
            historical_avg = round(sum(prices) / len(prices), 4)
    except Exception:
        historical_avg = None

    if historical_avg is not None:
        enriched["pricing_method"] = "historical_avg"
        enriched["historical_avg_price"] = historical_avg
        enriched["quote_status"] = "quoted"
    else:
        enriched["pricing_method"] = "bulk_discount"
        enriched["discount_pct"] = _bulk_discount_for_quantity(deliverable)
        enriched["quote_status"] = "quoted"

    return enriched


def _extract_historical_unit_prices(item_name: str, as_of_date: str) -> list:
    """
    Look up past SALE transactions for this item and compute the unit price
    of each (price / units). Returns a list of unit prices.

    Uses the transactions table directly (more reliable than parsing
    natural-language quote explanations).
    """
    try:
        with db_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT price, units FROM transactions "
                    "WHERE item_name = :item AND transaction_type = 'sales' "
                    "AND DATE(transaction_date) <= DATE(:as_of) "
                    "AND units > 0"
                ),
                {"item": item_name, "as_of": as_of_date},
            )
            rows = result.fetchall()
    except Exception:
        return []

    prices = []
    for row in rows:
        try:
            price = float(row[0])
            units = float(row[1])
            if units > 0:
                prices.append(price / units)
        except (TypeError, ValueError):
            continue
    return prices


@tool
def compute_quote_for_item(item_record_json: str, as_of_date: str) -> str:
    """
    Compute the complete quote record for ONE item.
    Takes the inventory item dict (as a JSON string) and returns the same
    dict ENRICHED with quoting fields: pricing_method, historical_avg_price,
    discount_pct, quote_status.

    Args:
        item_record_json: JSON string of one inventory item, containing fields
            like original_name, requested_item, deliverable_quantity, status, etc.
        as_of_date: Date in YYYY-MM-DD format (the request date).

    Returns:
        str: A JSON string of the enriched item with these added fields:
            - pricing_method: "historical_avg" | "bulk_discount" | null
            - historical_avg_price: float | null  (if historical_avg)
            - discount_pct: int | null            (if bulk_discount)
            - quote_status: "quoted" | "skipped_no_stock" | "skipped_unresolved"

        The function is deterministic — historical pricing is computed from past
        sales transactions; bulk discounts are computed from a fixed tier table.
    """
    import json as _json
    try:
        item = _json.loads(item_record_json)
    except _json.JSONDecodeError:
        return _json.dumps({"error": "invalid item_record_json"})
    record = _compute_quote_record(item, as_of_date)
    return _json.dumps(record)


# ─────────────────────────────────────────────
# QUOTING AGENT
# ─────────────────────────────────────────────

QUOTING_AGENT_PROMPT = """
You are the Quoting Agent for Munder Difflin.

═══════════════════════════════════════════════════════
THE MOST IMPORTANT RULE — READ THIS FIRST
═══════════════════════════════════════════════════════

YOU HAVE ZERO PRICING KNOWLEDGE.
You CANNOT guess discount percentages, historical prices, or any pricing data.

The ONLY way to get a quote for an item is to call compute_quote_for_item.
Every item in the input list MUST result in EXACTLY ONE call.
No exceptions. No skipping.

═══════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════

A JSON object like:
{
  "as_of_date": "YYYY-MM-DD",
  "items": [
    {
      "original_name": "...",
      "requested_item": "...",
      "delivery_deadline": "YYYY-MM-DD" or null,
      "requested_quantity": <int>,
      "stock_quantity": <int>,
      "deliverable_quantity": <int>,
      "shortage_quantity": <int>,
      "status": "ok" | "unresolved_item"
    }
  ]
}

═══════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════

For EACH item in the input list (in order, NO skipping):

  1. Call compute_quote_for_item(
         item_record_json=<the item dict serialized to a JSON string>,
         as_of_date=<as_of_date>
     )

  2. Take the JSON the tool returns. Add it as-is to your output list.

DO NOT modify the tool's output.
DO NOT compute prices, discounts, or averages yourself.
DO NOT skip any item — even ones with status "unresolved_item" or
deliverable_quantity 0. The tool handles those cases (returns
quote_status "skipped_unresolved" or "skipped_no_stock").

═══════════════════════════════════════════════════════
OUTPUT — JSON ONLY, NO PROSE, NO MARKDOWN
═══════════════════════════════════════════════════════

{
  "as_of_date": "YYYY-MM-DD",
  "items": [
    <exact JSON record returned by compute_quote_for_item, one per input item>
  ]
}
"""

quoting_agent = ToolCallingAgent(
    tools=[compute_quote_for_item],
    model=model,
    max_steps=15,
    name="quoting_agent",
    description=(
        "Generates quotes for each item using historical pricing OR bulk discount "
        "via tool calls. Pass a JSON string with as_of_date and the inventory items "
        "list (each item already has original_name, requested_item, deliverable_quantity, "
        "status, etc. from inventory_agent). Returns JSON where each item is enriched "
        "with: pricing_method (historical_avg | bulk_discount | null), "
        "historical_avg_price, discount_pct, quote_status. "
        "Call this AFTER inventory_agent."
    ),
    instructions=QUOTING_AGENT_PROMPT,
)


# ─────────────────────────────────────────────
# ORDERING AGENT — TOOL & DETERMINISTIC LOGIC
# ─────────────────────────────────────────────

# Supplier cost as a fraction of catalog unit_price.
# Conservative assumption: we buy from suppliers at 50% of retail.
# Future improvement: derive from real historical stock_order transactions.
SUPPLIER_COST_RATIO = 0.5


def _get_unit_price(item_name: str) -> float:
    """Lookup retail unit price for a catalog item; 0.0 if not found."""
    for entry in paper_supplies:
        if entry["item_name"] == item_name:
            return float(entry["unit_price"])
    return 0.0


def _effective_unit_price(item_record: dict) -> float:
    """
    Compute the effective unit sale price for an item, applying the quoting
    agent's pricing decision (historical_avg or bulk_discount).

    Falls back to the catalog base price if pricing_method is missing.
    """
    base = _get_unit_price(item_record.get("requested_item", ""))
    method = item_record.get("pricing_method")
    if method == "historical_avg":
        hist = item_record.get("historical_avg_price")
        if hist is not None and hist > 0:
            return float(hist)
        return base
    if method == "bulk_discount":
        disc = item_record.get("discount_pct") or 0
        return round(base * (1 - disc / 100.0), 4)
    return base


def _compute_order_outcome(
    item_record: dict,
    request_date: str,
    available_cash: float,
) -> dict:
    """
    Deterministic per-item ordering decision.

    Inputs:
        item_record: enriched item dict from quoting_agent (has original_name,
                     requested_item, delivery_deadline, requested_quantity,
                     stock_quantity, deliverable_quantity, shortage_quantity,
                     status, pricing_method, discount_pct, historical_avg_price,
                     quote_status).
        request_date: today's date for this request, "YYYY-MM-DD".
        available_cash: cash available BEFORE this item's restock spend
                        (caller decrements between items).

    Returns:
        Same dict ENRICHED with ordering fields:
          - sale_units_immediate     : int — sold from existing stock today
          - sale_units_restocked     : int — sold from incoming restock
          - restock_units            : int — units we ordered from supplier
          - restock_unit_cost        : float — cost per unit for restock
          - restock_total_cost       : float — total cost of the restock
          - restock_arrival_date     : str | null — when restock physically arrives
          - sale_unit_price          : float — discounted price per unit
          - sale_total_revenue       : float — total $ booked from sales
          - cash_spent_now           : float — cash that left bank for this item
          - timeline_ok              : bool — restock arrives by deadline?
          - cash_ok                  : bool — could afford the restock?
          - profitable               : bool — sale revenue >= restock cost?
          - order_status             : str — final disposition. One of:
              "fulfilled_full"        : got everything via stock + restock
              "fulfilled_partial"     : got some (stock only); restock blocked
              "fulfilled_stock_only"  : exact stock match (no shortage)
              "skipped_unresolved"    : item not in catalog
              "skipped_no_demand"     : requested_quantity == 0
              "skipped_no_stock"      : 0 stock + restock blocked
          - order_message            : str — human-readable explanation
    """
    # Start by carrying everything the upstream stages produced
    enriched = dict(item_record)

    # Defaults (set on every record so schema is always uniform)
    enriched.update({
        "sale_units_immediate": 0,
        "sale_units_restocked": 0,
        "restock_units": 0,
        "restock_unit_cost": 0.0,
        "restock_total_cost": 0.0,
        "restock_arrival_date": None,
        "sale_unit_price": 0.0,
        "sale_total_revenue": 0.0,
        "cash_spent_now": 0.0,
        "timeline_ok": False,
        "cash_ok": False,
        "profitable": False,
        "order_status": None,
        "order_message": "",
    })

    status = enriched.get("status", "ok")
    requested = int(enriched.get("requested_quantity", 0) or 0)
    deliverable = int(enriched.get("deliverable_quantity", 0) or 0)
    shortage = int(enriched.get("shortage_quantity", 0) or 0)
    deadline = enriched.get("delivery_deadline")
    item_name = enriched.get("requested_item", "")

    # ── Skip cases ──
    if status == "unresolved_item":
        enriched["order_status"] = "skipped_unresolved"
        enriched["order_message"] = (
            f"'{enriched.get('original_name', '')}' is not in our catalog."
        )
        return enriched

    if requested <= 0:
        enriched["order_status"] = "skipped_no_demand"
        enriched["order_message"] = "Quantity is zero, nothing to order."
        return enriched

    # Compute sale unit price up front (used for revenue calculations)
    sale_unit_price = _effective_unit_price(enriched)
    enriched["sale_unit_price"] = sale_unit_price

    # ── Branch 1: We have enough stock — fulfill fully from inventory ──
    if shortage == 0 and deliverable >= requested:
        revenue = round(requested * sale_unit_price, 4)
        enriched["sale_units_immediate"] = requested
        enriched["sale_total_revenue"] = revenue
        enriched["order_status"] = "fulfilled_stock_only"
        enriched["order_message"] = (
            f"Fulfilled {requested} units of {item_name} from existing stock."
        )
        return enriched

    # ── Branch 2: Shortage exists — try to restock for the missing amount ──
    # First record the immediate-stock sale (we always sell what we have)
    immediate_sale = deliverable
    immediate_revenue = round(immediate_sale * sale_unit_price, 4)
    enriched["sale_units_immediate"] = immediate_sale

    # Check timeline feasibility
    try:
        arrival = get_supplier_delivery_date(request_date, shortage)
    except Exception:
        arrival = None

    timeline_ok = False
    if arrival and deadline:
        try:
            arrival_dt = datetime.fromisoformat(arrival)
            deadline_dt = datetime.fromisoformat(deadline)
            timeline_ok = arrival_dt <= deadline_dt
        except Exception:
            timeline_ok = False
    elif arrival and not deadline:
        # No customer deadline given — assume best-effort timing acceptable
        timeline_ok = True
    enriched["timeline_ok"] = timeline_ok
    enriched["restock_arrival_date"] = arrival

    # Check cash affordability
    base_unit_price = _get_unit_price(item_name)
    supplier_unit_cost = round(base_unit_price * SUPPLIER_COST_RATIO, 4)
    restock_total_cost = round(shortage * supplier_unit_cost, 4)
    cash_ok = available_cash >= restock_total_cost
    enriched["cash_ok"] = cash_ok
    enriched["restock_unit_cost"] = supplier_unit_cost

    # Check profitability of the restocked units
    restocked_revenue = round(shortage * sale_unit_price, 4)
    profitable = restocked_revenue >= restock_total_cost
    enriched["profitable"] = profitable

    # ── Decide whether to restock ──
    can_restock = timeline_ok and cash_ok and profitable

    if can_restock:
        # Full fulfillment: stock sale + restock + restocked sale
        enriched["restock_units"] = shortage
        enriched["restock_total_cost"] = restock_total_cost
        enriched["sale_units_restocked"] = shortage
        enriched["sale_total_revenue"] = round(immediate_revenue + restocked_revenue, 4)
        enriched["cash_spent_now"] = restock_total_cost
        enriched["order_status"] = "fulfilled_full"
        enriched["order_message"] = (
            f"Fulfilled {requested} units of {item_name}: "
            f"{immediate_sale} from stock + {shortage} via restock arriving {arrival}."
        )
    else:
        # Partial / no fulfillment — record only what we can sell from stock
        enriched["sale_total_revenue"] = immediate_revenue

        if immediate_sale > 0:
            reason_parts = []
            if not timeline_ok:
                reason_parts.append("restock would arrive after the delivery deadline")
            if not cash_ok:
                reason_parts.append("insufficient cash for restock")
            if not profitable:
                reason_parts.append("restock not profitable at current pricing")
            reason = "; ".join(reason_parts) or "restock blocked"

            enriched["order_status"] = "fulfilled_partial"
            enriched["order_message"] = (
                f"Partially fulfilled {item_name}: "
                f"{immediate_sale} of {requested} units from stock. "
                f"Remaining {shortage} could not be restocked ({reason})."
            )
        else:
            # Nothing in stock and restock blocked
            reason_parts = []
            if not timeline_ok:
                reason_parts.append("supplier cannot deliver before deadline")
            if not cash_ok:
                reason_parts.append("insufficient cash for restock")
            if not profitable:
                reason_parts.append("restock not profitable at current pricing")
            reason = "; ".join(reason_parts) or "restock blocked"

            enriched["order_status"] = "skipped_no_stock"
            enriched["order_message"] = (
                f"Could not fulfill {item_name}: out of stock and "
                f"restock blocked ({reason})."
            )

    return enriched


def _execute_order_transactions(outcome: dict, request_date: str) -> None:
    """
    Persist the ordering decisions to the transactions table.
    Called AFTER _compute_order_outcome has produced the plan.

    Date rules (per design):
      - Sale of in-stock units → request_date
      - Restock order (cost)   → restock_arrival_date (when supplier delivers)
      - Sale of restocked units → restock_arrival_date

    All three calls go through the starter-code create_transaction helper.
    """
    item_name = outcome.get("requested_item", "")
    sale_unit_price = float(outcome.get("sale_unit_price") or 0.0)
    restock_unit_cost = float(outcome.get("restock_unit_cost") or 0.0)
    arrival = outcome.get("restock_arrival_date")

    immediate_units = int(outcome.get("sale_units_immediate") or 0)
    restock_units = int(outcome.get("restock_units") or 0)
    restocked_sale_units = int(outcome.get("sale_units_restocked") or 0)

    # 1. Sale of immediately available stock
    if immediate_units > 0:
        try:
            create_transaction(
                item_name=item_name,
                transaction_type="sales",
                quantity=immediate_units,
                price=round(immediate_units * sale_unit_price, 4),
                date=request_date,
            )
        except Exception as e:
            print(f"  [order] Failed to record immediate sale for {item_name}: {e}")

    # 2. Restock order (cash leaves on arrival date)
    if restock_units > 0 and arrival:
        try:
            create_transaction(
                item_name=item_name,
                transaction_type="stock_orders",
                quantity=restock_units,
                price=round(restock_units * restock_unit_cost, 4),
                date=arrival,
            )
        except Exception as e:
            print(f"  [order] Failed to record restock for {item_name}: {e}")

    # 3. Sale of restocked units
    if restocked_sale_units > 0 and arrival:
        try:
            create_transaction(
                item_name=item_name,
                transaction_type="sales",
                quantity=restocked_sale_units,
                price=round(restocked_sale_units * sale_unit_price, 4),
                date=arrival,
            )
        except Exception as e:
            print(f"  [order] Failed to record restocked sale for {item_name}: {e}")


@tool
def process_order_for_item(item_record_json: str, request_date: str, available_cash: float) -> str:
    """
    Compute the ordering decision for ONE item: full fulfillment, partial,
    restock+sale, or skip — based on stock, deadline, cash, and profitability.

    NOTE: This tool DOES NOT write to the database. The verification layer
    (verify_and_fix_response) is the sole writer to ensure consistency and
    avoid double-spending. The agent calls this tool to produce a plan;
    the runner persists the plan deterministically.

    Args:
        item_record_json: JSON string of one quoted item.
        request_date: Today's date (YYYY-MM-DD) for this request.
        available_cash: Cash available BEFORE this item's restock spend.

    Returns:
        str: A JSON string of the enriched item with all ordering fields
             populated (sale_units_immediate, restock_units, order_status,
             order_message, etc.). NO transactions are written.
    """
    import json as _json
    try:
        item = _json.loads(item_record_json)
    except _json.JSONDecodeError:
        return _json.dumps({"error": "invalid item_record_json"})

    outcome = _compute_order_outcome(item, request_date, float(available_cash or 0.0))
    # Intentionally NOT calling _execute_order_transactions here —
    # the verification layer commits to DB instead (single writer).
    return _json.dumps(outcome)


# ─────────────────────────────────────────────
# ORDERING AGENT
# ─────────────────────────────────────────────

ORDERING_AGENT_PROMPT = """
You are the Ordering Agent for Munder Difflin.

═══════════════════════════════════════════════════════
THE MOST IMPORTANT RULE — READ THIS FIRST
═══════════════════════════════════════════════════════

YOU HAVE ZERO ORDERING AUTHORITY OF YOUR OWN.
You CANNOT decide whether to fulfill, partial-fulfill, restock, or skip.
You CANNOT compute prices, costs, timelines, or cash amounts.

The ONLY way to process an item is to call process_order_for_item.
The tool checks stock, timeline, cash, and profitability deterministically
AND commits the transactions to the database.

Every item in the input list MUST result in EXACTLY ONE call to
process_order_for_item. No exceptions. No skipping.

═══════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════

A JSON object like:
{
  "as_of_date": "YYYY-MM-DD",
  "available_cash": <float>,
  "items": [
    {
      "original_name": "...",
      "requested_item": "...",
      "delivery_deadline": "YYYY-MM-DD" or null,
      "requested_quantity": <int>,
      "stock_quantity": <int>,
      "deliverable_quantity": <int>,
      "shortage_quantity": <int>,
      "status": "ok" | "unresolved_item",
      "pricing_method": "historical_avg" | "bulk_discount" | null,
      "historical_avg_price": <float|null>,
      "discount_pct": <int|null>,
      "quote_status": "quoted" | "skipped_no_stock" | "skipped_unresolved"
    }
  ]
}

═══════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════

For EACH item in the input list (in order, NO skipping):

  1. Call process_order_for_item(
         item_record_json=<the item dict serialized to a JSON string>,
         request_date=<as_of_date>,
         available_cash=<current available cash>
     )

  2. Take the JSON the tool returns. Add it as-is to your output list.

  3. After each call, DECREMENT available_cash by the returned
     "cash_spent_now" value. This way, the next item's affordability check
     uses the running balance, not the original starting cash.

DO NOT modify the tool's output.
DO NOT compute anything yourself — the tool handles all math, decisions,
and database writes.
DO NOT skip any item, even ones with status "unresolved_item" or
quote_status "skipped_*". The tool handles those cases.

═══════════════════════════════════════════════════════
OUTPUT — JSON ONLY, NO PROSE, NO MARKDOWN
═══════════════════════════════════════════════════════

{
  "as_of_date": "YYYY-MM-DD",
  "items": [
    <exact JSON record returned by process_order_for_item, one per input item>
  ]
}
"""

ordering_agent = ToolCallingAgent(
    tools=[process_order_for_item],
    model=model,
    max_steps=15,
    name="ordering_agent",
    description=(
        "Finalises sales/restock decisions and commits transactions via tool calls. "
        "Pass a JSON string with as_of_date, available_cash, and the quoted items "
        "list (each item already has stock + pricing data from earlier agents). "
        "The tool decides per-item: full fulfillment, partial, restock+sale, or "
        "skip — based on stock, delivery deadline, cash, and profitability. "
        "Returns JSON where each item is enriched with: sale_units_immediate, "
        "sale_units_restocked, restock_units, restock_arrival_date, "
        "sale_total_revenue, cash_spent_now, order_status, order_message. "
        "Call this AFTER quoting_agent."
    ),
    instructions=ORDERING_AGENT_PROMPT,
)


# ─────────────────────────────────────────────
# ORCHESTRATOR — AGENT
# (Only intent classifier wired in for now)
# ─────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """
You are the Orchestrator Agent for Munder Difflin, a paper supply company.

You DO NOT take any actions yourself. You ONLY route requests to sub-agents
and emit their structured outputs.

═══════════════════════════════════════════════════════
HARD RULES — NEVER VIOLATE
═══════════════════════════════════════════════════════

1. NEVER claim that any order has been placed, fulfilled, or out of stock,
   on your own. Only sub-agents determine these facts.
2. NEVER invent prices, delivery confirmations, or stock information.
3. NEVER paraphrase, summarise, or rephrase JSON returned by sub-agents.
   Preserve their EXACT output, including capitalisation.
4. NEVER add prose before or after the structured output.
5. Call each sub-agent EXACTLY ONCE per request. Do not re-invoke.
6. NEVER short-circuit the workflow because items "look unfulfillable".
   Even if the customer asks for items you suspect are unavailable
   (balloons, exotic items, unusual sizes, large quantities, anything),
   you MUST still call BOTH intent_classifier AND inventory_agent.
   They handle those cases properly.
7. ALWAYS emit a response that begins with "=== CLASSIFICATION ===".
   If your output does not start with that marker, you have failed.
8. Your response MUST contain ALL FOUR markers: "=== CLASSIFICATION ===",
   "=== INVENTORY ===", "=== QUOTE ===", AND "=== ORDER ===" — all are
   mandatory, no exceptions.
   Even if customer items appear "all out of stock", "all weird", or
   "obviously unfulfillable", you STILL run the full workflow.
   Writing prose like "items are out of stock" instead of running the
   workflow is the WORST failure mode.

   ⚠️  CRITICAL FORMAT — your final answer MUST be PLAIN TEXT with the
   markers as literal lines. It is NOT a JSON object.

   CORRECT format (plain text with marker lines):
   === CLASSIFICATION ===
   {"request_type": "order", ...}

   === INVENTORY ===
   {"as_of_date": "...", ...}

   ❌ WRONG (do NOT do this — JSON wrapping the markers):
   {"=== CLASSIFICATION ===": "{\"request_type\": ...}", "=== INVENTORY ===": "..."}

   The markers are SECTION HEADERS, not JSON keys. Write them as bare lines.
9. You don't know stock levels yourself. You don't know if items can be
   fulfilled. Only inventory_agent knows. Always defer to it.

═══════════════════════════════════════════════════════
WORKFLOW (Phase 4 — classifier + inventory + quoting + ordering wired in)
═══════════════════════════════════════════════════════

For every customer request, NO EXCEPTIONS:

STEP 1 — Call intent_classifier sub-agent with the FULL raw request text.
         It returns a JSON with: request_type, delivery_deadline, items[].
         Each item has: original_name, resolved_name, quantity.

STEP 2 — Extract from the customer text:
         • request date (look for "Date of request: YYYY-MM-DD") → as_of_date
         • available cash (look for "Available cash: $...") → available_cash
           If not present, use 0.

STEP 3 — Call inventory_agent. Pass it a JSON STRING built from the
         classifier output, in this exact shape:
         {
           "as_of_date": "<request date>",
           "delivery_deadline": "<delivery_deadline from classifier, or null>",
           "items": [
             {"original_name": "<from classifier>",
              "resolved_name": "<from classifier>",
              "quantity": <from classifier>}
           ]
         }

         ⚠️  CRITICAL FIELD MAPPING RULES:
         • Use classifier's "resolved_name" for inventory lookups
         • Use classifier's exact "quantity" — NEVER change, round, or zero it
         • Pass through "original_name" exactly as the classifier returned it
         • Pass the request-level "delivery_deadline" so it flows through to each item
         • Include EVERY item from the classifier — even unresolved ones
         • Pass quantities as integers, not strings

         WRONG (do NOT do this):
         Classifier says: {"original_name": "5,000 sheets of A3 paper",
                           "resolved_name": "unresolved", "quantity": 5000}
         You pass: {"resolved_name": "A3 paper", "quantity": 0}    ❌

         RIGHT:
         You pass: {"original_name": "5,000 sheets of A3 paper",
                    "resolved_name": "unresolved", "quantity": 5000} ✓

STEP 4 — Call quoting_agent. Pass it a JSON STRING built from the
         inventory_agent output, exactly as it returned (do NOT modify):
         {
           "as_of_date": "<request date>",
           "items": <inventory_agent's items array, unchanged>
         }
         The quoting agent enriches each item with pricing fields.

STEP 5 — Call ordering_agent. Pass it a JSON STRING built from the
         quoting_agent output, plus the available_cash (you do NOT need to
         look this up — the test runner injects it via the request text;
         if absent, just use 0):
         {
           "as_of_date": "<request date>",
           "available_cash": <number>,
           "items": <quoting_agent's items array, unchanged>
         }
         The ordering agent finalises sales, restocks, and writes
         transactions to the database.

STEP 6 — Emit your final answer in EXACTLY this format:

=== CLASSIFICATION ===
<exact JSON returned by intent_classifier — character-for-character>

=== INVENTORY ===
<exact JSON returned by inventory_agent — character-for-character>

=== QUOTE ===
<exact JSON returned by quoting_agent — character-for-character>

=== ORDER ===
<exact JSON returned by ordering_agent — character-for-character>

=== SUMMARY ===
request_type: <copy from classification>
delivery_deadline: <copy from classification>
total_items: <count of items>
unresolved_items: <count where status == "unresolved_item">
items_with_shortage: <count where shortage_quantity > 0>
items_quoted: <count where quote_status == "quoted">
items_fulfilled: <count where order_status starts with "fulfilled">
items_skipped: <count where order_status starts with "skipped">
total_revenue: <sum of sale_total_revenue across all items>
═══════════════════════════════════════════════════════
EXAMPLE 1 — Normal order
═══════════════════════════════════════════════════════

=== CLASSIFICATION ===
{
  "request_type": "order",
  "delivery_deadline": "2025-04-15",
  "items": [
    {"original_name": "200 sheets of glossy paper", "resolved_name": "Glossy paper", "quantity": 200},
    {"original_name": "500 sheets of cardstock",    "resolved_name": "Cardstock",    "quantity": 500}
  ]
}

=== INVENTORY ===
{
  "as_of_date": "2025-04-01",
  "items": [
    {"original_name": "200 sheets of glossy paper", "requested_item": "Glossy paper", "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 300, "deliverable_quantity": 200, "shortage_quantity": 0,   "status": "ok"},
    {"original_name": "500 sheets of cardstock",    "requested_item": "Cardstock",    "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 150, "deliverable_quantity": 150, "shortage_quantity": 350, "status": "ok"}
  ]
}

=== QUOTE ===
{
  "as_of_date": "2025-04-01",
  "items": [
    {"original_name": "200 sheets of glossy paper", "requested_item": "Glossy paper", "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 300, "deliverable_quantity": 200, "shortage_quantity": 0, "status": "ok", "pricing_method": "bulk_discount", "historical_avg_price": null, "discount_pct": 5, "quote_status": "quoted"},
    {"original_name": "500 sheets of cardstock", "requested_item": "Cardstock", "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 150, "deliverable_quantity": 150, "shortage_quantity": 350, "status": "ok", "pricing_method": "bulk_discount", "historical_avg_price": null, "discount_pct": 5, "quote_status": "quoted"}
  ]
}

=== ORDER ===
{
  "as_of_date": "2025-04-01",
  "items": [
    {"original_name": "200 sheets of glossy paper", "requested_item": "Glossy paper", "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 300, "deliverable_quantity": 200, "shortage_quantity": 0, "status": "ok", "sale_units_immediate": 200, "sale_units_restocked": 0, "restock_units": 0, "sale_total_revenue": 38.0, "order_status": "fulfilled_stock_only", "order_message": "Fulfilled 200 units of Glossy paper from existing stock."},
    {"original_name": "500 sheets of cardstock", "requested_item": "Cardstock", "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 150, "deliverable_quantity": 150, "shortage_quantity": 350, "status": "ok", "sale_units_immediate": 150, "sale_units_restocked": 350, "restock_units": 350, "sale_total_revenue": 71.25, "order_status": "fulfilled_full", "order_message": "Fulfilled 500 units of Cardstock: 150 from stock + 350 via restock."}
  ]
}

=== SUMMARY ===
request_type: order
delivery_deadline: 2025-04-15
total_items: 2
unresolved_items: 0
items_with_shortage: 1
items_quoted: 2
items_fulfilled: 2
items_skipped: 0
total_revenue: 109.25

═══════════════════════════════════════════════════════
EXAMPLE 2 — Order with unresolved items (e.g. balloons)
You MUST still run the full workflow (classifier + inventory + quoting).
═══════════════════════════════════════════════════════

Customer request: "I need 500 sheets of poster paper, 300 streamers,
and 200 balloons. (Date of request: 2025-04-03)"

=== CLASSIFICATION ===
{
  "request_type": "order",
  "delivery_deadline": "2025-04-15",
  "items": [
    {"original_name": "500 sheets of poster paper", "resolved_name": "Poster paper",    "quantity": 500},
    {"original_name": "300 streamers",              "resolved_name": "Party streamers", "quantity": 300},
    {"original_name": "200 balloons",               "resolved_name": "unresolved",      "quantity": 200}
  ]
}

=== INVENTORY ===
{
  "as_of_date": "2025-04-03",
  "items": [
    {"original_name": "500 sheets of poster paper", "requested_item": "Poster paper",    "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 0,   "deliverable_quantity": 0,   "shortage_quantity": 500, "status": "ok"},
    {"original_name": "300 streamers",              "requested_item": "Party streamers", "delivery_deadline": "2025-04-15", "requested_quantity": 300, "stock_quantity": 250, "deliverable_quantity": 250, "shortage_quantity": 50,  "status": "ok"},
    {"original_name": "200 balloons",               "requested_item": "unresolved",      "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 0,   "deliverable_quantity": 0,   "shortage_quantity": 200, "status": "unresolved_item"}
  ]
}

=== QUOTE ===
{
  "as_of_date": "2025-04-03",
  "items": [
    {"original_name": "500 sheets of poster paper", "requested_item": "Poster paper", "delivery_deadline": "2025-04-15", "requested_quantity": 500, "stock_quantity": 0, "deliverable_quantity": 0, "shortage_quantity": 500, "status": "ok", "pricing_method": null, "historical_avg_price": null, "discount_pct": null, "quote_status": "skipped_no_stock"},
    {"original_name": "300 streamers", "requested_item": "Party streamers", "delivery_deadline": "2025-04-15", "requested_quantity": 300, "stock_quantity": 250, "deliverable_quantity": 250, "shortage_quantity": 50, "status": "ok", "pricing_method": "bulk_discount", "historical_avg_price": null, "discount_pct": 5, "quote_status": "quoted"},
    {"original_name": "200 balloons", "requested_item": "unresolved", "delivery_deadline": "2025-04-15", "requested_quantity": 200, "stock_quantity": 0, "deliverable_quantity": 0, "shortage_quantity": 200, "status": "unresolved_item", "pricing_method": null, "historical_avg_price": null, "discount_pct": null, "quote_status": "skipped_unresolved"}
  ]
}

=== ORDER ===
{
  "as_of_date": "2025-04-03",
  "items": [
    {"original_name": "500 sheets of poster paper", "requested_item": "Poster paper", "delivery_deadline": "2025-04-15", "order_status": "fulfilled_full", "order_message": "Fulfilled 500 units of Poster paper via restock arriving 2025-04-10."},
    {"original_name": "300 streamers", "requested_item": "Party streamers", "delivery_deadline": "2025-04-15", "order_status": "fulfilled_full", "order_message": "Fulfilled 300 units of Party streamers: 250 from stock + 50 via restock."},
    {"original_name": "200 balloons", "requested_item": "unresolved", "delivery_deadline": "2025-04-15", "order_status": "skipped_unresolved", "order_message": "'200 balloons' is not in our catalog."}
  ]
}

=== SUMMARY ===
request_type: order
delivery_deadline: 2025-04-15
total_items: 3
unresolved_items: 1
items_with_shortage: 3
items_quoted: 1
items_fulfilled: 2
items_skipped: 1
total_revenue: 137.5
"""

orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[intent_classifier_agent, inventory_agent, quoting_agent, ordering_agent],
    max_steps=15,  # classify + inventory + quote + order + final answer + headroom
    name="orchestrator",
    description="Central coordinator for Munder Difflin.",
    instructions=ORCHESTRATOR_PROMPT,
)


# ─────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine=db_engine)

    # ── Clean stale audit logs from previous runs so we get a fresh dataset ──
    for stale in [
        "test_results.csv",
        "intent_classifications.csv",
        "inventory_checks.csv",
        "quote-agent-response.csv",
        "order-agent-response.csv",
    ]:
        if os.path.exists(stale):
            os.remove(stale)
            print(f"  Removed stale {stale}")

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date").reset_index(drop=True)
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Financial state tracking
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # ── PHASE 4: Run ALL requests through the full multi-agent pipeline ──
    # (classifier + inventory + quoting + ordering)
    # Set to a smaller slice e.g. iloc[:3] for quick tests.
    test_sample = quote_requests_sample
    print(f"\nRunning {len(test_sample)} requests...\n")

    results = []
    for idx, row in test_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"Request {idx + 1} | {row['job']} | {row['event']}")
        print(f"Date: {request_date}  |  Cash: ${current_cash:.2f}  |  Inventory: ${current_inventory:.2f}")
        print(f"{'='*60}")

        # Append request date AND current cash balance so agents have full context
        request_with_date = (
            f"{row['request']} "
            f"(Date of request: {request_date}, Available cash: ${current_cash:.2f})"
        )
        print(f"Raw request:\n{request_with_date}\n")

        # ── Call the orchestrator (resilient + auto-retry on malformed output) ──
        def _is_valid_response(resp: str) -> bool:
            """All four required section markers must be present."""
            return (
                isinstance(resp, str)
                and "=== CLASSIFICATION ===" in resp
                and "=== INVENTORY ===" in resp
                and "=== QUOTE ===" in resp
                and "=== ORDER ===" in resp
            )

        response = ""
        for attempt in (1, 2):  # at most one retry
            try:
                response = orchestrator_agent.run(request_with_date)
            except Exception as e:
                response = f"ERROR: orchestrator failed — {type(e).__name__}: {e}"
                print(f"  ⚠️  Request {idx + 1} attempt {attempt} failed: {e}")
                continue

            if _is_valid_response(response):
                if attempt == 2:
                    print(f"  ✓ Request {idx + 1} succeeded on retry.")
                break
            else:
                print(f"  ⚠️  Request {idx + 1} attempt {attempt} returned malformed response. "
                      f"{'Retrying...' if attempt == 1 else 'Giving up.'}")
                time.sleep(2)  # brief backoff before retry

        # ── Verification layer: recompute inventory + quote + order sections
        # deterministically and commit transactions.
        # Pass raw_request so the verifier can rescue if orchestrator output
        # is too mangled to parse.
        verified_response = verify_and_fix_response(
            response, request_date, raw_request=request_with_date
        )

        # ── Save per-stage audit logs (deterministic, no LLM) ──
        # These keep the structured per-agent data for analysis & reporting.
        save_pipeline_section(
            response=verified_response,
            section_name="CLASSIFICATION",
            file_path="intent_classifications.csv",
            metadata={"request_id": idx + 1, "request_date": request_date},
        )
        save_pipeline_section(
            response=verified_response,
            section_name="INVENTORY",
            file_path="inventory_checks.csv",
            metadata={"request_id": idx + 1, "request_date": request_date},
        )
        save_pipeline_section(
            response=verified_response,
            section_name="QUOTE",
            file_path="quote-agent-response.csv",
            metadata={"request_id": idx + 1, "request_date": request_date},
        )
        save_pipeline_section(
            response=verified_response,
            section_name="ORDER",
            file_path="order-agent-response.csv",
            metadata={"request_id": idx + 1, "request_date": request_date},
        )

        # ── Finalizer Tool: build the customer-facing natural-language reply
        # for test_results.csv. Deterministic, derived from the verified data.
        customer_response = finalize_customer_response(verified_response)

        # Update financial state after processing
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"\nFinal customer response:\n{customer_response}")
        print(f"\nUpdated Cash: ${current_cash:.2f}  |  Inventory: ${current_inventory:.2f}")

        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "job": row["job"],
            "event": row["event"],
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": customer_response,
        })

        time.sleep(1)  # avoid rate limiting

    # Save partial results for review
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print(f"\n✅ Results saved to test_results.csv")

    # Final summary
    final_date = test_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print(f"\n===== FINANCIAL SNAPSHOT =====")
    print(f"Cash:      ${final_report['cash_balance']:.2f}")
    print(f"Inventory: ${final_report['inventory_value']:.2f}")
    print(f"Assets:    ${final_report['total_assets']:.2f}")

    return results


if __name__ == "__main__":
    results = run_test_scenarios()