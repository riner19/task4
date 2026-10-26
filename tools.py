from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from task4 import process_store, STORES


class Warehouse:
    """Cleaned frames for every store, prepared once at start-up."""

    def __init__(self, data_root: str = "data", stores=None):
        self.stores = list(stores or STORES)
        self.by_store = {}
        for s in self.stores:
            self.by_store[s] = self._build(s, data_root)

    def _build(self, store: str, data_root: str):
        res = process_store(store, data_root)
        books = res["books_df"].copy()
        users = res["users_df"].copy()
        orders = res["orders_df"].copy()

        users["person_id"] = users["id"].map(res["user_to_root"])
        books["author_display"] = books["author_set"].map(lambda t: ", ".join(t))

        orders = orders.merge(
            books[["id", "title", "author_display", "genre", "publisher", "year"]]
            .rename(columns={"id": "book_id", "year": "book_year"}),
            on="book_id", how="left")
        orders = orders.merge(
            users[["id", "name", "email", "person_id"]]
            .rename(columns={"id": "user_id", "name": "user_name", "email": "user_email"}),
            on="user_id", how="left")

        bs = (orders.groupby("book_id")
                    .agg(copies_sold=("quantity", "sum"),
                         revenue=("paid_price", "sum"),
                         orders=("id", "count"))
                    .reset_index())
        books = books.merge(bs, left_on="id", right_on="book_id", how="left")
        for c in ("copies_sold", "orders"):
            books[c] = books[c].fillna(0).astype(int)
        books["revenue"] = books["revenue"].fillna(0.0).round(2)

        spend = (orders.groupby("person_id")
                       .agg(total_spent=("paid_price", "sum"),
                            orders=("id", "count"),
                            copies=("quantity", "sum"))
                       .reset_index())
        spend["total_spent"] = spend["total_spent"].round(2)
        people = (users.groupby("person_id")
                       .agg(user_ids=("id", lambda s: sorted(int(x) for x in s)),
                            names=("name", lambda s: sorted({str(x) for x in s})),
                            emails=("email", lambda s: sorted({str(x) for x in s if pd.notna(x)})))
                       .reset_index())
        people = people.merge(spend, on="person_id", how="left")
        people["total_spent"] = people["total_spent"].fillna(0.0)
        for c in ("orders", "copies"):
            people[c] = people[c].fillna(0).astype(int)

        sets_ = (orders.dropna(subset=["author_set"])
                       .groupby("author_display")
                       .agg(copies_sold=("quantity", "sum"),
                            revenue=("paid_price", "sum"),
                            orders=("id", "count"))
                       .reset_index())
        sets_["revenue"] = sets_["revenue"].round(2)

        return {
            "res": res, "orders": orders, "books": books,
            "users": users, "people": people, "author_sets": sets_,
        }

    def get(self, store):
        if store not in self.by_store:
            raise ToolError(f"unknown store {store!r}; available: {', '.join(self.stores)}")
        return self.by_store[store]


class ToolError(Exception):
    pass


def _rows(df: pd.DataFrame, limit: int, cols=None):
    """Truncate for the model but always report the true match count."""
    out = df if cols is None else df[cols]
    return {
        "matched": int(len(df)),
        "returned": int(min(limit, len(df))),
        "rows": json.loads(out.head(limit).to_json(orient="records")),
    }


def list_stores(wh, **_):
    """Which datasets exist."""
    return {"stores": wh.stores,
            "note": "Each store is an independent dataset, cleaned and reconciled separately."}


def store_overview(wh, store: str, **_):
    d = wh.get(store)
    r, o = d["res"], d["orders"]
    return {
        "store": store,
        "orders": int(len(o)),
        "total_revenue_usd": round(float(o["paid_price"].sum()), 2),
        "copies_sold": int(o["quantity"].sum()),
        "average_order_value_usd": round(float(o["paid_price"].mean()), 2),
        "unique_users": int(r["unique_users"]),
        "user_records_in_file": int(r["user_records"]),
        "unique_author_sets": int(r["unique_author_sets"]),
        "books_in_catalogue": int(len(d["books"])),
        "first_order_date": str(o["date"].min()),
        "last_order_date": str(o["date"].max()),
    }


def top_days_by_revenue(wh, store: str, n: int = 5, **_):
    d = wh.get(store)
    s = d["res"]["daily_revenue"].sort_values(ascending=False).head(int(n))
    return {"store": store, "days": [{"date": d_, "revenue_usd": round(float(v), 2)}
                                     for d_, v in s.items()]}


def revenue_in_period(wh, store: str, start_date: str = None, end_date: str = None,
                      group_by: str = "total", **_):
    d = wh.get(store)
    o = d["orders"]
    m = pd.Series(True, index=o.index)
    if start_date:
        m &= o["date"] >= str(start_date)
    if end_date:
        m &= o["date"] <= str(end_date)
    sub = o[m]
    if sub.empty:
        return {"store": store, "start_date": start_date, "end_date": end_date,
                "orders": 0, "revenue_usd": 0.0, "note": "no orders in this period"}
    base = {"store": store, "start_date": start_date or str(o["date"].min()),
            "end_date": end_date or str(o["date"].max()),
            "orders": int(len(sub)),
            "revenue_usd": round(float(sub["paid_price"].sum()), 2),
            "copies_sold": int(sub["quantity"].sum())}
    if group_by == "day":
        g = sub.groupby("date")["paid_price"].sum().round(2)
        base["by_day"] = [{"date": k, "revenue_usd": float(v)} for k, v in g.items()][:120]
    elif group_by == "month":
        g = sub.groupby(sub["date"].str.slice(0, 7))["paid_price"].sum().round(2)
        base["by_month"] = [{"month": k, "revenue_usd": float(v)} for k, v in g.items()]
    return base


def count_unique_users(wh, store: str, **_):
    r = wh.get(store)["res"]
    return {
        "store": store,
        "unique_users": int(r["unique_users"]),
        "user_records_in_file": int(r["user_records"]),
        "duplicate_records_merged": int(r["user_records"] - r["unique_users"]),
        "rule": "Two records are the same person when they agree on at least three of "
                "name, address, phone and email (at most one field changed), applied "
                "transitively.",
    }


def count_author_sets(wh, store: str, **_):
    d = wh.get(store)
    return {
        "store": store,
        "unique_author_sets": int(d["res"]["unique_author_sets"]),
        "books_in_catalogue": int(len(d["books"])),
        "definition": "A set is the exact group of authors credited on a book. A pair "
                      "writing together counts separately from each writing alone.",
    }


def top_author_sets(wh, store: str, n: int = 5, by: str = "copies_sold", **_):
    d = wh.get(store)
    by = by if by in ("copies_sold", "revenue", "orders") else "copies_sold"
    s = d["author_sets"].sort_values(by, ascending=False).head(int(n))
    return {"store": store, "ranked_by": by,
            "author_sets": [{"authors": r.author_display, "copies_sold": int(r.copies_sold),
                             "revenue_usd": float(r.revenue), "orders": int(r.orders)}
                            for r in s.itertuples()]}


def author_stats(wh, store: str, author_name: str, **_):
    """Sales for one author, counting every book they appear on, solo or shared."""
    d = wh.get(store)
    q = str(author_name).strip().lower()
    books = d["books"]
    hit = books[books["author_display"].str.lower().str.contains(q, regex=False, na=False)]
    if hit.empty:
        return {"store": store, "author_query": author_name, "found": False,
                "note": "no author matching that text"}
    return {
        "store": store, "author_query": author_name, "found": True,
        "books": int(len(hit)),
        "copies_sold": int(hit["copies_sold"].sum()),
        "revenue_usd": round(float(hit["revenue"].sum()), 2),
        "author_sets_they_appear_in": sorted(hit["author_display"].unique().tolist())[:25],
        "top_books": json.loads(
            hit.sort_values("copies_sold", ascending=False)
               .head(5)[["title", "copies_sold", "revenue"]].to_json(orient="records")),
    }


def top_customers(wh, store: str, n: int = 5, **_):
    d = wh.get(store)
    p = d["people"].sort_values("total_spent", ascending=False).head(int(n))
    return {"store": store,
            "customers": [{"user_ids": r.user_ids, "names": r.names, "emails": r.emails,
                           "total_spent_usd": float(r.total_spent), "orders": int(r.orders),
                           "copies": int(r.copies)} for r in p.itertuples()],
            "note": "user_ids lists every record belonging to that one real person."}


def lookup_customer(wh, store: str, query: str, **_):
    """Find a customer by user_id, name fragment or e-mail fragment."""
    d = wh.get(store)
    users, people = d["users"], d["people"]
    q = str(query).strip()
    if q.isdigit():
        hit = users[users["id"] == int(q)]
    else:
        ql = q.lower()
        hit = users[users["name"].astype(str).str.lower().str.contains(ql, na=False)
                    | users["email"].astype(str).str.lower().str.contains(ql, na=False)]
    if hit.empty:
        return {"store": store, "query": query, "found": False}
    pids = sorted(set(hit["person_id"].tolist()))
    rows = people[people["person_id"].isin(pids)]
    return {"store": store, "query": query, "found": True,
            "people_matched": int(len(rows)),
            "people": [{"user_ids": r.user_ids, "names": r.names, "emails": r.emails,
                        "total_spent_usd": float(r.total_spent), "orders": int(r.orders)}
                       for r in rows.head(10).itertuples()]}


def search_books(wh, store: str, title_contains: str = None, author_contains: str = None,
                 genre: str = None, year_min: int = None, year_max: int = None,
                 sort_by: str = "copies_sold", limit: int = 10, **_):
    d = wh.get(store)
    b = d["books"]
    m = pd.Series(True, index=b.index)
    if title_contains:
        m &= b["title"].astype(str).str.lower().str.contains(
            str(title_contains).lower(), regex=False, na=False)
    if author_contains:
        m &= b["author_display"].str.lower().str.contains(
            str(author_contains).lower(), regex=False, na=False)
    if genre:
        m &= b["genre"].astype(str).str.lower() == str(genre).lower()
    if year_min is not None:
        m &= b["year"].notna() & (b["year"] >= int(year_min))
    if year_max is not None:
        m &= b["year"].notna() & (b["year"] <= int(year_max))
    sub = b[m]
    sort_by = sort_by if sort_by in ("copies_sold", "revenue", "year", "title") else "copies_sold"
    sub = sub.sort_values(sort_by, ascending=(sort_by == "title"))
    out = _rows(sub, int(limit),
                ["id", "title", "author_display", "genre", "year", "copies_sold", "revenue"])
    out["store"] = store
    out["total_copies_sold_by_matches"] = int(sub["copies_sold"].sum())
    out["total_revenue_usd_by_matches"] = round(float(sub["revenue"].sum()), 2)
    return out


def search_users(wh, store: str, email_starts_with: str = None, email_contains: str = None,
                 name_contains: str = None, limit: int = 10, **_):
    """Search the raw user records (one row per record, not per reconciled person)."""
    d = wh.get(store)
    u = d["users"]
    m = pd.Series(True, index=u.index)
    if email_starts_with:
        letters = [c.lower() for c in str(email_starts_with) if c.isalnum()]
        if letters:
            m &= u["email"].astype(str).str.lower().str[0].isin(letters)
    if email_contains:
        m &= u["email"].astype(str).str.lower().str.contains(
            str(email_contains).lower(), regex=False, na=False)
    if name_contains:
        m &= u["name"].astype(str).str.lower().str.contains(
            str(name_contains).lower(), regex=False, na=False)
    sub = u[m]
    out = _rows(sub, int(limit), ["id", "name", "email", "phone", "person_id"])
    out["store"] = store
    out["distinct_people_among_matches"] = int(sub["person_id"].nunique())
    return out


def genre_breakdown(wh, store: str, n: int = 10, **_):
    d = wh.get(store)
    g = (d["orders"].groupby("genre")
         .agg(copies_sold=("quantity", "sum"), revenue=("paid_price", "sum"),
              orders=("id", "count"))
         .reset_index().sort_values("revenue", ascending=False).head(int(n)))
    return {"store": store,
            "genres": [{"genre": r.genre, "copies_sold": int(r.copies_sold),
                        "revenue_usd": round(float(r.revenue), 2), "orders": int(r.orders)}
                       for r in g.itertuples()]}


def compare_stores(wh, metric: str = "total_revenue", **_):
    """Rank all stores on one metric."""
    keys = {
        "total_revenue": lambda d: round(float(d["orders"]["paid_price"].sum()), 2),
        "orders": lambda d: int(len(d["orders"])),
        "unique_users": lambda d: int(d["res"]["unique_users"]),
        "unique_author_sets": lambda d: int(d["res"]["unique_author_sets"]),
        "copies_sold": lambda d: int(d["orders"]["quantity"].sum()),
        "books_in_catalogue": lambda d: int(len(d["books"])),
    }
    if metric not in keys:
        raise ToolError(f"unknown metric {metric!r}; available: {', '.join(keys)}")
    vals = [{"store": s, metric: keys[metric](wh.by_store[s])} for s in wh.stores]
    vals.sort(key=lambda r: r[metric], reverse=True)
    return {"metric": metric, "ranking": vals,
            "highest": vals[0]["store"], "lowest": vals[-1]["store"]}



def _store_param(required=True):
    p = {"type": "string", "enum": STORES,
         "description": "Which store dataset to query."}
    return p


TOOLS = [
    (list_stores, "List the available store datasets.", {"type": "object", "properties": {}}),
    (store_overview,
     "Headline figures for one store: order count, total revenue, copies sold, average "
     "order value, unique users, unique author sets, catalogue size and date range. "
     "Use this first for broad questions about a store.",
     {"type": "object", "properties": {"store": _store_param()}, "required": ["store"]}),
    (top_days_by_revenue,
     "The highest-revenue days for a store, as YYYY-MM-DD with the revenue for each.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "n": {"type": "integer", "description": "How many days to return (default 5)."}},
      "required": ["store"]}),
    (revenue_in_period,
     "Total revenue, order count and copies sold between two dates, optionally broken "
     "down by day or month. Use for any 'how much did we make in <period>' question.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "start_date": {"type": "string", "description": "Inclusive, YYYY-MM-DD."},
         "end_date": {"type": "string", "description": "Inclusive, YYYY-MM-DD."},
         "group_by": {"type": "string", "enum": ["total", "day", "month"]}},
      "required": ["store"]}),
    (count_unique_users,
     "How many real unique people the store has after reconciling duplicate records, "
     "plus how many raw records were merged and the rule used.",
     {"type": "object", "properties": {"store": _store_param()}, "required": ["store"]}),
    (count_author_sets,
     "How many distinct sets of authors appear in a store's catalogue.",
     {"type": "object", "properties": {"store": _store_param()}, "required": ["store"]}),
    (top_author_sets,
     "The best-selling author sets for a store, ranked by copies sold, revenue or order "
     "count. Use for 'most popular author' questions.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "n": {"type": "integer"},
         "by": {"type": "string", "enum": ["copies_sold", "revenue", "orders"]}},
      "required": ["store"]}),
    (author_stats,
     "Sales for one named author across every book they appear on, solo or co-written.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "author_name": {"type": "string", "description": "Full or partial author name."}},
      "required": ["store", "author_name"]}),
    (top_customers,
     "The biggest spenders in a store. Each entry lists every user_id belonging to that "
     "one real person. Use for 'best buyer' questions.",
     {"type": "object", "properties": {
         "store": _store_param(), "n": {"type": "integer"}},
      "required": ["store"]}),
    (lookup_customer,
     "Find a customer by user_id, name fragment or e-mail fragment, and return their "
     "merged identity and spending.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "query": {"type": "string", "description": "A user_id, part of a name, or part "
                                                    "of an e-mail address."}},
      "required": ["store", "query"]}),
    (search_books,
     "Search the catalogue by title text, author text, genre or publication year. Returns "
     "the number of matches plus a sample of rows with their sales. Use this to COUNT "
     "books matching a condition - read the 'matched' field for the count.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "title_contains": {"type": "string"},
         "author_contains": {"type": "string"},
         "genre": {"type": "string"},
         "year_min": {"type": "integer"},
         "year_max": {"type": "integer"},
         "sort_by": {"type": "string", "enum": ["copies_sold", "revenue", "year", "title"]},
         "limit": {"type": "integer", "description": "Rows to return (default 10)."}},
      "required": ["store"]}),
    (search_users,
     "Search raw user records by e-mail first letter, e-mail text or name text. Returns "
     "the number of matches plus a sample. Use this to COUNT users matching a condition - "
     "read the 'matched' field.",
     {"type": "object", "properties": {
         "store": _store_param(),
         "email_starts_with": {"type": "string",
                               "description": "One or more letters, e.g. 's' or 'sp' to "
                                              "match e-mails starting with s or p."},
         "email_contains": {"type": "string"},
         "name_contains": {"type": "string"},
         "limit": {"type": "integer"}},
      "required": ["store"]}),
    (genre_breakdown,
     "Revenue and copies sold per genre for a store.",
     {"type": "object", "properties": {
         "store": _store_param(), "n": {"type": "integer"}},
      "required": ["store"]}),
    (compare_stores,
     "Rank every store on one metric. Use for questions comparing stores or asking which "
     "store is highest or lowest.",
     {"type": "object", "properties": {
         "metric": {"type": "string",
                    "enum": ["total_revenue", "orders", "unique_users",
                             "unique_author_sets", "copies_sold", "books_in_catalogue"]}},
      "required": ["metric"]}),
]

REGISTRY = {fn.__name__: (fn, desc, schema) for fn, desc, schema in TOOLS}


def openai_schema():
    return [{"type": "function",
             "function": {"name": n, "description": d, "parameters": s}}
            for n, (_, d, s) in REGISTRY.items()]


def call_tool(wh: Warehouse, name: str, args: dict):
    """Dispatch one tool call. Errors come back as data so the model can recover."""
    entry = REGISTRY.get(name)
    if entry is None:
        return {"error": f"unknown tool {name!r}"}
    fn = entry[0]
    try:
        return fn(wh, **(args or {}))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:                      # never let a tool crash the chat
        return {"error": f"{type(e).__name__}: {e}"}