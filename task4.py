"""
Itransition Task 4 - Part 1
Load / clean / analyse the STORE* datasets and emit the six required figures.

This is the original single-file pipeline with the fixes described in README.md:
  * DD.MM.YYYY dates are normalised to ISO *before* pandas sees them, so dateutil
    cannot silently read them month-first.
  * timestamps are asserted to parse, instead of being coerced to NaT and dropped.
  * year / month / day are extracted as real columns.
  * daily revenue is plotted with matplotlib.
  * ties in "most popular author" are reported rather than hidden by idxmax().

Run:  python task4_pipeline.py            (writes charts/ and results.json)
"""
import os
import re
import json
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORES = ["STORE1", "STORE2", "STORE3", "STORE4", "STORE5"]
EUR_TO_USD = 1.2

# --------------------------------------------------------------------------- prices
def parse_price(val):
    """'$ 35' / '30.0USD' / 'USD 25.25' / '32.50 EUR' / '46$99c' / '36. $' -> float USD."""
    if pd.isna(val):
        return None
    s = str(val).upper()
    is_eur = ("EUR" in s) or ("€" in s)
    nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", s)]
    if not nums:
        return None
    if len(nums) >= 2:
        # the currency symbol is being used as the decimal separator: 46$99c, EUR22c25
        amount = nums[0] + nums[1] / 100.0
    elif "¢" in s or "CENT" in s:
        amount = nums[0] / 100.0
    else:
        amount = nums[0]
    if is_eur:
        amount = round(amount * EUR_TO_USD, 2)   # settle on whole cents once, here
    return amount


# ----------------------------------------------------------------------- timestamps
# In this data the dotted form is DAY first and the slashed form is MONTH first.
# Verified across all 49,545 rows: field 1 of the dotted form reaches 31 while field 2
# never exceeds 12; the slashed form is the exact reverse. dateutil assumes month-first
# for both, so the dotted form has to be rewritten to ISO before pandas touches it.
_DOTTED = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")


def normalise_timestamp(s):
    s = str(s)
    s = _DOTTED.sub(lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}", s)
    s = s.replace("A.M.", "AM").replace("P.M.", "PM")
    return re.sub(r"[;,]", " ", s)


def clean_text(val):
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    return "" if s in ("null", "none", "nan") else s


def clean_phone(val):
    return "" if pd.isna(val) else re.sub(r"\D", "", str(val))


def parse_author_set(val):
    if pd.isna(val):
        return ()
    return tuple(sorted({a.strip() for a in str(val).split(",") if a.strip()}))


# --------------------------------------------------------------------------- per store
def process_store(store_name, data_root="data"):
    base_path = os.path.join(data_root, store_name)

    # 1. Load -------------------------------------------------------------------
    with open(os.path.join(base_path, "books.yaml"), "r", encoding="utf-8") as f:
        books_df = pd.DataFrame(yaml.safe_load(f))
    orders_df = pd.read_parquet(os.path.join(base_path, "orders.parquet"))
    users_df = pd.read_csv(os.path.join(base_path, "users.csv"))

    # 2. Books ------------------------------------------------------------------
    books_df.columns = books_df.columns.str.lstrip(":")
    books_df = books_df.drop_duplicates(subset=["id"])
    books_df["author_set"] = books_df["author"].apply(parse_author_set)
    books_df["year"] = pd.to_numeric(books_df["year"], errors="coerce").astype("Int64")
    books_df.loc[~books_df["year"].between(1000, 2030), "year"] = pd.NA
    unique_author_sets_count = books_df.loc[
        books_df["author_set"].map(len) > 0, "author_set"].nunique()

    # 3. Orders: timestamps -----------------------------------------------------
    orders_df = orders_df.drop_duplicates(subset=["id"])
    clean_ts = orders_df["timestamp"].map(normalise_timestamp)
    orders_df["parsed_ts"] = pd.to_datetime(clean_ts, format="mixed", dayfirst=False,
                                            errors="coerce")
    unparsed = int(orders_df["parsed_ts"].isna().sum())
    assert unparsed == 0, f"{store_name}: {unparsed} timestamps failed to parse"

    orders_df["date"] = orders_df["parsed_ts"].dt.strftime("%Y-%m-%d")
    orders_df["year"] = orders_df["parsed_ts"].dt.year          # required by the task
    orders_df["month"] = orders_df["parsed_ts"].dt.month
    orders_df["day"] = orders_df["parsed_ts"].dt.day

    # 4. Orders: money ----------------------------------------------------------
    orders_df["unit_price_usd"] = orders_df["unit_price"].apply(parse_price)
    bad_price = int(orders_df["unit_price_usd"].isna().sum())
    assert bad_price == 0, f"{store_name}: {bad_price} prices failed to parse"
    orders_df["quantity"] = pd.to_numeric(orders_df["quantity"], errors="coerce").astype(int)
    orders_df["paid_price"] = (orders_df["quantity"] * orders_df["unit_price_usd"]).round(2)

    daily_revenue = orders_df.groupby("date")["paid_price"].sum().sort_index().round(2)
    top_5_days = daily_revenue.sort_values(ascending=False).head(5)

    # 5. User reconciliation ----------------------------------------------------
    # Same person == agrees on >= 3 of (name, address, phone, email), i.e. at most one
    # field changed. Blocking on each 3-field subset gives that in linear time.
    users_df = users_df.drop_duplicates(subset=["id"])
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_keys = {}
    for row in users_df.itertuples(index=False):
        uid = int(row.id)
        parent[uid] = uid
        n, a = clean_text(row.name), clean_text(row.address)
        p, e = clean_phone(row.phone), clean_text(row.email)
        for k in [("nap", n, a, p), ("nae", n, a, e), ("npe", n, p, e), ("ape", a, p, e)]:
            if all(k[1:]):
                if k in seen_keys:
                    union(seen_keys[k], uid)
                else:
                    seen_keys[k] = uid

    user_to_root = {uid: find(uid) for uid in parent}
    unique_users_count = len(set(user_to_root.values()))

    # 6. Best buyer -------------------------------------------------------------
    orders_df["real_user_id"] = orders_df["user_id"].map(user_to_root)
    by_person = orders_df.groupby("real_user_id")["paid_price"].sum()
    top_root = by_person.idxmax()
    best_buyer_ids = sorted(uid for uid, r in user_to_root.items() if r == top_root)
    best_buyer_spend = round(float(by_person.max()), 2)
    best_buyer_names = sorted(
        {str(n) for n in users_df.loc[users_df["id"].isin(best_buyer_ids), "name"]})

    # 7. Most popular author set (ties reported) --------------------------------
    book_to_authors = dict(zip(books_df["id"], books_df["author_set"]))
    orders_df["author_set"] = orders_df["book_id"].map(book_to_authors)
    sold = orders_df.groupby("author_set")["quantity"].sum()
    sold = sold[sold.index.map(len) > 0]
    best = sold.max()
    most_popular = [", ".join(t) for t in sold[sold == best].index]

    return {
        "store": store_name,
        "orders": int(len(orders_df)),
        "total_revenue": round(float(orders_df["paid_price"].sum()), 2),
        "top_5_days": top_5_days,
        "unique_users": unique_users_count,
        "user_records": int(len(users_df)),
        "unique_author_sets": int(unique_author_sets_count),
        "most_popular_authors": most_popular,
        "most_popular_sold": int(best),
        "best_buyer_ids": best_buyer_ids,
        "best_buyer_names": best_buyer_names,
        "best_buyer_spend": best_buyer_spend,
        "daily_revenue": daily_revenue,
        "orders_df": orders_df,
    }


def plot_daily_revenue(res, out_dir="charts"):
    os.makedirs(out_dir, exist_ok=True)
    dr = res["daily_revenue"]
    idx = pd.to_datetime(dr.index)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(idx, dr.values, lw=0.8, color="#2a78d6", alpha=.55, label="Daily revenue")
    ax.plot(idx, dr.rolling(7, min_periods=1).mean().values, lw=1.9,
            color="#eb6834", label="7-day average")
    top = res["top_5_days"]
    ax.scatter(pd.to_datetime(top.index), top.values, s=26, zorder=5,
               color="#2a78d6", edgecolor="white", linewidth=1.1, label="Top-5 day")
    ax.set_title(f"{res['store']} - daily revenue (USD)", fontsize=12, loc="left")
    ax.set_ylabel("paid_price")
    ax.grid(axis="y", color="#dce1ea", lw=.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{res['store']}_daily_revenue.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


if __name__ == "__main__":
    summary = {}
    for store in STORES:
        res = process_store(store)
        chart = plot_daily_revenue(res)
        print(f"\n=== {res['store']} ===")
        print("Orders:", res["orders"], "| Total revenue: $%.2f" % res["total_revenue"])
        print("Top 5 days by revenue:")
        for d, v in res["top_5_days"].items():
            print(f"   {d}  ${v:,.2f}")
        print("Unique users:", res["unique_users"], f"(from {res['user_records']} records)")
        print("Unique author sets:", res["unique_author_sets"])
        print("Most popular author(s):", "; ".join(res["most_popular_authors"]),
              f"({res['most_popular_sold']} copies)")
        print("Best buyer IDs:", res["best_buyer_ids"],
              f"- {', '.join(res['best_buyer_names'])}, ${res['best_buyer_spend']:,.2f}")
        print("Chart:", chart)
        summary[store] = {
            "top_5_days": [{"date": d, "revenue": round(float(v), 2)}
                           for d, v in res["top_5_days"].items()],
            "unique_users": res["unique_users"],
            "unique_author_sets": res["unique_author_sets"],
            "most_popular_authors": res["most_popular_authors"],
            "best_buyer_ids": res["best_buyer_ids"],
            "total_revenue": res["total_revenue"],
        }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote results.json")