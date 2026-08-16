def aggregate(rows):
    """Sum qty and revenue per name."""
    out = {}
    for r in rows:
        k = r["name"]
        if k not in out:
            out[k] = {"qty": 0, "revenue": 0}
        out[k]["qty"] += r["qty"]
        out[k]["revenue"] += r["qty"] * r["price"]
    return out
