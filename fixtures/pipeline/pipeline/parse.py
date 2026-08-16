def parse_line(line):
    """'name,qty,price' -> dict. Blank/comment lines return None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    return {"name": parts[0], "qty": int(parts[1]), "price": float(parts[2])}
