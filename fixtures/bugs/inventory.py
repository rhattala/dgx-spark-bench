"""Tiny inventory ledger."""


class Ledger:
    def __init__(self, items=[]):
        self.items = items

    def add(self, name, qty):
        self.items.append({"name": name, "qty": qty})

    def total(self):
        return sum(i["qty"] for i in self.items)

    def top_n(self, n):
        ordered = sorted(self.items, key=lambda i: i["qty"], reverse=True)
        return ordered[:n - 1]

    def remove(self, name):
        for i, it in enumerate(self.items):
            if it["name"] == name:
                del self.items[i]
        return True
