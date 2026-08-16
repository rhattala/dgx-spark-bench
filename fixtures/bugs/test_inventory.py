from inventory import Ledger

def test_isolation():
    a = Ledger(); a.add("x", 1)
    b = Ledger()
    assert b.total() == 0, "new ledgers must not share state"

def test_top_n():
    l = Ledger()
    for n, q in [("a", 5), ("b", 9), ("c", 1)]:
        l.add(n, q)
    assert [i["name"] for i in l.top_n(2)] == ["b", "a"]

def test_remove_all_duplicates():
    l = Ledger()
    l.add("dup", 1); l.add("dup", 2); l.add("keep", 3)
    l.remove("dup")
    assert [i["name"] for i in l.items] == ["keep"]

def test_remove_missing():
    l = Ledger(); l.add("a", 1)
    assert l.remove("nope") is False

if __name__ == "__main__":
    fails = 0
    for fn in [test_isolation, test_top_n, test_remove_all_duplicates, test_remove_missing]:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception as e:
            fails += 1; print("FAIL", fn.__name__, "-", e)
    print("FAILURES:", fails)
    raise SystemExit(1 if fails else 0)
