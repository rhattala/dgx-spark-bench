import sys
from pipeline import parse_line, aggregate

def main(path):
    rows = [parse_line(l) for l in open(path)]
    result = aggregate(rows)
    for name in sorted(result):
        v = result[name]
        print("%s qty=%d revenue=%.2f" % (name, v["qty"], v["revenue"]))

if __name__ == "__main__":
    main(sys.argv[1])
