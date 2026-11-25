"""Stream-parse GPL family soft files into probe->gene maps."""
import gzip, pandas as pd, sys

def parse(path, out, symbol_col):
    ids, syms = [], []
    header = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if header is None:
                if line.startswith("ID\t"):
                    header = line.rstrip("\n").split("\t")
                    i_id = 0
                    i_sym = header.index(symbol_col)
                continue
            if line.startswith("!Platform_table_end") or line.startswith("^"):
                if ids: break
                continue
            vals = line.rstrip("\n").split("\t")
            if len(vals) < len(header): continue
            pid = vals[0].strip().strip('"')
            sym = vals[i_sym].strip().strip('"')
            ids.append(pid); syms.append(sym)
    df = pd.DataFrame({"probe": ids, "symbol": syms}).drop_duplicates("probe")
    df.to_csv(out, index=False)
    print(path, "->", df.shape, "mapped:", (df["symbol"] != "").sum())

parse("/workspace/data/GPL6244_family.soft.gz", "/workspace/data/gpl6244_map.csv", "GB_LIST")
parse("/workspace/data/GPL6104_family.soft.gz", "/workspace/data/gpl6104_map.csv", "Symbol")
parse("/workspace/data/GPL8432_family.soft.gz", "/workspace/data/gpl8432_map.csv", "Symbol")
