"""
Summary-statistics tables for the reframed Data and Background section.

Two output tables (LaTeX), matching the paper's three levels of analysis:

  Table 1  (tab:sumstats_nm) -- two panels, New Mexico only:
     Panel A: ZIP-code level (the border-exposure panel design, eq. 1-2)
     Panel B: dispensary level (the supply-side mechanism check, eq. 6)

  Table 2  (tab:sumstats_state) -- county level, New Mexico vs. Colorado
     (the cross-state entry-regime comparison, eq. 4-5)

Each panel splits by border / interior using the paper's own predetermined,
geography-only classification (centroid within 75 miles of a nonlegal-state
line AND >= 1,000 nonlegal-state residents within that same 75-mile reach).

Writes to:
  New Mexico project/figures and tables/tab_sumstats_nm.tex
  New Mexico project/figures and tables/tab_sumstats_state.tex
and copies both into Latest try/reframe/tables/ .
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
REFRAME_TBL = os.path.join(os.path.dirname(BASE), "Latest try", "reframe", "tables")
EG_TBL = os.path.join(os.path.dirname(BASE), "entry game-marijuana", "tables")


def stat_row(label, series, scale=1.0, fmt="{:,.1f}"):
    s = pd.to_numeric(series, errors="coerce").dropna() * scale
    if len(s) == 0:
        return f"{label} & -- & -- & -- & -- & 0 \\\\"
    return (f"{label} & {fmt.format(s.mean())} & {fmt.format(s.std())} & "
            f"{fmt.format(s.min())} & {fmt.format(s.max())} & {len(s):,} \\\\")


# ----------------------------------------------------------------------
# TABLE 1 -- New Mexico ZIP-level (Panel A) and dispensary-level (Panel B)
# ----------------------------------------------------------------------
def table1():
    # ---- Panel A: ZIP-code level ----
    z = pd.read_csv(os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv"), dtype={"zip5": str})
    zc = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str})
    # revenue_per_dispensary and dispensary-conditional density only defined where >=1 dispensary
    z["dispensary_per_1000"] = z["dispensary_per_capita"]  # already per-1,000 in this file? verify below
    # nm_zip_panel_zeroinclusive: dispensary_per_capita is n/pop*1000 per build script
    zdisp = zc.copy()

    zz = z.drop_duplicates("zip5")
    n_border = int((zz.border_zip == 1).sum())
    n_interior = int((zz.border_zip == 0).sum())
    q = sorted(z.yq.unique())

    def panelA(df, dfd, tag):
        rows = []
        rows.append(stat_row("Revenue per capita (\\$/quarter)", df["revenue_per_capita"], fmt="{:,.1f}"))
        rows.append(stat_row("Dispensaries per 1,000 residents", df["dispensary_per_capita"], fmt="{:,.3f}"))
        rows.append(stat_row("Revenue per dispensary (\\$000s/quarter)",
                             dfd["revenue_per_dispensary"], scale=1e-3, fmt="{:,.1f}"))
        rows.append(stat_row("ZIP population", df.drop_duplicates('zip5')["zip_population"], fmt="{:,.0f}"))
        rows.append(stat_row("Distance to nonlegal state line (mi)",
                             df.drop_duplicates('zip5')["dist_stateline_mi"], fmt="{:,.1f}"))
        rows.append(stat_row("Nonlegal population within 75 mi (000s)",
                             df.drop_duplicates('zip5')["nonlegal_pop_within_75mi"], scale=1e-3, fmt="{:,.1f}"))
        return rows

    all_A = panelA(z, zdisp, "all")
    bor_A = panelA(z[z.border_zip == 1], zdisp[zdisp.border_zip == 1], "border")
    int_A = panelA(z[z.border_zip == 0], zdisp[zdisp.border_zip == 0], "interior")

    # ---- Panel B: dispensary level ----
    d = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"),
                    dtype={"zip5_nearest": str, "fips5": str})
    lp = pd.read_csv(os.path.join(TBL, "nm_dispensary_local_population.csv"))
    d = d.merge(lp[["dispensary_id", "pop_1km_areal"]], on="dispensary_id", how="left")

    dd = d.drop_duplicates("dispensary_id")
    nb_d = int((dd.border_dispensary == 1).sum())
    ni_d = int((dd.border_dispensary == 0).sum())

    def panelB(df):
        rows = []
        rows.append(stat_row("Revenue (\\$000s/quarter)", df["total_sales"], scale=1e-3, fmt="{:,.1f}"))
        rows.append(stat_row("Population within 1 km", df.drop_duplicates('dispensary_id')["pop_1km_areal"], fmt="{:,.0f}"))
        rows.append(stat_row("Population within 25 mi (000s)",
                             df.drop_duplicates('dispensary_id')["pop_within_25mi"], scale=1e-3, fmt="{:,.1f}"))
        rows.append(stat_row("Competing dispensaries within 25 mi",
                             df["local_dispensary_count_25mi"], fmt="{:,.0f}"))
        rows.append(stat_row("Dispensary age (quarters)", df["dispensary_age_q"], fmt="{:,.1f}"))
        rows.append(stat_row("Distance to nonlegal state line (mi)",
                             df.drop_duplicates('dispensary_id')["dist_stateline_mi"], fmt="{:,.1f}"))
        return rows

    all_B = panelB(d)
    bor_B = panelB(d[d.border_dispensary == 1])
    int_B = panelB(d[d.border_dispensary == 0])

    # ---- assemble LaTeX ----
    def block(rows_all, rows_b, rows_i):
        # merge the three column-groups: label & all(5) ... we instead print
        # THREE separate mini-tables? No -- one table, columns = All | Border | Interior
        # each stat_row already emits "label & mean & sd & min & max & n".
        # We want label & mean_all & mean_bor & mean_int  (compact). Rebuild.
        return None

    # Compact form: label | mean (all) | mean (border) | mean (interior) | N border ZIPs shown once
    def compact(series_fmt):
        pass

    # Simpler + more standard: full mean/sd for All, then mean for Border and Interior.
    def parse(r):
        parts = [p.strip() for p in r.rstrip("\\").split("&")]
        return parts  # [label, mean, sd, min, max, n]

    def compact_rows(rall, rbor, rint):
        out = []
        for a, b, i in zip(rall, rbor, rint):
            pa, pb, pi = parse(a), parse(b), parse(i)
            out.append(f"{pa[0]} & {pa[1]} & {pa[2]} & {pb[1]} & {pi[1]} \\\\")
        return out

    A = compact_rows(all_A, bor_A, int_A)
    B = compact_rows(all_B, bor_B, int_B)

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Summary statistics, New Mexico}")
    tex.append(r"\label{tab:sumstats_nm}")
    tex.append(r"\footnotesize")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r" & \multicolumn{2}{c}{Full sample} & Border & Interior \\")
    tex.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}")
    tex.append(r"Variable & Mean & SD & Mean & Mean \\")
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{5}{l}{\textit{Panel A: ZIP-code level (quarterly, 2022Q2--" + q[-1] + r")}} \\")
    tex.extend(A)
    tex.append(r"\addlinespace")
    tex.append(rf"ZIP codes & {len(zz):,} & & {n_border:,} & {n_interior:,} \\")
    tex.append(rf"ZIP--quarters & {len(z):,} & & {int((z.border_zip==1).sum()):,} & {int((z.border_zip==0).sum()):,} \\")
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{5}{l}{\textit{Panel B: Dispensary level (quarterly, 2022Q2--" + q[-1] + r")}} \\")
    tex.extend(B)
    tex.append(r"\addlinespace")
    tex.append(rf"Dispensaries & {dd.dispensary_id.nunique():,} & & {nb_d:,} & {ni_d:,} \\")
    tex.append(rf"Dispensary--quarters & {len(d):,} & & {int((d.border_dispensary==1).sum()):,} & {int((d.border_dispensary==0).sum()):,} \\")
    tex.append(rf"Distinct brands & {d.brand.nunique():,} & & & \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\begin{minipage}{0.92\textwidth}")
    tex.append(r"\vspace{0.4em}\footnotesize")
    tex.append(r"\textit{Notes:} New Mexico, 2022Q2 through " + q[-1] +
               r". A ZIP code (Panel A) or dispensary (Panel B) is classified as \emph{border} "
               r"if its centroid lies within 75 miles of the New Mexico line with a state that has no "
               r"operating recreational cannabis market (Texas, Oklahoma, or Utah) and at least 1{,}000 "
               r"residents of that nonlegal state live within the same 75-mile reach; Arizona and Colorado "
               r"are legal-market neighbors and count as interior. The classification uses geography only "
               r"and does not depend on where dispensaries locate. Panel A revenue and density outcomes use "
               r"all 368 populated New Mexico ZIP codes (zeros retained); revenue per dispensary is defined "
               r"only for ZIP--quarters with at least one operating dispensary. Panel B local-resident "
               r"population is the 2020 Census block population within a 1-km radius of each dispensary.")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{table}")
    txt = "\n".join(tex) + "\n"
    for p in (os.path.join(OUT, "tab_sumstats_nm.tex"), os.path.join(REFRAME_TBL, "tab_sumstats_nm.tex")):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    print("wrote tab_sumstats_nm.tex")
    print(f"  ZIP: {len(zz)} ({n_border} border / {n_interior} interior); "
          f"dispensaries: {dd.dispensary_id.nunique()} ({nb_d} border / {ni_d} interior)")


# ----------------------------------------------------------------------
# TABLE 2 -- county level, New Mexico vs. Colorado
# ----------------------------------------------------------------------
def table2():
    mq = pd.read_csv(os.path.join(EG_TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    cb = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    geo_border = cb.groupby("state_abbr")["border_county"].agg(["sum", "count"]).to_dict("index")
    mq = mq[mq.state_abbr.isin(["CO", "NM"])].copy()
    mq = mq[(mq.year < 2025) | ((mq.year == 2025) & (mq.quarter <= 2))].copy()
    mq = mq.merge(cb[["fips5", "border_county", "dist_stateline_mi", "nonlegal_pop_within_75mi"]],
                  on="fips5", how="left")

    REGIME = {"NM": "Free entry", "CO": "Local option / capped"}
    RETAIL = {"NM": "1 April 2022", "CO": "1 January 2014"}

    def col(state):
        d = mq[mq.state_abbr == state]
        dd = d.drop_duplicates("fips5")
        out = {}
        out["regime"] = REGIME[state]
        out["retail"] = RETAIL[state]
        out["ncounty"] = dd.fips5.nunique()
        out["nborder"] = int((dd.border_county == 1).sum())
        out["ninterior"] = int((dd.border_county == 0).sum())
        out["geo_total"] = int(geo_border[state]["count"])
        out["geo_border"] = int(geo_border[state]["sum"])
        out["ncq"] = len(d)
        out["ncq_border"] = int((d.border_county == 1).sum())
        out["rpc"] = pd.to_numeric(d["revenue_per_capita"], errors="coerce").mean()
        out["dens"] = pd.to_numeric(d["dispensary_per_1000"], errors="coerce").mean()
        out["rpd"] = pd.to_numeric(d["revenue_per_store"], errors="coerce").mean() / 1e3
        out["pop"] = pd.to_numeric(dd["population"], errors="coerce").mean()
        out["nlp"] = pd.to_numeric(dd.loc[dd.border_county == 1, "nonlegal_pop_within_75mi"],
                                   errors="coerce").mean() / 1e3
        return out

    nm, co = col("NM"), col("CO")

    def r(label, key, fmt="{:,.1f}"):
        def f(v):
            return "--" if pd.isna(v) else fmt.format(v)
        return f"{label} & {f(nm[key])} & {f(co[key])} \\\\"

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Cross-state entry-regime comparison: county-level sample, New Mexico and Colorado}")
    tex.append(r"\label{tab:sumstats_state}")
    tex.append(r"\footnotesize")
    tex.append(r"\begin{tabular}{lcc}")
    tex.append(r"\toprule")
    tex.append(r" & New Mexico & Colorado \\")
    tex.append(r"\midrule")
    tex.append(rf"Entry regime & {nm['regime']} & {co['regime']} \\")
    tex.append(rf"First recreational retail sales & {nm['retail']} & {co['retail']} \\")
    tex.append(r"\addlinespace")
    tex.append(rf"Counties (total) & {nm['geo_total']:,} & {co['geo_total']:,} \\")
    tex.append(rf"\quad Border counties (geography) & {nm['geo_border']:,} & {co['geo_border']:,} \\")
    tex.append(r"\addlinespace")
    tex.append(rf"Counties in estimation sample & {nm['ncounty']:,} & {co['ncounty']:,} \\")
    tex.append(rf"\quad Border counties in sample & {nm['nborder']:,} & {co['nborder']:,} \\")
    tex.append(rf"\quad Interior counties in sample & {nm['ninterior']:,} & {co['ninterior']:,} \\")
    tex.append(rf"County--quarters & {nm['ncq']:,} & {co['ncq']:,} \\")
    tex.append(rf"\quad Border county--quarters & {nm['ncq_border']:,} & {co['ncq_border']:,} \\")
    tex.append(r"\addlinespace")
    tex.append(r"\multicolumn{3}{l}{\textit{County-quarter means}} \\")
    tex.append(r(r"\quad Revenue per capita (\$)", "rpc", "{:,.1f}"))
    tex.append(r(r"\quad Dispensaries per 1,000 residents", "dens", "{:,.3f}"))
    tex.append(r(r"\quad Revenue per dispensary (\$000s)", "rpd", "{:,.1f}"))
    tex.append(r(r"\quad County population", "pop", "{:,.0f}"))
    tex.append(r(r"\quad Nonlegal pop.\ within 75 mi, border counties (000s)", "nlp", "{:,.1f}"))
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\begin{minipage}{0.86\textwidth}")
    tex.append(r"\vspace{0.4em}\footnotesize")
    tex.append(r"\textit{Notes:} County-quarter panel, 2015Q1 (Colorado) / 2022Q2 (New Mexico) through 2025Q2, "
               r"the sample used for the cross-state entry-regime comparison (equations~\ref{state_specific} "
               r"and~\ref{entry_barrier_reg}). A county is \emph{border} if its centroid lies within 75 miles "
               r"of a state line with a neighbor that has no operating recreational market and at least 1{,}000 "
               r"residents of that nonlegal state live within the same 75-mile reach. New Mexico's nonlegal "
               r"neighbors are Texas, Oklahoma, and Utah; Colorado's are Utah, Wyoming, Nebraska, Kansas, and "
               r"Oklahoma (Arizona and, from April 2022, New Mexico are legal). The estimation sample is "
               r"smaller than the geographic county count because a county enters only in the year it first "
               r"reports dispensary activity; many of Colorado's rural eastern-plains and Utah-border counties "
               r"never do. New Mexico is coded free entry; Colorado is coded locally capped because it delegates "
               r"licensing to local governments and, as of 2019, 48.4 percent of its counties banned "
               r"recreational cannabis businesses outright \citep{PayanBrownSong2021}. Montana is also in the "
               r"pooled three-state specification (equation~\ref{entry_barrier_reg}); the headline "
               r"counterfactual is the New Mexico--Colorado pairwise comparison.")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{table}")
    txt = "\n".join(tex) + "\n"
    for p in (os.path.join(OUT, "tab_sumstats_state.tex"), os.path.join(REFRAME_TBL, "tab_sumstats_state.tex")):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    print("wrote tab_sumstats_state.tex")
    print(f"  NM: {nm['ncounty']} counties ({nm['nborder']} border), {nm['ncq']} county-quarters")
    print(f"  CO: {co['ncounty']} counties ({co['nborder']} border), {co['ncq']} county-quarters")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REFRAME_TBL, exist_ok=True)
    table1()
    table2()
