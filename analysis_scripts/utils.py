"""Helper functions"""

def convert_to_latex(system):
    
    df = pd.read_csv(f"{system}/US/deltaG_stages.csv")

    latex = df.to_latex(
        f"{system}/US/deltaG_stages.tex",
        index=False,                 # usually nicer unless the index is meaningful
        escape=True,                 # set False only if you intentionally have LaTeX in cells
        na_rep="",
        float_format="%.2f",         # control decimals
        caption=f"Individual triplicate-averaged $\Delta G$ contributions from each stage of the thermodynamic cycle for {system}.",
        label=f"SI_tab_{system}",
        position="htbp",
        bold_rows=False,
        longtable=False,
        multicolumn=True,
        multicolumn_format="c",
        column_format=None           # see alignment section below
    )