from pathlib import Path
import numpy as np, pandas as pd, joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import train as T

st.set_page_config(page_title="Employee Attrition Risk", page_icon="🧭", layout="wide")
CSV = next((p for p in ["Palo_Alto_Networks.csv", "data/Palo_Alto_Networks.csv"] if Path(p).exists()), None)
if CSV is None:
    st.error("Palo_Alto_Networks.csv not found next to app.py"); st.stop()


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    d = T.load(CSV)
    try:
        b = joblib.load("models.joblib")
        sc, metrics, imp = pd.read_csv("employees_scored.csv"), pd.read_csv("metrics.csv"), pd.read_csv("importance.csv")
    except Exception:                                   # artifacts missing or incompatible: retrain once
        r = T.build(CSV)
        b = {k: r[k] for k in ("raw", "final", "ref", "best")}
        sc, metrics, imp = r["scored"], r["metrics"], r["importance"]
    return b, sc, metrics, imp, d


B, SC, METRICS, IMP, D = load_all()
RAW, FINAL, REF = B["raw"], B["final"], B["ref"]
XALL = D[RAW]

# ------------------------------------------------ sidebar
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", sorted(SC.Department.unique()))
pool = SC[SC.Department.isin(depts)] if depts else SC
roles = st.sidebar.multiselect("Job role", sorted(pool.JobRole.unique()))
st.sidebar.subheader("Risk thresholds")
low = st.sidebar.slider("Low / Medium boundary", 0.05, 0.60, 0.30, 0.05)
high = st.sidebar.slider("Medium / High boundary", 0.20, 0.95, 0.60, 0.05)
if low >= high:
    st.sidebar.error("Low boundary must be below the High boundary"); st.stop()
st.sidebar.subheader("Employee")
top_id = int(SC.sort_values("risk_prob", ascending=False).EmployeeID.iloc[0])
emp = st.sidebar.number_input("Employee ID (1 to %d)" % len(SC), 1, len(SC), top_id, 1)

SC = SC.assign(risk_level=T.risk_cat(SC.risk_prob, low, high))
sf = SC.copy()
if depts: sf = sf[sf.Department.isin(depts)]
if roles: sf = sf[sf.JobRole.isin(roles)]
if sf.empty:
    st.warning("No employees match these filters."); st.stop()

st.title("🧭 Employee Attrition Prediction & Risk Scoring")
st.caption(f"Palo Alto Networks workforce, {len(SC):,} employees | model: {B['best']} | "
           "probabilities are calibrated, cross-validated (out-of-fold)")
t1, t2, t3, t4 = st.tabs(["Attrition risk dashboard", "Employee risk profile", "Department & role view", "Explainability & what-if"])

# ------------------------------------------------ 1 dashboard
with t1:
    n_high, n_med = int((sf.risk_level == "High").sum()), int((sf.risk_level == "Medium").sum())
    c = st.columns(5)
    c[0].metric("Employees", f"{len(sf):,}")
    c[1].metric("Average risk", f"{sf.risk_prob.mean():.1%}")
    c[2].metric("High risk", f"{n_high}", f"{n_high / len(sf):.1%} of group", delta_color="off")
    c[3].metric("Medium risk", f"{n_med}")
    c[4].metric("Actual attrition (recorded)", f"{sf.Attrition.mean():.1%}")
    a, b = st.columns(2)
    fig = px.histogram(sf, x="risk_prob", nbins=30, title="Distribution of attrition probability")
    fig.add_vline(x=low, line_color="orange", line_dash="dash"); fig.add_vline(x=high, line_color="red", line_dash="dash")
    a.plotly_chart(fig, width="stretch")
    cnt = sf.risk_level.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    cnt.columns = ["Risk", "Employees"]
    b.plotly_chart(px.bar(cnt, x="Risk", y="Employees", color="Risk", title="Employees by risk category",
                          color_discrete_map={"Low": "#2e9e6b", "Medium": "#f2a93b", "High": "#d64545"}), width="stretch")
    st.subheader("Highest-risk employees")
    st.dataframe(sf.sort_values("risk_prob", ascending=False).head(15)[
        ["EmployeeID", "Department", "JobRole", "risk_prob", "risk_level", "top_reasons"]].rename(
        columns={"risk_prob": "Attrition probability", "risk_level": "Risk", "top_reasons": "Main reasons"}).round(3),
        width="stretch", hide_index=True)

# ------------------------------------------------ 2 employee profile
with t2:
    row = D[D.EmployeeID == emp].iloc[0]
    s = SC[SC.EmployeeID == emp].iloc[0]
    p = float(s.risk_prob)
    g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                               title={"text": f"Employee {emp}: attrition probability ({s.risk_level} risk)"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                                      "steps": [{"range": [0, low * 100], "color": "#bfe8d2"},
                                                {"range": [low * 100, high * 100], "color": "#fbe3b0"},
                                                {"range": [high * 100, 100], "color": "#f4b8b8"}]}))
    g.update_layout(height=300)
    a, b = st.columns([1, 2])
    a.plotly_chart(g, width="stretch")
    a.caption(f"Recorded outcome: **{'Left the company' if row.Attrition == 1 else 'Still employed'}**")
    show = ["Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "BusinessTravel", "JobSatisfaction",
            "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion", "DistanceFromHome", "StockOptionLevel"]
    b.dataframe(pd.DataFrame({"Attribute": show, "Value": [str(row[k]) for k in show]}), hide_index=True, width="stretch")
    _, contrib = T.explain_rows(FINAL, XALL.loc[[row.name]], REF, RAW)
    cs = contrib.iloc[0].astype(float)
    sel = pd.concat([cs.sort_values(ascending=False).head(6), cs.sort_values().head(4)]).drop_duplicates()
    sel = sel[sel.abs() > 0.002].sort_values()
    lab = [f"{k} = {XALL.loc[row.name, k]}" for k in sel.index]
    fig = go.Figure(go.Bar(x=sel.values * 100, y=lab, orientation="h",
                           marker_color=["#d64545" if v > 0 else "#2e9e6b" for v in sel.values]))
    fig.update_layout(title="Key contributing factors (percentage points vs a typical employee)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.info("Reason codes: " + s.top_reasons)

# ------------------------------------------------ 3 department / role
with t3:
    dep = sf.groupby("Department").agg(Employees=("EmployeeID", "count"), Avg_risk=("risk_prob", "mean"),
                                       High_risk=("risk_level", lambda x: (x == "High").sum()),
                                       Recorded_attrition=("Attrition", "mean")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(dep, x="Department", y="Avg_risk", text=dep.Avg_risk.map("{:.1%}".format), title="Average risk by department"), width="stretch")
    rl = sf.groupby("JobRole").agg(Employees=("EmployeeID", "count"), Avg_risk=("risk_prob", "mean"),
                                   High_risk=("risk_level", lambda x: (x == "High").sum())).reset_index().sort_values("Avg_risk")
    b.plotly_chart(px.bar(rl, x="Avg_risk", y="JobRole", orientation="h", title="Average risk by job role"), width="stretch")
    pv = sf.pivot_table(index="JobRole", columns="Department", values="risk_prob", aggfunc="mean")
    st.plotly_chart(px.imshow(pv, text_auto=".0%", aspect="auto", color_continuous_scale="Reds", title="Average risk: role x department"), width="stretch")
    st.dataframe(dep.round(3), hide_index=True, width="stretch")
    st.dataframe(rl.round(3), hide_index=True, width="stretch")

# ------------------------------------------------ 4 explainability
with t4:
    a, b = st.columns(2)
    a.plotly_chart(px.bar(IMP.head(15)[::-1], x="Importance", y="Feature", orientation="h",
                          title=f"Feature importance ({B['best']}, permutation, AUC drop)"), width="stretch")
    b.subheader("Model comparison (hold-out)")
    b.dataframe(METRICS.round(3), hide_index=True, width="stretch")
    b.caption("Class imbalance handled with class weights; stratified 80/20 split; ROC-AUC also cross-validated (5-fold).")
    st.subheader(f"What-if scenario for employee {emp}")
    base = XALL.loc[[row.name]].copy()
    k = f"_{emp}"
    c = st.columns(4)
    new = base.copy()
    new["OverTime"] = c[0].selectbox("OverTime", ["No", "Yes"], index=["No", "Yes"].index(row.OverTime), key="ot" + k)
    new["BusinessTravel"] = c[1].selectbox("BusinessTravel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                           index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(row.BusinessTravel), key="bt" + k)
    new["JobSatisfaction"] = c[2].slider("JobSatisfaction", 1, 4, int(row.JobSatisfaction), key="js" + k)
    new["WorkLifeBalance"] = c[3].slider("WorkLifeBalance", 1, 4, int(row.WorkLifeBalance), key="wl" + k)
    c = st.columns(4)
    new["EnvironmentSatisfaction"] = c[0].slider("EnvironmentSatisfaction", 1, 4, int(row.EnvironmentSatisfaction), key="es" + k)
    new["MonthlyIncome"] = c[1].slider("MonthlyIncome", int(D.MonthlyIncome.min()), int(D.MonthlyIncome.max()), int(row.MonthlyIncome), 100, key="mi" + k)
    new["YearsSinceLastPromotion"] = c[2].slider("YearsSinceLastPromotion", 0, 15, int(row.YearsSinceLastPromotion), key="yp" + k)
    new["StockOptionLevel"] = c[3].slider("StockOptionLevel", 0, 3, int(row.StockOptionLevel), key="so" + k)
    p0, p1 = float(FINAL.predict_proba(base)[0, 1]), float(FINAL.predict_proba(new)[0, 1])
    m = st.columns(3)
    m[0].metric("Current profile", f"{p0:.1%}")
    m[1].metric("Scenario", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
    m[2].metric("Scenario risk category", str(T.risk_cat(np.array([p1]), low, high)[0]))
    st.caption("What-if uses the final model fitted on all employees, so the current-profile value can differ slightly from the cross-validated score above.")
