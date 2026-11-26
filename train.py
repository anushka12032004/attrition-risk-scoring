"""Employee attrition prediction & risk scoring.  Run: python train.py [csv]
Creates models.joblib, employees_scored.csv, metrics.csv, importance.csv, summary.json and figures/*.png"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
ENG = ["IncomePerYearExp", "PromotionDelayRatio", "PromotionDelayFlag", "EngagementScore",
       "WorkloadStressFlag", "CompaniesPerYear"]
LOW, HIGH = 0.30, 0.60


def engineer_df(X):
    X = X.copy()
    X["IncomePerYearExp"] = X["MonthlyIncome"] / (X["TotalWorkingYears"] + 1)
    X["PromotionDelayRatio"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
    X["PromotionDelayFlag"] = (X["YearsSinceLastPromotion"] >= 3).astype(int)
    X["EngagementScore"] = X[["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
                              "WorkLifeBalance", "JobInvolvement"]].mean(axis=1)
    X["WorkloadStressFlag"] = ((X["OverTime"] == "Yes") & ((X["WorkLifeBalance"] <= 2) |
                               (X["BusinessTravel"] == "Travel_Frequently"))).astype(int)
    X["CompaniesPerYear"] = X["NumCompaniesWorked"] / (X["TotalWorkingYears"] + 1)
    return X


def make_pipe(model, raw_cols):
    num = [c for c in raw_cols if c not in CAT] + ENG
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return Pipeline([("fe", FunctionTransformer(engineer_df, validate=False)), ("pre", pre), ("m", model)])


def zoo():
    return {
        "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                                            class_weight="balanced", random_state=SEED)}


def risk_cat(p, low=LOW, high=HIGH):
    return np.where(p < low, "Low", np.where(p <= high, "Medium", "High"))


def load(csv):
    d = pd.read_csv(csv)
    d.insert(0, "EmployeeID", np.arange(1, len(d) + 1))
    return d


def explain_rows(model, X, ref, cols):
    """Occlusion reason codes: drop in probability when a feature is replaced by the typical value."""
    base = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=X.index, columns=cols, dtype=float)
    for c in cols:
        Xm = X.copy(); Xm[c] = ref[c]
        out[c] = base - model.predict_proba(Xm)[:, 1]
    return base, out


def reference_row(X):
    return {c: (X[c].mode().iloc[0] if c in CAT else X[c].median()) for c in X.columns}


def reasons_text(contrib_row, xrow, n=3, thr=0.01):
    top = contrib_row.sort_values(ascending=False).head(n)
    return "; ".join(f"{k} = {xrow[k]}" for k, v in top.items() if v > thr) or "No strong risk driver"


def build(csv):
    d = load(csv)
    y = d["Attrition"].astype(int)
    raw = [c for c in d.columns if c not in ("EmployeeID", "Attrition")]
    X = d[raw]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, probs = [], {}, {}
    for name, m in zoo().items():
        pipe = make_pipe(m, raw).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
        cv = cross_validate(make_pipe(zoo()[name], raw), X, y, cv=skf, scoring="roc_auc")
        rows.append(dict(Model=name, Accuracy=accuracy_score(yte, pred), Precision=precision_score(yte, pred),
                         Recall=recall_score(yte, pred), F1=f1_score(yte, pred), ROC_AUC=roc_auc_score(yte, p),
                         CV_ROC_AUC=cv["test_score"].mean()))
        fitted[name], probs[name] = pipe, p
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values("CV_ROC_AUC", ascending=False).iloc[0].Model

    # calibrated probabilities: out-of-fold for dashboards, final model for what-if
    cal = lambda: CalibratedClassifierCV(estimator=make_pipe(zoo()[best], raw), method="sigmoid", cv=5)
    oof = cross_val_predict(cal(), X, y, cv=skf, method="predict_proba")[:, 1]
    final = cal().fit(X, y)
    ref = reference_row(X)
    p_final, contrib = explain_rows(final, X, ref, raw)
    scored = d[["EmployeeID", "Department", "JobRole", "OverTime", "MonthlyIncome", "JobSatisfaction", "Attrition"]].copy()
    scored["risk_prob"] = oof.round(4)
    scored["risk_category"] = risk_cat(oof)
    scored["model_prob"] = p_final.round(4)
    scored["top_reasons"] = [reasons_text(contrib.iloc[i], X.iloc[i]) for i in range(len(X))]
    imp = permutation_importance(fitted[best], Xte, yte, n_repeats=25, random_state=SEED, scoring="roc_auc")
    importance = pd.DataFrame({"Feature": raw, "Importance": imp.importances_mean}).sort_values("Importance", ascending=False)
    return dict(data=d, raw=raw, X=X, y=y, metrics=metrics, best=best, final=final, ref=ref, scored=scored,
                importance=importance, contrib=contrib, fitted=fitted, probs=probs, yte=yte, oof=oof)


def figures(r, out):
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    d, y, sc = r["data"], r["y"], r["scored"]
    plt.figure(figsize=(6, 5))
    for n, p in r["probs"].items():
        f, t, _ = roc_curve(r["yte"], p); plt.plot(f, t, label=f"{n} ({roc_auc_score(r['yte'], p):.2f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (hold-out)"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "01_roc.png", dpi=150); plt.close()
    cm = confusion_matrix(r["yte"], (r["probs"][r["best"]] >= .5).astype(int))
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
    for (i, j), v in np.ndenumerate(cm): plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["Stay", "Leave"]); plt.yticks([0, 1], ["Stay", "Leave"]); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix: {r['best']}"); plt.tight_layout(); plt.savefig(fig_dir / "02_confusion.png", dpi=150); plt.close()
    r["importance"].head(12)[::-1].plot.barh(x="Feature", y="Importance", legend=False, figsize=(7, 5), title="Permutation importance (AUC drop)")
    plt.tight_layout(); plt.savefig(fig_dir / "03_importance.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, c in zip(ax, ["OverTime", "MaritalStatus", "BusinessTravel"]):
        (d.groupby(c)["Attrition"].mean() * 100).plot.bar(ax=a, title=f"Attrition % by {c}"); a.set_xlabel("")
    plt.tight_layout(); plt.savefig(fig_dir / "04_attrition_by_group.png", dpi=150); plt.close()
    (d.groupby("JobRole")["Attrition"].mean() * 100).sort_values().plot.barh(figsize=(7, 4.5), title="Attrition % by job role")
    plt.tight_layout(); plt.savefig(fig_dir / "05_attrition_by_role.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(sc.risk_prob, bins=30); plt.axvline(LOW, c="orange"); plt.axvline(HIGH, c="red")
    plt.title("Distribution of attrition probability"); plt.xlabel("Probability"); plt.tight_layout(); plt.savefig(fig_dir / "06_risk_distribution.png", dpi=150); plt.close()
    sc.groupby("Department")["risk_prob"].mean().plot.bar(figsize=(5.5, 4), title="Average risk by department")
    plt.tight_layout(); plt.savefig(fig_dir / "07_department_risk.png", dpi=150); plt.close()


def main(csv, out="."):
    out = Path(out)
    r = build(csv)
    joblib.dump({k: r[k] for k in ("raw", "final", "ref", "best")}, out / "models.joblib")
    r["scored"].to_csv(out / "employees_scored.csv", index=False)
    r["metrics"].round(4).to_csv(out / "metrics.csv", index=False)
    r["importance"].to_csv(out / "importance.csv", index=False)
    figures(r, out)
    d, sc = r["data"], r["scored"]
    summ = {"rows": len(d), "attrition_rate": float(d.Attrition.mean()), "leavers": int(d.Attrition.sum()), "best_model": r["best"],
            "risk_counts": sc.risk_category.value_counts().to_dict(),
            "leaver_rate_by_risk": sc.groupby("risk_category")["Attrition"].mean().round(3).to_dict(),
            "avg_prob_leavers": float(sc[sc.Attrition == 1].risk_prob.mean()), "avg_prob_stayers": float(sc[sc.Attrition == 0].risk_prob.mean()),
            "dept_risk": sc.groupby("Department")["risk_prob"].mean().round(3).to_dict(),
            "role_risk": sc.groupby("JobRole")["risk_prob"].mean().round(3).sort_values(ascending=False).to_dict(),
            "top_importance": r["importance"].head(10).round(4).to_dict("records")}
    for c in ["OverTime", "MaritalStatus", "BusinessTravel", "Department", "JobRole"]:
        summ[f"attr_by_{c}"] = (d.groupby(c)["Attrition"].mean() * 100).round(1).to_dict()
    for c in ["MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "Age", "YearsSinceLastPromotion", "DistanceFromHome"]:
        summ[f"mean_{c}"] = d.groupby("Attrition")[c].mean().round(2).to_dict()
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=str)
    print(r["metrics"].round(3).to_string(index=False)); print("Best:", r["best"])
    print(json.dumps({k: summ[k] for k in ("risk_counts", "leaver_rate_by_risk", "avg_prob_leavers", "avg_prob_stayers")}, indent=1))
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Palo_Alto_Networks.csv")
