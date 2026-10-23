import json, requests
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

URL = "https://raw.githubusercontent.com/cbuntain/umd.inst414/main/data/imdb_movies_2000to2022.prolific.json"
resp = requests.get(URL); resp.raise_for_status()
movies = [json.loads(l) for l in resp.text.splitlines() if l.strip()]

names, counts, nmovies = {}, {}, {}
for m in movies:
    for aid, aname in m["actors"]:
        names[aid] = aname
        nmovies[aid] = nmovies.get(aid, 0) + 1
        c = counts.setdefault(aid, {})
        for g in m["genres"]:
            c[g] = c.get(g, 0) + 1

df = pd.DataFrame.from_dict(counts, orient="index").fillna(0)
df = df.drop(columns=[c for c in df.columns if c.strip() == ""], errors="ignore")
df["n_movies"] = pd.Series(nmovies)
print("movies", len(movies), "actors", len(df))
df = df[df["n_movies"] >= 5].drop(columns="n_movies")   # enough history to judge
print("actors with >=5 movies", len(df))
shares = df.div(df.sum(axis=1), axis=0)  # genre mix, rows sum to 1

ks = range(2, 21); inert, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=414).fit(shares)
    inert.append(km.inertia_)
    sil.append(silhouette_score(shares, km.labels_, sample_size=5000, random_state=414))
    print(k, round(km.inertia_,1), round(sil[-1],3))
pd.DataFrame({"k":list(ks),"inertia":inert,"silhouette":sil}).to_csv("k_selection.csv", index=False)

# ---- final model ----
K = 6
km = KMeans(n_clusters=K, n_init=10, random_state=414).fit(shares)
shares["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=shares.columns[:-1]).clip(lower=0)
sizes = shares["cluster"].value_counts().sort_index()
for c in range(K):
    top = cent.loc[c].sort_values(ascending=False).head(4)
    members = shares[shares.cluster == c].drop(columns="cluster")
    d = np.linalg.norm(members.values - km.cluster_centers_[c], axis=1)
    m = pd.DataFrame({"dist": d, "n": [nmovies[a] for a in members.index]}, index=members.index)
    ex = m[m.n >= 15].sort_values("dist").head(5)
    print(f"\nCluster {c} size {sizes[c]}:", ", ".join(f"{g} {v:.0%}" for g, v in top.items()))
    print("  examples:", "; ".join(f"{names[a]} ({nmovies[a]})" for a in ex.index))
    print("  most prolific:", "; ".join(f"{names[a]} ({nmovies[a]})" for a in m.sort_values('n', ascending=False).head(4).index))
cent.round(3).to_csv("centroids.csv")
shares.assign(name=[names[a] for a in shares.index])[["name","cluster"]].to_csv("actor_clusters.csv")

# figures
ks_df = pd.read_csv("k_selection.csv")
fig, ax1 = plt.subplots(figsize=(9,4.5))
ax1.plot(ks_df.k, ks_df.inertia, marker="o", color="#4a4a4a"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia", color="#4a4a4a")
ax2 = ax1.twinx(); ax2.plot(ks_df.k, ks_df.silhouette, marker="s", color="#1f6fd1"); ax2.set_ylabel("Silhouette", color="#1f6fd1")
ax1.axvline(6, ls="--", color="#999"); ax1.set_xticks(ks_df.k)
plt.title("k = 6: inertia bends and silhouette peaks"); plt.tight_layout(); plt.savefig("fig_k_selection.png", dpi=150)

keep = cent.max(axis=0).sort_values(ascending=False).head(12).index
plt.figure(figsize=(10,4.2))
plt.imshow(cent[keep], cmap="Blues", aspect="auto")
plt.xticks(range(len(keep)), keep, rotation=45, ha="right"); plt.yticks(range(K), [f"Cluster {c}" for c in range(K)])
for i in range(K):
    for j,g in enumerate(keep):
        plt.text(j, i, f"{cent.loc[i,g]:.0%}", ha="center", va="center", fontsize=8, color="white" if cent.loc[i,g]>0.25 else "black")
plt.title("Average genre share of each cluster's movies"); plt.tight_layout(); plt.savefig("fig_cluster_profiles.png", dpi=150)

# ---- sanity checks ----
feat = shares.drop(columns="cluster")
assert np.allclose(feat.sum(axis=1), 1), "genre shares should sum to 1"
assert feat.isna().sum().sum() == 0
tt = [k for k, v in names.items() if v == "Tony Todd"][0]
print("Tony Todd raw counts:", {g: int(v) for g, v in df.loc[tt].items() if v > 0})
print("checks passed; actors clustered:", len(feat))
