# ============================================
# Steam games analysis + Cytoscape exports
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import combinations

plt.style.use("ggplot")  # default theme

# -------------------------------
# 1. Load data
# -------------------------------

df = pd.read_csv("games.csv", encoding="utf-8", low_memory=False)

print("Original shape:", df.shape)
print(df.head(3))

print("\nSample raw 'Release date' values (first 20):")
print(df["Release date"].head(20).tolist())
# NOTE: In this dataset 'Release date' column is not a real date,
# so it is NOT used as a time variable in this project.

# -------------------------------
# 2. Basic cleaning & features
# -------------------------------

# 2.1 Owners (numeric)
df["Owners"] = df["Estimated owners"]

# 2.2 Review-based success metrics
df["TotalReviews"] = df["Positive"] + df["Negative"]
df = df[df["TotalReviews"] > 0].copy()  # need at least 1 review

df["PositivePct"] = df["Positive"] / df["TotalReviews"]

# 2.3 Keep only rows with non-missing genres, publishers, developers
df = df.dropna(subset=["Genres", "Publishers", "Developers"]).copy()

# 2.4 Simplify genres: main genre = first in list
df["MainGenre"] = df["Genres"].str.split(",").str[0].str.strip()

# 2.5 Make a cleaned genre list for later (for co-occurrence)
df["GenreList"] = (
    df["Genres"]
    .astype(str)
    .str.split(",")
    .apply(lambda lst: [g.strip() for g in lst if g.strip()])
)

# 2.6 Basic price cleaning
df = df[df["Price"] >= 0].copy()

print("\nCleaned shape after basic filtering:", df.shape)

# -------------------------------
# 3. Indie vs AAA classification
# -------------------------------

publisher_counts = df["Publishers"].value_counts()


def classify_publisher(pub):
    if publisher_counts.get(pub, 0) <= 2:
        return "Indie"
    else:
        return "AAA"


df["PublisherType"] = df["Publishers"].apply(classify_publisher)

print("\nPublisher type counts:")
print(df["PublisherType"].value_counts())

# -------------------------------
# 4. Descriptive overview
# -------------------------------

print("\nBasic numeric description:")
print(df[["Price", "Owners", "TotalReviews", "PositivePct"]].describe())

print("\nTop 10 main genres by count:")
print(df["MainGenre"].value_counts().head(10))

# -------------------------------
# 5. RQ1, RQ2, RQ4 – main analyses
# -------------------------------

# ---------- RQ1: Do some genres have higher ratings? ----------

genre_stats = (
    df.groupby("MainGenre")
    .agg(
        Count=("AppID", "count"),
        AvgPositivePct=("PositivePct", "mean"),
        AvgTotalReviews=("TotalReviews", "mean"),
        AvgOwners=("Owners", "mean"),
    )
    .sort_values("AvgPositivePct", ascending=False)
)

print("\nGenre stats (top 15 by AvgPositivePct):")
print(genre_stats.head(15))

# Plot: Average rating by genre (>= 100 games)
min_games_for_genre = 100
genre_stats_plot = genre_stats[genre_stats["Count"] >= min_games_for_genre]

plt.figure(figsize=(12, 4))
genre_stats_plot["AvgPositivePct"].sort_values(ascending=False).plot(kind="bar")
plt.ylabel("Average positive review ratio")
plt.title("Average rating by main genre (genres with ≥ 100 games)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ---------- RQ2: How does price correlate with success? ----------

corr_price_totalreviews = df["Price"].corr(df["TotalReviews"])
corr_price_owners = df["Price"].corr(df["Owners"])
corr_price_positivepct = df["Price"].corr(df["PositivePct"])

print("\nCorrelations with price:")
print("Price vs TotalReviews:", corr_price_totalreviews)
print("Price vs Owners:", corr_price_owners)
print("Price vs PositivePct:", corr_price_positivepct)

# Scatter: price vs total reviews
plt.figure(figsize=(10, 4))
plt.scatter(df["Price"], df["TotalReviews"], alpha=0.3, s=5)
plt.yscale("log")
plt.xlabel("Price")
plt.ylabel("Total reviews (log scale)")
plt.title("Price vs total reviews")
plt.tight_layout()
plt.show()

# Scatter: price vs owners
plt.figure(figsize=(10, 4))
plt.scatter(df["Price"], df["Owners"], alpha=0.3, s=5)
plt.yscale("log")
plt.xlabel("Price")
plt.ylabel("Owners (log scale)")
plt.title("Price vs estimated owners")
plt.tight_layout()
plt.show()

# ---------- RQ3 (new): Free vs paid ----------

df["IsFree"] = df["Price"] == 0

free_paid_stats = (
    df.groupby("IsFree")
    .agg(
        Count=("AppID", "count"),
        AvgOwners=("Owners", "mean"),
        MedianOwners=("Owners", "median"),
        AvgTotalReviews=("TotalReviews", "mean"),
        MedianTotalReviews=("TotalReviews", "median"),
        AvgPositivePct=("PositivePct", "mean"),
    )
)

print("\nFree vs paid stats:")
print(free_paid_stats)

owners_paid = df.loc[~df["IsFree"], "Owners"]
owners_free = df.loc[df["IsFree"], "Owners"]

rating_paid = df.loc[~df["IsFree"], "PositivePct"]
rating_free = df.loc[df["IsFree"], "PositivePct"]

# Boxplot: owners distribution (log scale, outliers hidden)
plt.figure(figsize=(8, 4))
plt.boxplot(
    [owners_paid, owners_free],
    tick_labels=["Paid", "Free"],
    showfliers=False,
)
plt.yscale("log")
plt.title("Estimated owners: free vs paid (outliers hidden)")
plt.ylabel("Owners (log scale)")
plt.tight_layout()
plt.show()

# Boxplot: rating distribution (outliers hidden)
plt.figure(figsize=(8, 4))
plt.boxplot(
    [rating_paid, rating_free],
    tick_labels=["Paid", "Free"],
    showfliers=False,
)
plt.title("Positive review ratio: free vs paid (outliers hidden)")
plt.ylabel("Positive review ratio")
plt.tight_layout()
plt.show()

# ---------- RQ4: Indie vs AAA success ----------

publisher_stats = (
    df.groupby("PublisherType")
    .agg(
        Count=("AppID", "count"),
        AvgOwners=("Owners", "mean"),
        MedianOwners=("Owners", "median"),
        AvgTotalReviews=("TotalReviews", "mean"),
        MedianTotalReviews=("TotalReviews", "median"),
        AvgPositivePct=("PositivePct", "mean"),
    )
)

print("\nIndie vs AAA stats:")
print(publisher_stats)

rating_aaa = df.loc[df["PublisherType"] == "AAA", "PositivePct"]
rating_indie = df.loc[df["PublisherType"] == "Indie", "PositivePct"]

owners_aaa = df.loc[df["PublisherType"] == "AAA", "Owners"]
owners_indie = df.loc[df["PublisherType"] == "Indie", "Owners"]

# Boxplot: rating distribution by publisher type (outliers hidden)
plt.figure(figsize=(8, 4))
plt.boxplot(
    [rating_aaa, rating_indie],
    tick_labels=["AAA", "Indie"],
    showfliers=False,
)
plt.title("Positive review ratio by publisher type")
plt.ylabel("Positive review ratio")
plt.tight_layout()
plt.show()

# Boxplot: owners distribution by publisher type (log scale, outliers hidden)
plt.figure(figsize=(8, 4))
plt.boxplot(
    [owners_aaa, owners_indie],
    tick_labels=["AAA", "Indie"],
    showfliers=False,
)
plt.yscale("log")
plt.title("Estimated owners by publisher type (outliers hidden)")
plt.ylabel("Owners (log scale)")
plt.tight_layout()
plt.show()

# ============================================
# 6. RQ3 (new): Genre co-occurrence network
# ============================================

# Only keep games with at least 2 genres
df_multi_genre = df[df["GenreList"].apply(len) >= 2].copy()

cooc_edges = []

for genres in df_multi_genre["GenreList"]:
    # remove duplicates within a game, sort for consistency
    unique_genres = sorted(set(genres))
    if len(unique_genres) < 2:
        continue
    for g1, g2 in combinations(unique_genres, 2):
        cooc_edges.append((g1, g2, 1))

edges_genre = pd.DataFrame(cooc_edges, columns=["Source", "Target", "Weight"])

# Sum weights for each pair
edges_genre = (
    edges_genre.groupby(["Source", "Target"], as_index=False)["Weight"].sum()
)

print("\nTotal edges in genre co-occurrence network:", len(edges_genre))

# Build node table: one row per genre with success metrics
nodes_genre = (
    df.explode("GenreList")
    .groupby("GenreList")
    .agg(
        Count=("AppID", "count"),
        AvgPositivePct=("PositivePct", "mean"),
        AvgOwners=("Owners", "mean"),
    )
    .reset_index()
    .rename(columns={"GenreList": "Node"})
)

print("Total genre nodes:", len(nodes_genre))

# Save for Cytoscape
nodes_genre.to_csv("nodes_genre_cooc.csv", index=False)
edges_genre.to_csv("edges_genre_cooc.csv", index=False)

print("\nSaved genre co-occurrence network files:")
print("  nodes_genre_cooc.csv")
print("  edges_genre_cooc.csv")

# ============================================
# 7. Developer–Publisher network for Cytoscape
# ============================================

def clean_company_name(name):
    if pd.isna(name):
        return ""
    name = str(name)

    suffixes = [
        r"\bLTD\b",
        r"\bLLC\b",
        r"\bINC\b",
        r"\bCO\b",
        r"\bCO\.\b",
        r"\bS\.A\.\b",
        r"\bGMBH\b",
        r"\bBV\b",
        r"\bAB\b",
        r"\bSRL\b",
        r"\bSAS\b",
        r"\bKG\b",
        r"\bPLC\b",
        r"\bK\.K\.\b",
    ]

    for suf in suffixes:
        name = re.sub(suf, "", name, flags=re.IGNORECASE)

    # Remove trailing punctuation and extra spaces
    name = re.sub(r"[,\.;]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


df["Developers_clean"] = df["Developers"].apply(clean_company_name)
df["Publishers_clean"] = df["Publishers"].apply(clean_company_name)


def split_and_clean(values):
    if pd.isna(values):
        return []
    parts = (
        str(values)
        .replace(";", ",")
        .replace("|", ",")
        .split(",")
    )
    return [p.strip() for p in parts if p.strip()]


df["DevList"] = df["Developers_clean"].apply(split_and_clean)
df["PubList"] = df["Publishers_clean"].apply(split_and_clean)

edges = []

for _, row in df.iterrows():
    devs = row["DevList"]
    pubs = row["PubList"]
    for d in devs:
        for p in pubs:
            edges.append([d, p, 1])  # weight 1 per game

edges_df = pd.DataFrame(edges, columns=["Developer", "Publisher", "Weight"])
edges_df = edges_df.groupby(["Developer", "Publisher"], as_index=False).sum()

print("\nTotal edges in full dev–pub network:", len(edges_df))

# Filter to active devs/pubs
dev_degree = edges_df.groupby("Developer")["Weight"].sum()
pub_degree = edges_df.groupby("Publisher")["Weight"].sum()

MIN_DEV_GAMES = 5   # dev must appear in at least 5 games
MIN_PUB_GAMES = 10  # pub must appear in at least 10 games

keep_devs = dev_degree[dev_degree >= MIN_DEV_GAMES].index
keep_pubs = pub_degree[pub_degree >= MIN_PUB_GAMES].index

filtered_edges = edges_df[
    edges_df["Developer"].isin(keep_devs)
    & edges_df["Publisher"].isin(keep_pubs)
].copy()

print("Original dev–pub edges:", len(edges_df))
print("Filtered dev–pub edges:", len(filtered_edges))

dev_nodes = pd.DataFrame(
    {"Node": filtered_edges["Developer"].unique(), "Type": "Developer"}
)
pub_nodes = pd.DataFrame(
    {"Node": filtered_edges["Publisher"].unique(), "Type": "Publisher"}
)

nodes_filtered = pd.concat([dev_nodes, pub_nodes], ignore_index=True)
nodes_filtered.drop_duplicates(subset=["Node"], inplace=True)

print("Filtered developer nodes:", len(dev_nodes))
print("Filtered publisher nodes:", len(pub_nodes))

# Save dev–pub network (wrap in try/except in case files are open)
try:
    filtered_edges.to_csv("edges_dev_pub_filtered.csv", index=False)
    nodes_filtered.to_csv("nodes_dev_pub_filtered.csv", index=False)
    print("\nSaved dev–publisher network files:")
    print("  edges_dev_pub_filtered.csv")
    print("  nodes_dev_pub_filtered.csv")
except PermissionError:
    print(
        "\n[WARNING] Could not overwrite dev–pub CSV files. "
        "Close them in Excel/Cytoscape and rerun this part if needed."
    )

# -------------------------------
# 8. Save summary tables for report
# -------------------------------

genre_stats.to_csv("genre_stats_summary.csv")
free_paid_stats.to_csv("free_paid_stats_summary.csv")
publisher_stats.to_csv("publisher_stats_summary.csv")

print("\nAnalysis complete. Summary tables and network files saved.")

# Tik pirmas grafikas buvo panaudotas egzamine, kiti klausimai buvo atmesti
# dėl per mažo duomenų kiekio.
