# ============================================
# Steam TAG co-occurrence network for Cytoscape
# Atsakant į:
# "Kokie Steam žymekliai (tags) dažniausiai pasirodo kartu
#  ir kaip jie formuoja teminių grupių tinklą?"
# ============================================

import pandas as pd
from collections import Counter
from itertools import combinations

# ------------------------------------------------
# 1. Load data
# ------------------------------------------------

df = pd.read_csv("games.csv", encoding="utf-8", low_memory=False)

print("Original shape:", df.shape)
print("Columns:", list(df.columns))

# use reviews to restrict to games with enough data
df["TotalReviews"] = df["Positive"] + df["Negative"]
df = df[df["TotalReviews"] > 0].copy()

print("\nAfter removing games with 0 reviews:", df.shape)

# ------------------------------------------------
# 2. Filter to *reliable* games (enough reviews)
# ------------------------------------------------
MIN_REVIEWS = 1000      # only games with >= 1000 total reviews

df_good = df[df["TotalReviews"] >= MIN_REVIEWS].copy()
print(f"\nGames with >= {MIN_REVIEWS} reviews:", df_good.shape[0])

# ------------------------------------------------
# 3. Parse Tags column into lists
# ------------------------------------------------

def parse_tags(tag_string):
    """
    Turn a raw 'Tags' string into a cleaned list of tags.
    - Handles delimiters ',', ';', '|'
    - Strips spaces
    - Removes empty entries
    - Deduplicates tags per game
    """
    if pd.isna(tag_string):
        return []
    s = str(tag_string)
    # Normalize delimiters
    s = s.replace(";", ",").replace("|", ",")
    parts = [t.strip() for t in s.split(",")]
    # Deduplicate per game
    unique = sorted(set([p for p in parts if p]))
    return unique

df_good["TagList"] = df_good["Tags"].apply(parse_tags)

# Remove games with fewer than 2 tags (no co-occurrence possible)
df_good = df_good[df_good["TagList"].apply(len) >= 2].copy()
print("After requiring at least 2 tags per game:", df_good.shape[0])

# Quick peek
print("\nExample TagList values:")
print(df_good["TagList"].head(5).tolist())

# ------------------------------------------------
# 4. Count tag frequencies and select top tags
# ------------------------------------------------
tag_counter = Counter()

for tags in df_good["TagList"]:
    tag_counter.update(tags)

print("\nTotal distinct tags in filtered games:", len(tag_counter))

# Keep only the most common tags to get a clear, readable network
TOP_N_TAGS = 40       # 30 / 50
top_tags = [t for t, c in tag_counter.most_common(TOP_N_TAGS)]
top_tag_set = set(top_tags)

print(f"\nTop {TOP_N_TAGS} tags:")
for t, c in tag_counter.most_common(TOP_N_TAGS):
    print(f"  {t}: {c} games")

# ------------------------------------------------
# 5. Build tag co-occurrence counts
# ------------------------------------------------
pair_counter = Counter()

for tags in df_good["TagList"]:
    # keep only top tags for this game
    filtered = [t for t in tags if t in top_tag_set]
    if len(filtered) < 2:
        continue
    # generate sorted pairs so ("Action","RPG") == ("RPG","Action")
    for a, b in combinations(sorted(filtered), 2):
        pair_counter[(a, b)] += 1

print("\nTotal tag pairs (including weak ones):", len(pair_counter))

# ------------------------------------------------
# 6. Convert to DataFrames and filter by strength
# ------------------------------------------------
edges = []
for (t1, t2), w in pair_counter.items():
    edges.append((t1, t2, w))

edges_df = pd.DataFrame(edges, columns=["Tag1", "Tag2", "Weight"])

# Keep only strong co-occurrences for a clean network
MIN_COOCC = 60   # "this pair occurs together in at least 20 games"
edges_df = edges_df[edges_df["Weight"] >= MIN_COOCC].copy()

print(f"Edges after Weight >= {MIN_COOCC} filter:", edges_df.shape[0])

# It’s possible some tags drop out because all their edges are weak
used_tags = sorted(set(edges_df["Tag1"]).union(set(edges_df["Tag2"])))

# Node table: one row per tag
nodes_df = pd.DataFrame({
    "Tag": used_tags,
})

nodes_df["CountGames"] = nodes_df["Tag"].map(tag_counter)
nodes_df["Type"] = "Tag"   # so Cytoscape knows these are tags

print("Nodes in final network:", nodes_df.shape[0])

# ------------------------------------------------
# 7. Save CSVs for Cytoscape
# ------------------------------------------------
edges_df.to_csv("edges_tag_cooc.csv", index=False)
nodes_df.to_csv("nodes_tag_cooc.csv", index=False)

print("\nSaved files for Cytoscape:")
print("  edges_tag_cooc.csv  (columns: Tag1, Tag2, Weight)")
print("  nodes_tag_cooc.csv  (columns: Tag, CountGames, Type)")

# ------------------------------------------------
# 8. Small summary
# ------------------------------------------------
print("\nTop 15 strongest tag pairs:")
print(
    edges_df.sort_values("Weight", ascending=False)
            .head(15)
            .to_string(index=False)
)
