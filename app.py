import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Analysis & Recommender", layout="wide")
sns.set_theme(style="whitegrid")


def names(text):
    try:
        return [d["name"] for d in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []


@st.cache_data
def load():
    df = pd.read_csv("tmdb_5000_movies.csv").drop_duplicates()
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df["year"] = df["release_date"].dt.year.astype(int)
    for c in ["genres", "keywords"]:
        df[c] = df[c].apply(names)
    df = df[df["status"] == "Released"].reset_index(drop=True)
    df[["budget", "revenue"]] = df[["budget", "revenue"]].replace(0, np.nan)
    return df


@st.cache_resource
def build_similarity(_df):
    df = _df
    tok = lambda items: " ".join(i.replace(" ", "") for i in items)
    tags = df["overview"] + " " + df["tagline"] + " " + df["genres"].apply(tok) + " " + df["keywords"].apply(tok)
    m = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2)).fit_transform(tags)
    return cosine_similarity(m)


movies = load()
sim = build_similarity(movies)

fin = movies.dropna(subset=["budget", "revenue"]).copy()
fin = fin[fin["budget"] >= 10_000]
fin["profit"] = fin["revenue"] - fin["budget"]
fin["roi"] = np.round((fin["revenue"].to_numpy() - fin["budget"].to_numpy()) / fin["budget"].to_numpy() * 100, 2)
fin["hit_flop"] = np.where(fin["roi"] > 0, "Hit", "Flop")
fin["category"] = np.select(
    [fin["roi"] < 0, fin["roi"].between(0, 100), fin["roi"] > 100],
    ["Loss", "Moderate", "Blockbuster"], default="Unknown")

st.title("🎬 Movie Analysis & Recommender")
st.caption("TMDB 5000 movies dataset")

tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Box office", "ROI", "Word cloud"])

with tab1:
    title = st.selectbox("Pick a movie", sorted(movies["title"].unique()),
                         index=int(np.where(np.sort(movies["title"].unique()) == "Avatar")[0][0]))
    n = st.slider("Number of recommendations", 3, 15, 5)
    idx = movies.index[movies["title"] == title][0]
    top = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:n + 1]
    out = movies.iloc[[i for i, _ in top]][["title", "year", "genres", "vote_average"]].copy()
    out["genres"] = out["genres"].apply(", ".join)
    out["similarity"] = [round(s, 3) for _, s in top]
    st.dataframe(out, hide_index=True, width="stretch")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=fin, x="budget", y="revenue", hue="hit_flop", alpha=0.6, ax=ax)
        ax.set_title("Budget vs Revenue")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=fin.nlargest(10, "profit"), x="profit", y="title", hue="title",
                    palette="viridis", legend=False, ax=ax)
        ax.set_title("Top 10 by Profit")
        st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(fin[["budget", "revenue", "profit", "roi", "vote_average", "popularity", "runtime"]].corr(),
                annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

with tab3:
    g = fin.explode("genres")
    roi = fin["roi"].to_numpy()
    a, b, c = st.columns(3)
    a.metric("Median ROI", f"{np.median(roi):.0f}%")
    b.metric("90th percentile", f"{np.percentile(roi, 90):.0f}%")
    c.metric("Movies analysed", len(fin))
    st.bar_chart(fin["category"].value_counts())
    fig, ax = plt.subplots(figsize=(10, 4))
    g.groupby("genres")["roi"].median().sort_values(ascending=False).plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Median ROI by Genre")
    ax.set_ylabel("ROI (%)")
    st.pyplot(fig)

with tab4:
    wc = WordCloud(width=1000, height=500, background_color="white",
                   stopwords=STOPWORDS, max_words=150).generate(" ".join(movies["overview"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
