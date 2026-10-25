import ast, html, random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineScope", page_icon="🎬", layout="wide")
RED, GOLD = "#E50914", "#F5C518"

# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #2a0a12 0%, transparent 60%),
         radial-gradient(900px 500px at 100% 0%, #10143a 0%, transparent 55%), #0B0B12; }
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes fadeUp { from { opacity:0; transform: translateY(24px);} to { opacity:1; transform:none;} }
@keyframes float { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-8px);} }
@keyframes grow { from { width:0; } }
@keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(229,9,20,.5);} 50% { box-shadow:0 0 0 12px rgba(229,9,20,0);} }
@property --n { syntax:'<integer>'; initial-value:0; inherits:false; }
@keyframes count { from { --n:0; } to { --n:var(--to); } }
.hero { text-align:center; padding: 1.2rem 0 .4rem; animation: fadeUp .9s ease both; }
.hero .reel { font-size:3rem; display:inline-block; animation: float 3s ease-in-out infinite; }
.hero h1 { font-weight:800; font-size:3.4rem; margin:0; letter-spacing:2px;
  background: linear-gradient(90deg, #E50914, #F5C518, #ff6b6b, #E50914); background-size:200% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation: shimmer 5s linear infinite; }
.hero p { color:#9aa0b4; font-weight:300; margin-top:.2rem; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1rem; margin:1rem 0 1.4rem; }
.kpi { background:linear-gradient(145deg,#171726,#10101a); border:1px solid #26263a; border-radius:18px;
  padding:1.1rem 1.2rem; animation: fadeUp .8s ease both; transition: transform .25s, border-color .25s; }
.kpi:hover { transform: translateY(-6px); border-color:#E50914; }
.kpi-v { font-size:2.1rem; font-weight:800; color:#F5C518; }
.kpi-l { color:#9aa0b4; font-size:.85rem; letter-spacing:1px; text-transform:uppercase; }
.count { animation: count 1.8s ease-out forwards; counter-reset: num var(--n); }
.count::after { content: counter(num); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:1rem; }
.card { background:linear-gradient(160deg,#181828,#0f0f19); border:1px solid #26263a; border-radius:16px;
  padding:1rem 1.1rem; animation: fadeUp .6s ease both; transition: all .3s; position:relative; overflow:hidden; }
.card:hover { transform: translateY(-8px) scale(1.02); border-color:#E50914; box-shadow:0 14px 30px rgba(229,9,20,.25); }
.card-t { font-weight:600; font-size:1.05rem; }
.yr { color:#7b8099; font-weight:300; font-size:.85rem; }
.rate { color:#F5C518; font-weight:600; margin:.2rem 0 .4rem; }
.chip { display:inline-block; background:#22223a; color:#c9cdea; border-radius:20px; padding:2px 10px;
  font-size:.7rem; margin:0 4px 4px 0; }
.card p { color:#aab; font-size:.8rem; font-weight:300; margin:.4rem 0; }
.bar { background:#22223a; border-radius:10px; height:6px; overflow:hidden; }
.bar div { height:100%; background:linear-gradient(90deg,#E50914,#F5C518); border-radius:10px; animation: grow 1.2s ease both; }
.stButton>button { background:linear-gradient(90deg,#E50914,#b20710); color:white; border:none; border-radius:30px;
  padding:.5rem 1.4rem; font-weight:600; transition: transform .2s; animation: pulse 2.4s infinite; }
.stButton>button:hover { transform: scale(1.06); color:white; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#14141F; border-radius:30px; padding:.4rem 1.1rem; transition: background .2s; }
.stTabs [data-baseweb="tab"]:hover { background:#241a24; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,#E50914,#b20710) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
section[data-testid="stSidebar"] { background:#0f0f19; border-right:1px solid #26263a; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- DATA -----------------------------
def names(t):
    try:
        return [d["name"] for d in ast.literal_eval(t)]
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
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"
    for c in ["genres", "keywords"]:
        df[c] = df[c].apply(names)
    df = df[df["status"] == "Released"].reset_index(drop=True)
    df[["budget", "revenue"]] = df[["budget", "revenue"]].replace(0, np.nan)
    return df


@st.cache_resource
def build_index(_df):
    tok = lambda items: " ".join(i.replace(" ", "") for i in items)
    tags = _df["overview"] + " " + _df["tagline"] + " " + _df["genres"].apply(tok) + " " + _df["keywords"].apply(tok)
    vec = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
    return vec, vec.fit_transform(tags)


def add_finance(d):
    f = d.dropna(subset=["budget", "revenue"]).copy()
    f = f[f["budget"] >= 10_000]
    f["profit"] = f["revenue"] - f["budget"]
    f["roi"] = np.round((f["revenue"].to_numpy() - f["budget"].to_numpy()) / f["budget"].to_numpy() * 100, 2)
    f["category"] = np.select([f["roi"] < 0, f["roi"].between(0, 100), f["roi"] > 100],
                              ["Loss", "Moderate", "Blockbuster"], default="Unknown")
    return f


movies = load()
vec, matrix = build_index(movies)
all_genres = sorted({g for gs in movies["genres"] for g in gs})
titles = sorted(movies["title"].unique())


def style(fig, h=430):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=55, b=10), font=dict(family="Poppins"),
                      title_font_size=18)
    return fig


def show(fig, h=430):
    st.plotly_chart(style(fig, h), width="stretch")


def kpi(label, value, prefix="", suffix="", delay=0.0):
    return (f'<div class="kpi" style="animation-delay:{delay}s"><div class="kpi-v">{prefix}'
            f'<span class="count" style="--to:{int(value)}"></span>{suffix}</div>'
            f'<div class="kpi-l">{label}</div></div>')


def card(i, r, score=None):
    chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in r.genres)
    ov = r.overview if len(r.overview) < 170 else r.overview[:170] + "…"
    sc = (f'<div class="bar"><div style="width:{score * 100:.0f}%"></div></div>'
          f'<small style="color:#7b8099">{score * 100:.0f}% match</small>') if score is not None else ""
    return (f'<div class="card" style="animation-delay:{i * 0.08}s"><div class="card-t">{html.escape(r.title)} '
            f'<span class="yr">{r.year}</span></div><div class="rate">★ {r.vote_average:.1f}</div>{chips}'
            f'<p>{html.escape(ov)}</p>{sc}</div>')


def cards(df, scores=None):
    out = "".join(card(i, r, None if scores is None else scores[i]) for i, r in enumerate(df.itertuples()))
    st.markdown(f'<div class="grid">{out}</div>', unsafe_allow_html=True)


# ----------------------------- SIDEBAR FILTERS -----------------------------
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    yrs = st.slider("Release year", int(movies.year.min()), int(movies.year.max()), (1980, int(movies.year.max())))
    sel_g = st.multiselect("Genres", all_genres, placeholder="All genres")
    min_r = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    min_v = st.slider("Minimum votes", 0, 5000, 50, 50)
    st.caption("Filters apply to Overview, Box Office, ROI, Explorer and Word Cloud.")

view = movies[movies.year.between(*yrs) & (movies.vote_average >= min_r) & (movies.vote_count >= min_v)]
if sel_g:
    view = view[view.genres.apply(lambda g: bool(set(g) & set(sel_g)))]
if view.empty:
    st.warning("No movies match these filters. Try widening them.")
    st.stop()
fin = add_finance(view)

# ----------------------------- HERO -----------------------------
st.markdown('<div class="hero"><span class="reel">🎬</span><h1>CINESCOPE</h1>'
            '<p>Explore box office, ROI and find your next movie · TMDB 5000</p></div>', unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "🍿 Recommender", "💰 Box Office", "📈 ROI", "⚔️ Compare", "🔎 Explorer", "☁️ Word Cloud"])

# ----------------------------- OVERVIEW -----------------------------
with tabs[0]:
    st.markdown('<div class="kpis">' + kpi("Movies", len(view)) + kpi("Total revenue", fin.revenue.sum() / 1e9, "$", "B", .1)
                + kpi("Median ROI", fin.roi.median() if len(fin) else 0, "", "%", .2)
                + f'<div class="kpi" style="animation-delay:.3s"><div class="kpi-v">★ {view.vote_average.mean():.2f}</div>'
                  '<div class="kpi-l">Avg rating</div></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        by_year = view.groupby("year").size().reset_index(name="movies")
        show(px.area(by_year, x="year", y="movies", title="Movies released per year", color_discrete_sequence=[RED]), 340)
    with c2:
        ry = fin.groupby("year")["revenue"].median().reset_index()
        show(px.line(ry, x="year", y="revenue", markers=True, title="Median revenue by year",
                     color_discrete_sequence=[GOLD]), 340)
    g = view.explode("genres").groupby("genres").agg(movies=("title", "count"), rating=("vote_average", "mean")).reset_index()
    show(px.treemap(g, path=[px.Constant("All genres"), "genres"], values="movies", color="rating",
                    color_continuous_scale="YlOrRd", title="Genre universe (size = movies, colour = avg rating)"), 450)

# ----------------------------- RECOMMENDER -----------------------------
with tabs[1]:
    if "pick" not in st.session_state:
        st.session_state.pick = "Avatar"

    def surprise():
        st.session_state.pick = random.choice(titles)

    mode = st.radio("Find by", ["Similar to a movie", "Describe what you want"], horizontal=True)
    n = st.slider("How many?", 3, 12, 6)
    if mode == "Similar to a movie":
        a, b = st.columns([4, 1])
        a.selectbox("Pick a movie", titles, key="pick")
        b.markdown("<br>", unsafe_allow_html=True)
        b.button("🎲 Surprise me", on_click=surprise)
        idx = movies.index[movies.title == st.session_state.pick][0]
        scores = cosine_similarity(matrix[idx], matrix).ravel()
        scores[idx] = -1
    else:
        q = st.text_input("Describe the plot, mood or theme", "space adventure with robots and a rebellion")
        scores = cosine_similarity(vec.transform([q]), matrix).ravel()
    top = np.argsort(scores)[::-1][:n]
    cards(movies.iloc[top], scores[top])

# ----------------------------- BOX OFFICE -----------------------------
with tabs[2]:
    if len(fin) < 5:
        st.info("Not enough movies with budget & revenue data for these filters.")
    else:
        st.markdown("##### ▶ Press play to watch budgets vs revenue evolve by decade")
        fd = fin.sort_values("decade")
        fig = px.scatter(fd, x="budget", y="revenue", animation_frame="decade", animation_group="title",
                         size="popularity", color="category", hover_name="title", log_x=True, log_y=True,
                         size_max=38, range_x=[fd.budget.min() * .8, fd.budget.max() * 1.3],
                         range_y=[fd.revenue.min() * .8, fd.revenue.max() * 1.3],
                         color_discrete_map={"Loss": "#6c7aff", "Moderate": GOLD, "Blockbuster": RED})
        show(fig, 520)
        c1, c2 = st.columns([1, 2])
        metric = c1.selectbox("Rank by", ["revenue", "profit", "roi", "vote_average", "popularity"])
        topn = c1.slider("Top N", 5, 25, 10)
        base = view[view.vote_count >= 500] if metric == "vote_average" else (view if metric == "popularity" else fin)
        t = base.nlargest(topn, metric).sort_values(metric)
        with c2:
            show(px.bar(t, x=metric, y="title", orientation="h", color=metric, color_continuous_scale="Reds",
                        title=f"Top {topn} by {metric}"), 460)
        d1, d2 = st.columns(2)
        with d1:
            corr = fin[["budget", "revenue", "profit", "roi", "vote_average", "popularity", "runtime"]].corr().round(2)
            show(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Correlation heatmap"), 440)
        with d2:
            pv = view.explode("genres").pivot_table(index="genres", columns="decade", values="vote_average", aggfunc="mean")
            show(px.imshow(pv, color_continuous_scale="YlOrRd", aspect="auto", title="Avg rating: genre × decade"), 440)

# ----------------------------- ROI -----------------------------
with tabs[3]:
    if len(fin) < 5:
        st.info("Not enough financial data for these filters.")
    else:
        roi = fin.roi.to_numpy()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean ROI", f"{np.mean(roi):,.0f}%")
        m2.metric("Median ROI", f"{np.median(roi):,.0f}%")
        m3.metric("90th percentile", f"{np.percentile(roi, 90):,.0f}%")
        m4.metric("Hit rate", f"{np.mean(roi > 0) * 100:.0f}%")
        c1, c2 = st.columns(2)
        with c1:
            show(px.histogram(fin[fin.roi.between(-100, 1000)], x="roi", nbins=50, title="ROI distribution (−100% to 1000%)",
                              color_discrete_sequence=[RED]), 380)
        with c2:
            cc = fin.category.value_counts().reset_index()
            show(px.pie(cc, names="category", values="count", hole=.55, title="Loss / Moderate / Blockbuster",
                        color="category", color_discrete_map={"Loss": "#6c7aff", "Moderate": GOLD, "Blockbuster": RED}), 380)
        ge = fin.explode("genres")
        gr = ge.groupby("genres")["roi"].median().sort_values(ascending=False).reset_index()
        show(px.bar(gr, x="genres", y="roi", color="roi", color_continuous_scale="Sunset", title="Median ROI by genre"), 380)
        l, r = st.columns(2)
        big = fin[fin.budget >= 1e6]
        l.markdown("**🏆 Best ROI (budget ≥ $1M)**")
        l.dataframe(big.nlargest(10, "roi")[["title", "budget", "revenue", "roi"]], hide_index=True, width="stretch")
        r.markdown("**💸 Biggest losses**")
        r.dataframe(fin.nsmallest(10, "profit")[["title", "budget", "revenue", "profit"]], hide_index=True, width="stretch")

        st.markdown("### 🔮 ROI Simulator")
        s1, s2 = st.columns(2)
        sg = s1.multiselect("Genre(s)", all_genres, default=["Action"])
        bud = s2.number_input("Budget ($)", 100_000, 400_000_000, 50_000_000, step=1_000_000)
        pool = ge[ge.genres.isin(sg)] if sg else ge
        if len(pool) >= 5:
            p25, p50, p75 = np.percentile(pool.roi.to_numpy(), [25, 50, 75])
            x1, x2, x3 = st.columns(3)
            x1.metric("Pessimistic (25th pct)", f"${bud * (1 + p25 / 100):,.0f}")
            x2.metric("Typical (median)", f"${bud * (1 + p50 / 100):,.0f}")
            x3.metric("Optimistic (75th pct)", f"${bud * (1 + p75 / 100):,.0f}")
            st.caption("Based on historical ROI of similar movies in the filtered data. For fun, not financial advice.")

# ----------------------------- COMPARE -----------------------------
with tabs[4]:
    a, b = st.columns(2)
    m_a = a.selectbox("Movie A", titles, index=titles.index("Avatar"))
    m_b = b.selectbox("Movie B", titles, index=titles.index("The Dark Knight"))
    ra, rb = (movies[movies.title == t].iloc[0] for t in (m_a, m_b))
    cols = ["vote_average", "popularity", "runtime", "budget", "revenue", "vote_count"]
    p99 = movies[cols].quantile(.99)
    fig = go.Figure()
    for r, col in ((ra, RED), (rb, GOLD)):
        vals = [min(float(np.nan_to_num(r[c])) / p99[c], 1) for c in cols]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cols + cols[:1], fill="toself", name=r.title,
                                      line_color=col, opacity=.75))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False)), title="Head-to-head (scaled)")
    c1, c2 = st.columns([3, 2])
    with c1:
        show(fig, 460)
    with c2:
        tbl = pd.DataFrame({m_a: [ra[c] for c in cols], m_b: [rb[c] for c in cols]}, index=cols)
        st.dataframe(tbl.round(1), width="stretch")
        sim = cosine_similarity(matrix[ra.name], matrix[rb.name])[0][0]
        st.metric("Content similarity", f"{sim * 100:.0f}%")

# ----------------------------- EXPLORER -----------------------------
with tabs[5]:
    q = st.text_input("🔎 Search title or keyword", "")
    ex = view
    if q:
        ex = view[view.title.str.contains(q, case=False, na=False) | view.keywords.apply(lambda k: any(q.lower() in x.lower() for x in k))]
    sort = st.selectbox("Sort by", ["popularity", "vote_average", "revenue", "budget", "year"])
    show_df = ex.sort_values(sort, ascending=False)[["title", "year", "genres", "vote_average", "vote_count", "budget", "revenue", "runtime"]].copy()
    show_df["genres"] = show_df["genres"].apply(", ".join)
    st.caption(f"{len(show_df):,} movies")
    st.dataframe(show_df, hide_index=True, width="stretch", height=520, column_config={
        "vote_average": st.column_config.ProgressColumn("Rating", min_value=0, max_value=10, format="%.1f"),
        "budget": st.column_config.NumberColumn("Budget", format="$%d"),
        "revenue": st.column_config.NumberColumn("Revenue", format="$%d")})
    st.download_button("⬇️ Download filtered data (CSV)", show_df.to_csv(index=False).encode(), "movies_filtered.csv", "text/csv")

# ----------------------------- WORD CLOUD -----------------------------
with tabs[6]:
    src = st.radio("Build from", ["Overviews", "Keywords"], horizontal=True)
    text = " ".join(view.overview) if src == "Overviews" else " ".join(k.replace(" ", "_") for ks in view.keywords for k in ks)
    if text.strip():
        wc = WordCloud(width=1200, height=550, mode="RGBA", background_color=None, stopwords=STOPWORDS,
                       colormap="YlOrRd", max_words=160).generate(text)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        fig.patch.set_alpha(0)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
