import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
import traceback
from azure.storage.blob import BlobServiceClient
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ── THEME ────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.labelcolor":  "#cbd5e1",
    "axes.edgecolor":   "#334155",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#f1f5f9",
    "grid.color":       "#334155",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})
ACCENT  = "#38bdf8"
DANGER  = "#f43f5e"
SUCCESS = "#34d399"
WARNING = "#fb923c"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rail·IA — Analyse Ferroviaire",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0f172a; color: #e2e8f0; }
h1 { font-family: 'Space Mono', monospace; letter-spacing: -1px; color: #38bdf8; }
h2, h3 { font-family: 'Space Mono', monospace; color: #94a3b8; font-size: 0.95rem;
          text-transform: uppercase; letter-spacing: 2px; }
.kpi-card { background: #1e293b; border: 1px solid #334155;
            border-top: 3px solid #38bdf8; border-radius: 12px;
            padding: 1.2rem 1.5rem; margin-bottom: 0.5rem; }
.kpi-value { font-family: 'Space Mono', monospace; font-size: 2rem;
             font-weight: 700; color: #f1f5f9; }
.kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 1px; }
.chat-bubble-user { background: #1e40af; border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0 0.5rem 20%;
    color: #e0f2fe; font-size: 0.92rem; }
.chat-bubble-ai { background: #1e293b; border: 1px solid #334155;
    border-radius: 16px 16px 16px 4px; padding: 0.8rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0; color: #e2e8f0; font-size: 0.92rem; }
div[data-testid="stButton"] button {
    background: #1e293b; border: 1px solid #334155; color: #cbd5e1;
    border-radius: 8px; font-family: 'Space Mono', monospace;
    font-size: 0.78rem; transition: all 0.2s; }
div[data-testid="stButton"] button:hover {
    background: #334155; border-color: #38bdf8; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🚆 Rail·IA")
st.markdown("**Assistant IA d'analyse ferroviaire** — posez n'importe quelle question sur vos données")
st.divider()

# ── SECRETS ───────────────────────────────────────────────────────────────────
try:
    AZURE_CONN  = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
    MISTRAL_KEY = st.secrets["MISTRAL_API_KEY"]
except KeyError:
    st.error("⚠️ Clés manquantes dans les Secrets Streamlit.")
    st.stop()

# ── CHARGEMENT DONNÉES ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Connexion Azure & chargement des données…")
def load_data():
    try:
        svc  = BlobServiceClient.from_connection_string(AZURE_CONN)
        blob = svc.get_blob_client(container="ztacontainer", blob="otp_transformed.xlsx")
        raw  = blob.download_blob().readall()
        return pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        st.error(f"Erreur Azure : {e}")
        return None

df_full = load_data()
if df_full is None:
    st.stop()

df = df_full.sample(n=min(10_000, len(df_full)), random_state=42)
if len(df_full) > 10_000:
    st.info(f"📊 Échantillon de 10 000 lignes sur {len(df_full):,} au total.")

# ── KPI HEADER ────────────────────────────────────────────────────────────────
total    = len(df)
retards  = int(df["is_delayed"].sum())
ponct    = (total - retards) / total * 100
moy_ret  = df[df["is_delayed"] == 1]["delay_minutes"].mean()
nb_gares = df["station"].nunique() if "station" in df.columns else "N/A"

k1, k2, k3, k4 = st.columns(4)
for col, val, label, color in [
    (k1, f"{ponct:.1f}%",       "Ponctualité globale",  SUCCESS),
    (k2, f"{retards:,}",        "Trains en retard",     DANGER),
    (k3, f"{moy_ret:.1f} min",  "Retard moyen",         WARNING),
    (k4, str(nb_gares),         "Gares analysées",      ACCENT),
]:
    col.markdown(f"""
    <div class="kpi-card" style="border-top-color:{color}">
        <div class="kpi-value" style="color:{color}">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_fig"     not in st.session_state: st.session_state.last_fig     = None
if "all_figures"  not in st.session_state: st.session_state.all_figures  = []

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str,
               color=ACCENT, top_n: int = 10) -> plt.Figure:
    data = series.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(3, len(data) * 0.55)))
    bars = ax.barh(data.index.astype(str)[::-1], data.values[::-1],
                   color=color, edgecolor="none", height=0.65)
    ax.bar_label(bars, fmt="%.1f", padding=4, color="#cbd5e1", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#f1f5f9", pad=12)
    ax.set_xlabel(ylabel, fontsize=8)
    ax.set_ylabel(xlabel, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig

def _save_fig(title: str, fig: plt.Figure):
    st.session_state.last_fig = fig
    st.session_state.all_figures.append((title, fig))

# ── OUTILS ───────────────────────────────────────────────────────────────────

@tool
def analyser_gares_ponctualite(top_ou_flop: str = "top", n: int = 10) -> str:
    """
    Analyse la ponctualité par gare.
    top_ou_flop : 'top' pour les meilleures gares, 'flop' pour les pires.
    n : nombre de gares à retourner (défaut 10).
    """
    if "station" not in df.columns:
        return "Erreur : colonne 'station' introuvable dans les données."
    grp = df.groupby("station").agg(total=("is_delayed","count"), retards=("is_delayed","sum"))
    grp["ponctualite"] = (grp["total"] - grp["retards"]) / grp["total"] * 100
    grp = grp[grp["total"] >= 10]

    if top_ou_flop.lower() == "flop":
        ranked = grp.nsmallest(n, "ponctualite")
        color, titre = DANGER,  f"🔴 {n} gares les moins ponctuelles"
    else:
        ranked = grp.nlargest(n, "ponctualite")
        color, titre = SUCCESS, f"🟢 {n} gares les plus ponctuelles"

    fig = _bar_chart(ranked["ponctualite"], title=titre,
                     xlabel="Gare", ylabel="Ponctualité (%)", color=color, top_n=n)
    _save_fig(titre, fig)

    lines = [titre, ""]
    for i, (gare, row) in enumerate(ranked.iterrows(), 1):
        lines.append(f"{i:2}. {gare:<30} {row['ponctualite']:5.1f}%  ({int(row['total'])} passages)")
    return "\n".join(lines)


@tool
def analyser_retards_par_gare(top_n: int = 10) -> str:
    """
    Gares avec le plus grand retard cumulé en minutes.
    top_n : nombre de gares (défaut 10).
    """
    if "station" not in df.columns:
        return "Erreur : colonne 'station' introuvable."
    grp = df[df["is_delayed"] == 1].groupby("station")["delay_minutes"].agg(["sum","mean","count"])
    grp.columns = ["retard_total_min","retard_moyen_min","nb_retards"]
    ranked = grp.nlargest(top_n, "retard_total_min")

    fig = _bar_chart(ranked["retard_total_min"],
                     title=f"⏱ Top {top_n} gares — retard cumulé (min)",
                     xlabel="Gare", ylabel="Retard cumulé (min)", color=WARNING, top_n=top_n)
    _save_fig(f"Top {top_n} gares retard cumulé", fig)

    lines = [f"⏱ Top {top_n} gares par retard cumulé", ""]
    for i, (gare, row) in enumerate(ranked.iterrows(), 1):
        lines.append(
            f"{i:2}. {gare:<30} "
            f"Total={int(row['retard_total_min'])} min  "
            f"Moy={row['retard_moyen_min']:.1f} min  "
            f"({int(row['nb_retards'])} retards)"
        )
    return "\n".join(lines)


@tool
def analyser_heures_critiques() -> str:
    """Identifie les heures de la journée avec le plus de retards et génère un graphique."""
    grp = df.groupby("hour").agg(total=("is_delayed","count"), retards=("is_delayed","sum"))
    grp["taux_retard"] = grp["retards"] / grp["total"] * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grp.index, grp["retards"], color=DANGER, alpha=0.7, label="Nb retards")
    ax2 = ax.twinx()
    ax2.plot(grp.index, grp["taux_retard"], color=WARNING, linewidth=2.5,
             marker="o", markersize=5, label="Taux retard (%)")
    ax.set_title("Retards par heure de la journée", fontsize=11,
                 fontweight="bold", color="#f1f5f9", pad=12)
    ax.set_xlabel("Heure", fontsize=8)
    ax.set_ylabel("Nombre de retards", fontsize=8, color=DANGER)
    ax2.set_ylabel("Taux de retard (%)", fontsize=8, color=WARNING)
    ax.set_xticks(grp.index)
    fig.tight_layout()
    _save_fig("Retards par heure", fig)

    top3 = grp.nlargest(3, "retards")
    lines = ["📊 Analyse des retards par heure", "", "Top 3 heures critiques :"]
    for h, row in top3.iterrows():
        lines.append(f"  - {int(h):02d}h : {int(row['retards'])} retards ({row['taux_retard']:.1f}%)")
    return "\n".join(lines)


@tool
def statistiques_globales() -> str:
    """Fournit les statistiques générales complètes sur le jeu de données."""
    total   = len(df)
    retards = int(df["is_delayed"].sum())
    ponct   = (total - retards) / total * 100
    moy_ret = df[df["is_delayed"] == 1]["delay_minutes"].mean()
    max_ret = df["delay_minutes"].max()
    nb_g    = df["station"].nunique() if "station" in df.columns else "N/A"
    return (
        f"📈 Statistiques globales\n"
        f"  Total trains      : {total:,}\n"
        f"  Trains en retard  : {retards:,} ({100*retards/total:.1f}%)\n"
        f"  Ponctualité       : {ponct:.2f}%\n"
        f"  Retard moyen      : {moy_ret:.1f} min\n"
        f"  Retard maximum    : {max_ret:.0f} min\n"
        f"  Gares analysées   : {nb_g}"
    )


@tool
def ponctualite_direction(direction: str) -> str:
    """Calcule la ponctualité pour une direction (N ou S)."""
    direction = direction.strip().upper()
    if direction not in ["N", "S"]:
        return "Erreur : direction doit être 'N' ou 'S'."
    d = df[df["direction"].str.upper() == direction]
    if d.empty:
        return f"Aucune donnée pour la direction {direction}."
    total = len(d)
    ok    = len(d[d["is_delayed"] == 0])
    return (f"Direction {direction} : {total:,} trains — "
            f"{ok:,} à l'heure — ponctualité = {ok/total*100:.2f}%")


@tool
def trains_les_plus_en_retard(n: int = 5) -> str:
    """Liste les N trains avec les retards les plus longs (défaut 5)."""
    top = df.nlargest(n, "delay_minutes")
    lines = [f"🚂 Top {n} trains avec les plus longs retards", ""]
    for _, r in top.iterrows():
        gare = f"@ {r['station']}" if "station" in df.columns else ""
        lines.append(f"  - Train {r['train_id']} {gare} : {r['delay_minutes']:.0f} min")
    return "\n".join(lines)


@tool
def distribution_retards_gare(gare: str) -> str:
    """
    Analyse détaillée des retards pour une gare spécifique avec histogramme.
    gare : nom exact ou partiel de la gare.
    """
    if "station" not in df.columns:
        return "Erreur : colonne 'station' introuvable."
    sub = df[df["station"].str.contains(gare, case=False, na=False)]
    if sub.empty:
        return f"Aucune donnée pour la gare '{gare}'."
    retards = sub[sub["is_delayed"] == 1]["delay_minutes"]
    if retards.empty:
        return f"Aucun retard enregistré pour '{gare}'."

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(retards, bins=30, color=ACCENT, edgecolor="#0f172a", alpha=0.9)
    ax.set_title(f"Distribution des retards — {gare}", fontsize=11,
                 fontweight="bold", color="#f1f5f9", pad=12)
    ax.set_xlabel("Retard (min)", fontsize=8)
    ax.set_ylabel("Nombre de trains", fontsize=8)
    fig.tight_layout()
    _save_fig(f"Distribution {gare}", fig)

    return (
        f"📍 Gare : {gare} ({len(sub):,} passages)\n"
        f"  Ponctualité   : {(1 - sub['is_delayed'].mean())*100:.1f}%\n"
        f"  Retard moyen  : {retards.mean():.1f} min\n"
        f"  Retard médian : {retards.median():.1f} min\n"
        f"  Retard max    : {retards.max():.0f} min\n"
        f"  Nb retards    : {len(retards):,}"
    )


TOOLS = [
    analyser_gares_ponctualite,
    analyser_retards_par_gare,
    analyser_heures_critiques,
    statistiques_globales,
    ponctualite_direction,
    trains_les_plus_en_retard,
    distribution_retards_gare,
]

# ── AGENT ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=MISTRAL_KEY,
        temperature=0,
    )
    system = (
        "Tu es un expert en analyse de données ferroviaires. "
        "Utilise toujours les outils disponibles pour répondre aux questions. "
        "Présente les résultats de façon claire et structurée. "
        "Réponds toujours en français."
    )
    return create_react_agent(model=llm, tools=TOOLS, prompt=system)

agent = build_agent()

def run_agent(query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_chat, tab_viz = st.tabs(["💬 Chat IA", "📈 Visualisations"])

# ════════ TAB 1 — CHAT ════════
with tab_chat:

    for role, msg in st.session_state.chat_history:
        css = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
        ico = "🧑" if role == "user" else "🤖"
        st.markdown(f'<div class="{css}">{ico} {msg}</div>', unsafe_allow_html=True)

    if st.session_state.last_fig is not None:
        st.pyplot(st.session_state.last_fig, use_container_width=True)

    st.write("")

    with st.form("chat_form", clear_on_submit=True):
        query     = st.text_input("Votre question :",
                                  placeholder="Ex : quelles sont les 10 gares les plus en retard ?")
        submitted = st.form_submit_button("Envoyer ➤")

    if submitted and query.strip():
        st.session_state.chat_history.append(("user", query))
        st.session_state.last_fig = None
        with st.spinner("🤔 Analyse en cours…"):
            try:
                response = run_agent(query)
                st.session_state.chat_history.append(("ai", response))
            except Exception as e:
                err = str(e)
                if "429" in err or "rate limit" in err.lower():
                    msg = "⏳ Limite API Mistral atteinte — réessayez dans quelques minutes."
                else:
                    msg = f"❌ Erreur : {err}"
                st.session_state.chat_history.append(("ai", msg))
        st.rerun()

    st.write("---")
    st.markdown("##### ⚡ Questions rapides")
    quick = [
        ("🟢 Top 10 gares",          "Quelles sont les 10 gares les plus ponctuelles ?"),
        ("🔴 Flop 10 gares",         "Quelles sont les 10 gares les moins ponctuelles ?"),
        ("⏱ Retard cumulé",          "Quelles gares ont le plus grand retard cumulé ?"),
        ("🕐 Heures critiques",       "Quelles sont les heures les plus critiques ?"),
        ("📊 Stats globales",         "Donne-moi les statistiques globales."),
        ("🚂 Top retards trains",     "Quels sont les 5 trains les plus en retard ?"),
        ("📍 Direction N",            "Quelle est la ponctualité de la direction N ?"),
        ("📍 Direction S",            "Quelle est la ponctualité de la direction S ?"),
    ]
    cols = st.columns(4)
    for i, (label, q) in enumerate(quick):
        with cols[i % 4]:
            if st.button(label, use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                st.session_state.last_fig = None
                with st.spinner("Analyse…"):
                    try:
                        r = run_agent(q)
                        st.session_state.chat_history.append(("ai", r))
                    except Exception as e:
                        st.session_state.chat_history.append(("ai", f"❌ {e}"))
                st.rerun()

# ════════ TAB 2 — VISUALISATIONS ════════
with tab_viz:
    st.markdown("### 📈 Graphiques générés par l'IA")

    if not st.session_state.all_figures:
        st.info("Aucun graphique pour l'instant. Posez une question dans le chat pour en générer.")
    else:
        if st.button("🗑 Effacer tous les graphiques"):
            st.session_state.all_figures = []
            st.session_state.last_fig    = None
            st.rerun()

        for title, fig in reversed(st.session_state.all_figures):
            st.markdown(f"**{title}**")
            st.pyplot(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            st.download_button(
                label="⬇ Télécharger",
                data=buf.getvalue(),
                file_name=f"{title.replace(' ', '_')}.png",
                mime="image/png",
                key=f"dl_{title}_{id(fig)}",
            )
            st.divider()
