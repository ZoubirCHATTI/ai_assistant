# -*- coding: utf-8 -*-
"""
Module de génération de visualisations pour l'analyse TER.

Organisation :
- Fonctions `_agg_*` / `_prepare_*` : calcul pur (pandas), sans dépendance Streamlit,
  facilement testables unitairement.
- Fonctions `plot_*` : rendu Streamlit + Plotly.
- `CHART_BUILDERS` : dispatch table remplaçant le gros if/elif de la version précédente.
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import Config

# ---------------------------------------------------------------------------
# Constantes / config centralisée
# ---------------------------------------------------------------------------

OBJECTIF_REGULARITE = getattr(Config, "OBJECTIF_REGULARITE", 90)
COLOR_SCALE = getattr(Config, "COLOR_SCALE", "RdYlGn")
CHART_HEIGHT = getattr(Config, "CHART_HEIGHT", 500)

# Ordre chronologique explicite pour éviter un tri alphabétique des mois
ORDRE_MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


# ---------------------------------------------------------------------------
# Helpers internes (purs, testables, mis en cache quand ils sont coûteux)
# ---------------------------------------------------------------------------

def _apply_common_layout(fig: go.Figure) -> go.Figure:
    """Applique une mise en forme cohérente à toutes les figures."""
    fig.update_layout(
        height=CHART_HEIGHT,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def _sort_months(index: pd.Index) -> list:
    """Trie une liste de mois selon l'ordre chronologique (pas alphabétique)."""
    index_lower = [str(m).lower() for m in index]
    known = [m for m in ORDRE_MOIS if m in index_lower]
    unknown = [m for m in index_lower if m not in ORDRE_MOIS]
    return known + sorted(unknown)


@st.cache_data(show_spinner=False)
def _monthly_mean(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Moyenne mensuelle d'une colonne, mise en cache."""
    return (
        df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col]
        .mean()
        .reset_index()
    )


@st.cache_data(show_spinner=False)
def _group_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return df.groupby(group_col)[value_col].mean().sort_values(ascending=True)


@st.cache_data(show_spinner=False)
def _pivot_mean(df: pd.DataFrame, index_col: str, columns_col: str, value_col: str) -> pd.DataFrame:
    return df.pivot_table(values=value_col, index=index_col, columns=columns_col, aggfunc="mean")


def _aggregate_for_share(
    df: pd.DataFrame, x_col: str, y_col: Optional[str]
) -> tuple[pd.DataFrame, str, str]:
    """
    Agrégation commune utilisée par pie / donut / treemap / funnel :
    - si y_col fourni : somme de y_col groupée par x_col
    - sinon : comptage des occurrences de x_col
    Retourne (dataframe, colonne_labels, colonne_valeurs).
    """
    if y_col:
        agg = df.groupby(x_col)[y_col].sum().reset_index()
        agg.columns = [x_col, "values"]
        return agg, x_col, "values"

    agg = df[x_col].value_counts().reset_index()
    agg.columns = [x_col, "count"]
    return agg, x_col, "count"


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------

def plot_kpi_cards(df: pd.DataFrame) -> None:
    """Affiche des cartes KPI en haut du dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    if "taux_regularite" in df.columns:
        avg_regularite = df["taux_regularite"].mean()
        delta_regularite = avg_regularite - OBJECTIF_REGULARITE
        with col1:
            st.metric(
                label="📊 Régularité Moyenne",
                value=f"{avg_regularite:.1f}%",
                delta=f"{delta_regularite:+.1f}% vs objectif",
                delta_color="normal" if avg_regularite >= OBJECTIF_REGULARITE else "inverse",
            )
    else:
        with col1:
            st.info("Colonne `taux_regularite` absente.")

    total_trains = 0
    if "nombre_trains_prevus" in df.columns:
        total_trains = df["nombre_trains_prevus"].sum()
        with col2:
            st.metric(label="🚂 Trains Prévus", value=f"{total_trains:,.0f}")
    else:
        with col2:
            st.info("Colonne `nombre_trains_prevus` absente.")

    if "nombre_trains_supprimes" in df.columns:
        total_supprimes = df["nombre_trains_supprimes"].sum()
        taux_suppression = (total_supprimes / total_trains * 100) if total_trains > 0 else 0
        with col3:
            st.metric(
                label="❌ Trains Supprimés",
                value=f"{total_supprimes:,.0f}",
                delta=f"{taux_suppression:.2f}%",
            )
    else:
        with col3:
            st.info("Colonne `nombre_trains_supprimes` absente.")

    if "region" in df.columns:
        with col4:
            st.metric(label="🗺️ Régions Analysées", value=f"{df['region'].nunique()}")
    else:
        with col4:
            st.info("Colonne `region` absente.")


# ---------------------------------------------------------------------------
# Graphiques métier prédéfinis
# ---------------------------------------------------------------------------

def plot_regularite_evolution(df: pd.DataFrame) -> None:
    """Graphique d'évolution de la régularité dans le temps."""
    if "date" not in df.columns or "taux_regularite" not in df.columns:
        st.warning("⚠️ Colonnes `date` et/ou `taux_regularite` manquantes.")
        return

    st.subheader("📈 Évolution de la Régularité")

    df_monthly = _monthly_mean(df, "date", "taux_regularite")
    if df_monthly.empty:
        st.info("Pas de données à afficher pour cette période.")
        return

    fig = px.line(
        df_monthly,
        x="date",
        y="taux_regularite",
        title="Taux de Régularité Mensuel",
        labels={"date": "Date", "taux_regularite": "Taux de Régularité (%)"},
        markers=True,
    )
    fig.add_hline(
        y=OBJECTIF_REGULARITE,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Objectif {OBJECTIF_REGULARITE}%",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(_apply_common_layout(fig), use_container_width=True)


def plot_regularite_by_region(df: pd.DataFrame) -> None:
    """Graphique de régularité par région."""
    if "region" not in df.columns or "taux_regularite" not in df.columns:
        st.warning("⚠️ Colonnes `region` et/ou `taux_regularite` manquantes.")
        return

    st.subheader("🗺️ Régularité par Région")

    df_region = _group_mean(df, "region", "taux_regularite")
    if df_region.empty:
        st.info("Pas de données à afficher.")
        return

    fig = px.bar(
        x=df_region.values,
        y=df_region.index,
        orientation="h",
        title="Taux de Régularité Moyen par Région",
        labels={"x": "Taux de Régularité (%)", "y": "Région"},
        color=df_region.values,
        color_continuous_scale=COLOR_SCALE,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(_apply_common_layout(fig), use_container_width=True)


def plot_causes_retards(df: pd.DataFrame) -> None:
    """Distribution des causes de retards (si disponible)."""
    st.subheader("🔍 Analyse des Perturbations")
    col1, col2 = st.columns(2)

    with col1:
        if "nombre_trains_retard" in df.columns and "nombre_trains_supprimes" in df.columns:
            total_retards = df["nombre_trains_retard"].sum()
            total_supprimes = df["nombre_trains_supprimes"].sum()

            if total_retards + total_supprimes == 0:
                st.info("Aucune perturbation enregistrée sur la période.")
            else:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Trains en Retard", "Trains Supprimés"],
                            values=[total_retards, total_supprimes],
                            hole=0.3,
                        )
                    ]
                )
                fig.update_layout(title="Répartition des Perturbations")
                st.plotly_chart(_apply_common_layout(fig), use_container_width=True)
        else:
            st.info("Colonnes de retards/suppressions manquantes.")

    with col2:
        if "mois" in df.columns and "nombre_trains_retard" in df.columns:
            df_monthly_retards = df.groupby("mois")["nombre_trains_retard"].sum()
            ordered_index = [m for m in _sort_months(df_monthly_retards.index) if m in df_monthly_retards.index]
            # fallback si la casse ou les valeurs ne correspondent pas exactement
            if not ordered_index:
                ordered_index = df_monthly_retards.index
            df_monthly_retards = df_monthly_retards.reindex(ordered_index)

            fig = px.bar(
                x=df_monthly_retards.index,
                y=df_monthly_retards.values,
                title="Nombre de Trains en Retard par Mois",
                labels={"x": "Mois", "y": "Nombre de Trains"},
            )
            st.plotly_chart(_apply_common_layout(fig), use_container_width=True)
        else:
            st.info("Colonnes `mois`/`nombre_trains_retard` manquantes.")


def plot_heatmap_regularite(df: pd.DataFrame) -> None:
    """Heatmap régularité par région et mois."""
    required = {"region", "mois", "taux_regularite"}
    if not required.issubset(df.columns):
        st.warning(f"⚠️ Colonnes manquantes pour la heatmap : {required - set(df.columns)}")
        return

    st.subheader("🔥 Heatmap : Régularité par Région et Mois")

    pivot = _pivot_mean(df, "region", "mois", "taux_regularite")
    if pivot.empty:
        st.info("Pas de données à afficher.")
        return

    ordered_cols = [m for m in _sort_months(pivot.columns) if m in pivot.columns]
    if ordered_cols:
        pivot = pivot[ordered_cols]

    fig = px.imshow(
        pivot,
        labels=dict(x="Mois", y="Région", color="Régularité (%)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=COLOR_SCALE,
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Visualisation personnalisée : dispatch table à la place du gros if/elif
# ---------------------------------------------------------------------------

def _build_line(df, x, y, color, size):
    return px.line(df, x=x, y=y, color=color, markers=True, title=f"Évolution de {y} selon {x}")

def _build_bar(df, x, y, color, size):
    return px.bar(df, x=x, y=y, color=color, title=f"{y} par {x}")

def _build_bar_h(df, x, y, color, size):
    return px.bar(df, x=y, y=x, color=color, orientation="h", title=f"{y} par {x}")

def _build_bar_stack(df, x, y, color, size):
    return px.bar(df, x=x, y=y, color=color, barmode="stack", title=f"{y} par {x} (empilé)")

def _build_bar_group(df, x, y, color, size):
    return px.bar(df, x=x, y=y, color=color, barmode="group", title=f"{y} par {x} (groupé)")

def _build_histogram(df, x, y, color, size):
    return px.histogram(df, x=x, color=color, nbins=30, title=f"Distribution de {x}")

def _build_box(df, x, y, color, size):
    return px.box(df, x=x, y=y, color=color, title=f"Distribution de {y} par {x}")

def _build_violin(df, x, y, color, size):
    return px.violin(df, x=x, y=y, color=color, box=True, title=f"Distribution de {y} par {x}")

def _build_pie(df, x, y, color, size):
    agg, names, values = _aggregate_for_share(df, x, y)
    return px.pie(agg, names=names, values=values, title=f"Répartition de {y or x} par {x}")

def _build_donut(df, x, y, color, size):
    agg, names, values = _aggregate_for_share(df, x, y)
    return px.pie(agg, names=names, values=values, hole=0.4, title=f"Répartition de {y or x} par {x}")

def _build_treemap(df, x, y, color, size):
    agg, names, values = _aggregate_for_share(df, x, y)
    return px.treemap(agg, path=[names], values=values, title=f"Treemap de {y or x}")

def _build_sunburst(df, x, y, color, size):
    if not color:
        st.warning("⚠️ Le Sunburst nécessite une colonne de couleur (2e dimension).")
        return None
    if y:
        agg = df.groupby([x, color])[y].sum().reset_index()
        values = y
    else:
        agg = df.groupby([x, color]).size().reset_index(name="count")
        values = "count"
    return px.sunburst(agg, path=[x, color], values=values, title=f"Sunburst de {values}")

def _build_scatter(df, x, y, color, size):
    return px.scatter(df, x=x, y=y, color=color, size=size, title=f"Relation entre {x} et {y}")

def _build_scatter_trend(df, x, y, color, size):
    return px.scatter(df, x=x, y=y, color=color, trendline="ols", title=f"Relation entre {x} et {y} (avec tendance)")

def _build_bubble(df, x, y, color, size):
    if not size:
        st.warning("⚠️ Le Bubble Chart nécessite une colonne de taille.")
        return None
    return px.scatter(df, x=x, y=y, color=color, size=size, title=f"Bubble chart : {x} vs {y}")

def _build_corr_heatmap(df, x, y, color, size):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.warning("⚠️ Il faut au moins deux colonnes numériques pour une matrice de corrélation.")
        return None
    corr_matrix = numeric_df.corr()
    return px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", title="Matrice de corrélation",
    )

def _build_custom_heatmap(df, x, y, color, size):
    if not color:
        st.warning("⚠️ La Heatmap personnalisée nécessite une colonne de couleur (2e dimension).")
        return None
    pivot_df = df.pivot_table(values=y, index=x, columns=color, aggfunc="mean")
    return px.imshow(pivot_df, text_auto=".2f", aspect="auto", title=f"Heatmap : {y} par {x} et {color}")

def _build_area(df, x, y, color, size):
    return px.area(df, x=x, y=y, color=color, title=f"Évolution cumulée de {y}")

def _build_funnel(df, x, y, color, size):
    agg, x_col, values_col = _aggregate_for_share(df, x, y)
    return px.funnel(agg, x=values_col, y=x_col, title=f"Funnel de {y or x}")

def _build_waterfall(df, x, y, color, size):
    if not y:
        st.warning("⚠️ Le Waterfall nécessite une colonne Y.")
        return None
    fig = go.Figure(go.Waterfall(x=df[x], y=df[y], connector={"line": {"color": "rgb(63, 63, 63)"}}))
    fig.update_layout(title=f"Waterfall : {y} par {x}")
    return fig

def _build_gauge(df, x, y, color, size):
    if not y or len(df) == 0:
        st.warning("⚠️ Le Gauge nécessite une colonne Y et des données.")
        return None
    value = df[y].mean()
    return go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": f"Moyenne de {y}"},
            gauge={
                "axis": {"range": [None, df[y].max()]},
                "bar": {"color": "darkblue"},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": df[y].quantile(0.75),
                },
            },
        )
    )

def _build_parallel_categories(df, x, y, color, size):
    cat_cols = [x] + ([color] if color else [])
    if len(cat_cols) < 2:
        st.warning("⚠️ Les Parallel Categories nécessitent au moins deux dimensions (X + couleur).")
        return None
    return px.parallel_categories(df, dimensions=cat_cols, color=y if y else None, title="Diagramme de Sankey catégoriel")

def _build_parallel_coordinates(df, x, y, color, size):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("⚠️ Les Parallel Coordinates nécessitent au moins deux colonnes numériques.")
        return None
    return px.parallel_coordinates(
        df, dimensions=numeric_cols[:5], color=y if y else numeric_cols[0],
        title="Coordonnées parallèles",
    )

def _build_density_heatmap(df, x, y, color, size):
    return px.density_heatmap(df, x=x, y=y, title=f"Densité : {x} vs {y}")

def _build_density_contour(df, x, y, color, size):
    return px.density_contour(df, x=x, y=y, color=color, title=f"Contours de densité : {x} vs {y}")

def _build_strip(df, x, y, color, size):
    return px.strip(df, x=x, y=y, color=color, title=f"Strip plot : {y} par {x}")

def _build_ecdf(df, x, y, color, size):
    return px.ecdf(df, x=x, color=color, title=f"Fonction de répartition empirique de {x}")


# Table de dispatch : plus facile à étendre/tester que le if/elif géant.
CHART_BUILDERS: Dict[str, Callable[..., Any]] = {
    "Ligne": _build_line,
    "Barre": _build_bar,
    "Barre horizontale": _build_bar_h,
    "Barre empilée": _build_bar_stack,
    "Barre groupée": _build_bar_group,
    "Histogramme": _build_histogram,
    "Box Plot": _build_box,
    "Violin Plot": _build_violin,
    "Camembert (Pie)": _build_pie,
    "Donut": _build_donut,
    "Treemap": _build_treemap,
    "Sunburst": _build_sunburst,
    "Scatter": _build_scatter,
    "Scatter avec tendance": _build_scatter_trend,
    "Bubble Chart": _build_bubble,
    "Heatmap (Matrice de corrélation)": _build_corr_heatmap,
    "Heatmap personnalisée": _build_custom_heatmap,
    "Area Chart": _build_area,
    "Funnel": _build_funnel,
    "Waterfall": _build_waterfall,
    "Gauge (Jauge)": _build_gauge,
    "Parallel Categories": _build_parallel_categories,
    "Parallel Coordinates": _build_parallel_coordinates,
    "Density Heatmap": _build_density_heatmap,
    "Density Contour": _build_density_contour,
    "Strip Plot": _build_strip,
    "ECDF": _build_ecdf,
}


def plot_custom_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Générateur de visualisation personnalisée.

    Utilise CHART_BUILDERS comme table de dispatch : chaque type de graphique
    correspond à une fonction `_build_*(df, x, y, color, size) -> Figure | None`.
    """
    builder = CHART_BUILDERS.get(chart_type)
    if builder is None:
        st.error(f"❌ Type de graphique inconnu : « {chart_type} ».")
        return None

    try:
        fig = builder(df, x_col, y_col, color_col, size_col)
    except KeyError as e:
        st.error(f"❌ Colonne introuvable : {e}")
        return None
    except (ValueError, TypeError) as e:
        st.error(f"❌ Données incompatibles avec « {chart_type} » : {e}")
        return None

    if fig is None:
        # Le builder a déjà affiché un st.warning() explicite si besoin.
        return None

    return _apply_common_layout(fig)