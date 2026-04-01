# app.py
# -*- coding: utf-8 -*-
"""
Dashboard d'analyse de la régularité des TER
Streamlit app — toutes les pages fonctionnelles
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

from config import Config
from data_loader import TERDataLoader
from weather_analyzer import WeatherAnalyzer
from ai_agent import TERAnalysisAgent
from visualizations import (
    plot_kpi_cards,
    plot_regularite_evolution,
    plot_regularite_by_region,
    plot_causes_retards,
    plot_heatmap_regularite,
    plot_custom_visualization,
)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
.kpi-box {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

for key in ['chat_history', 'agent', 'df_enriched', 'current_df_hash',
            'weather_analyzer', 'weather_results']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'chat_history' else []

# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_ter_data():
    """Charge les données TER avec cache d'une heure"""
    try:
        loader = TERDataLoader()
        return loader.load_data()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


with st.spinner("⏳ Chargement des données TER..."):
    df = load_ter_data()

if df is None or len(df) == 0:
    st.error("❌ Impossible de charger les données TER")
    st.info("""
    **Causes possibles :**
    - Azure Blob Storage mal configuré (vérifiez AZURE_STORAGE_CONNECTION_STRING)
    - L'API SNCF est temporairement indisponible
    - Problème de connexion réseau

    **Solutions :**
    - Vérifiez vos secrets Streamlit (AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_BLOB_NAME)
    - Réessayez dans quelques minutes
    """)
    st.stop()

# ─── KPIs globaux en bandeau ────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'taux_regularite' in df.columns:
        avg_reg = df['taux_regularite'].mean()
        st.metric("📊 Régularité moyenne", f"{avg_reg:.2f}%")

with col2:
    if 'region' in df.columns:
        st.metric("🗺️ Régions", df['region'].nunique())

with col3:
    if 'date' in df.columns:
        d_min = df['date'].min()
        d_max = df['date'].max()
        nb_mois = (d_max.year - d_min.year) * 12 + (d_max.month - d_min.month) + 1
        st.metric("📅 Période", f"{nb_mois} mois")

with col4:
    st.metric("📦 Enregistrements", f"{len(df):,}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.title("🚆 TER Analysis Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📊 Vue d'ensemble",
        "🔍 Explorateur de données",
        "📈 Visualisations personnalisées",
        "🌦️ Analyse Météo",
        "💬 Chat IA",
        "⚙️ Paramètres",
    ],
    index=0
)

st.sidebar.markdown("---")
source = "Azure Blob Storage" if Config.AZURE_CONNECTION_STRING else "API SNCF"
st.sidebar.info(f"""
**À propos**

Dashboard d'analyse de la régularité des trains TER.

**Données :** {source}
**IA :** Mistral AI ({Config.MISTRAL_MODEL})
**Météo :** Open-Meteo
""")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🏠 ACCUEIL
# ═══════════════════════════════════════════════════════════════════════

if page == "🏠 Accueil":
    st.title("🚆 Dashboard d'Analyse TER")

    st.markdown("""
    ## Bienvenue sur le Dashboard d'Analyse de la Régularité des TER

    Ce tableau de bord vous permet d'analyser la ponctualité des trains TER en France,
    d'enrichir les données avec la météo et d'interroger un assistant IA.

    ### 📊 Fonctionnalités disponibles

    **📊 Vue d'ensemble** — statistiques globales, comparaisons régions, évolution temporelle

    **🔍 Explorateur de données** — filtres avancés, tableau interactif, export CSV/Excel

    **📈 Visualisations personnalisées** — créez vos propres graphiques

    **🌦️ Analyse Météo** — enrichissement API Open-Meteo, corrélations météo/régularité

    **💬 Chat IA** — posez vos questions en langage naturel à l'assistant Mistral AI

    **⚙️ Paramètres** — configuration, informations sur le dataset

    ### 🚀 Pour commencer

    1. Explorez la **Vue d'ensemble** pour les statistiques globales
    2. Enrichissez les données via **Analyse Météo** pour des analyses approfondies
    3. Utilisez le **Chat IA** pour des questions spécifiques
    """)

    st.markdown("---")
    st.subheader("📈 Aperçu rapide")

    if 'taux_regularite' in df.columns and 'region' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**🏆 Top 5 régions les plus régulières**")
            top5 = df.groupby('region')['taux_regularite'].mean().nlargest(5)
            for i, (region, taux) in enumerate(top5.items(), 1):
                st.write(f"{i}. {region} : {taux:.2f}%")

        with col2:
            st.write("**⚠️ Top 5 régions les moins régulières**")
            bottom5 = df.groupby('region')['taux_regularite'].mean().nsmallest(5)
            for i, (region, taux) in enumerate(bottom5.items(), 1):
                st.write(f"{i}. {region} : {taux:.2f}%")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 📊 VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════

elif page == "📊 Vue d'ensemble":
    st.title("📊 Vue d'ensemble")

    with st.expander("🔧 Filtres", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'region' in df.columns:
                regions = ['Toutes'] + sorted(df['region'].unique().tolist())
                selected_region = st.selectbox("Région", regions)
            else:
                selected_region = 'Toutes'

        with col2:
            if 'date' in df.columns:
                annees = ['Toutes'] + sorted(df['date'].dt.year.unique().tolist(), reverse=True)
                selected_annee = st.selectbox("Année", annees)
            else:
                selected_annee = 'Toutes'

        with col3:
            if 'date' in df.columns:
                mois_labels = {
                    0: 'Tous', 1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
                    5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
                    9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
                }
                selected_mois = st.selectbox("Mois", list(mois_labels.keys()),
                                             format_func=lambda x: mois_labels[x])
            else:
                selected_mois = 0

    # Application des filtres
    df_filtered = df.copy()
    if selected_region != 'Toutes' and 'region' in df.columns:
        df_filtered = df_filtered[df_filtered['region'] == selected_region]
    if selected_annee != 'Toutes' and 'date' in df.columns:
        df_filtered = df_filtered[df_filtered['date'].dt.year == int(selected_annee)]
    if selected_mois != 0 and 'date' in df.columns:
        df_filtered = df_filtered[df_filtered['date'].dt.month == selected_mois]

    st.caption(f"📊 {len(df_filtered):,} enregistrements après filtres")

    plot_kpi_cards(df_filtered)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        plot_regularite_evolution(df_filtered)
    with col2:
        plot_regularite_by_region(df_filtered)

    st.markdown("---")
    plot_causes_retards(df_filtered)
    st.markdown("---")
    plot_heatmap_regularite(df_filtered)

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🔍 EXPLORATEUR DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔍 Explorateur de données":
    st.title("🔍 Explorateur de Données")

    with st.expander("🔧 Filtres avancés", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            regions_filter = []
            if 'region' in df.columns:
                regions_filter = st.multiselect(
                    "Régions",
                    options=sorted(df['region'].unique()),
                    default=[]
                )

        with col2:
            annees_filter = []
            if 'date' in df.columns:
                annees_filter = st.multiselect(
                    "Années",
                    options=sorted(df['date'].dt.year.unique(), reverse=True),
                    default=[]
                )

        with col3:
            regularite_range = None
            if 'taux_regularite' in df.columns:
                min_r = float(df['taux_regularite'].min())
                max_r = float(df['taux_regularite'].max())
                regularite_range = st.slider(
                    "Plage de régularité (%)",
                    min_value=min_r, max_value=max_r,
                    value=(min_r, max_r)
                )

    df_explore = df.copy()
    if regions_filter and 'region' in df.columns:
        df_explore = df_explore[df_explore['region'].isin(regions_filter)]
    if annees_filter and 'date' in df.columns:
        df_explore = df_explore[df_explore['date'].dt.year.isin(annees_filter)]
    if regularite_range and 'taux_regularite' in df.columns:
        df_explore = df_explore[
            df_explore['taux_regularite'].between(regularite_range[0], regularite_range[1])
        ]

    st.markdown(f"### 📊 {len(df_explore):,} enregistrements")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Lignes", f"{len(df_explore):,}")
    if 'region' in df_explore.columns:
        col2.metric("🗺️ Régions", df_explore['region'].nunique())
    if 'taux_regularite' in df_explore.columns:
        col3.metric("📊 Régularité moy.", f"{df_explore['taux_regularite'].mean():.1f}%")
    col4.metric("💾 Taille", f"{df_explore.memory_usage(deep=True).sum() / 1024:.0f} KB")

    st.markdown("### 📋 Tableau de données")
    col1, col2 = st.columns(2)
    with col1:
        cols_show = st.multiselect(
            "Colonnes à afficher",
            options=df_explore.columns.tolist(),
            default=df_explore.columns.tolist()[:8]
        )
    with col2:
        rows_pp = st.selectbox("Lignes par page", [10, 25, 50, 100], index=1)

    if cols_show:
        st.dataframe(df_explore[cols_show].head(rows_pp), use_container_width=True, height=400)

    st.markdown("### 💾 Export")
    col1, col2 = st.columns(2)

    with col1:
        csv = df_explore.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger CSV",
            data=csv,
            file_name=f"ter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_explore.to_excel(writer, index=False, sheet_name='TER_Data')
        buf.seek(0)
        st.download_button(
            "📥 Télécharger Excel",
            data=buf,
            file_name=f"ter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 📈 VISUALISATIONS PERSONNALISÉES
# ═══════════════════════════════════════════════════════════════════════

elif page == "📈 Visualisations personnalisées":
    st.title("📈 Créateur de Visualisations")
    st.markdown("Créez vos propres graphiques en sélectionnant les paramètres ci-dessous.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")

        # Catégorisation des graphiques
        chart_categories = {
            "📊 Graphiques de base": [
                "Ligne", "Barre", "Barre horizontale",
                "Barre empilée", "Barre groupée", "Area Chart"
            ],
            "📉 Distribution": [
                "Histogramme", "Box Plot", "Violin Plot",
                "Strip Plot", "ECDF"
            ],
            "🥧 Proportions": [
                "Camembert (Pie)", "Donut", "Treemap",
                "Sunburst", "Funnel"
            ],
            "🔍 Relations": [
                "Scatter", "Scatter avec tendance",
                "Bubble Chart", "Density Heatmap", "Density Contour"
            ],
            "🌡️ Matrices": [
                "Heatmap (Matrice de corrélation)",
                "Heatmap personnalisée"
            ],
            "🎯 Avancés": [
                "Waterfall", "Gauge (Jauge)",
                "Parallel Categories", "Parallel Coordinates"
            ]
        }

        # Sélection
        category = st.selectbox(
            "Catégorie de graphique",
            list(chart_categories.keys())
        )

        chart_type = st.selectbox(
            "Type de graphique",
            chart_categories[category]
        )

        # Aide contextuelle
        help_texts = {
            "Camembert (Pie)": "📌 Idéal pour montrer des proportions (ex: répartition par région)",
            "Treemap": "📌 Affiche des proportions hiérarchiques avec des rectangles",
            "Heatmap (Matrice de corrélation)": "📌 Affiche les corrélations entre variables numériques",
            "Bubble Chart": "📌 Nécessite 3 dimensions : X, Y et Taille",
            "Waterfall": "📌 Montre l'évolution cumulative (positif/négatif)",
            "Sunburst": "📌 Nécessite au moins 2 colonnes catégorielles",
            "Parallel Coordinates": "📌 Compare plusieurs variables numériques simultanément"
        }

        if chart_type in help_texts:
            st.info(help_texts[chart_type])

        # Colonnes
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        all_cols = df.columns.tolist()

        # Besoins
        needs_y = chart_type not in [
            "Histogramme", "Camembert (Pie)", "Donut",
            "Treemap", "ECDF", "Heatmap (Matrice de corrélation)"
        ]

        needs_numeric_x = chart_type in [
            "Scatter", "Scatter avec tendance", "Bubble Chart",
            "Density Heatmap", "Density Contour",
            "Heatmap (Matrice de corrélation)"
        ]

        # Axe X
        if chart_type == "Heatmap (Matrice de corrélation)":
            x_col = None
            st.info("📊 La matrice utilisera toutes les colonnes numériques")
        else:
            x_col = st.selectbox(
                "Axe X",
                numeric_cols if needs_numeric_x else all_cols
            )

        # Axe Y
        y_col = None
        if needs_y:
            y_col = st.selectbox("Axe Y", numeric_cols)

        # Couleur
        use_color = st.checkbox("Ajouter une dimension couleur")
        color_col = None
        if use_color and chart_type != "Heatmap (Matrice de corrélation)":
            color_col = st.selectbox(
                "Colonne de couleur",
                categorical_cols + numeric_cols
            )

        # Taille
        size_col = None
        if chart_type == "Bubble Chart":
            use_size = st.checkbox("Ajouter une dimension taille", value=True)
            if use_size:
                size_col = st.selectbox("Colonne de taille", numeric_cols)

        st.markdown("---")

        # Filtres
        if "region" in df.columns:
            filter_region = st.multiselect(
                "🗺️ Filtrer par régions",
                options=sorted(df["region"].unique()),
                default=[]
            )
        else:
            filter_region = []

        # Filtre date
        if "date" in df.columns:
            use_date_filter = st.checkbox("📅 Filtrer par période")
            if use_date_filter:
                min_date = df["date"].min()
                max_date = df["date"].max()

                date_range = st.date_input(
                    "Sélectionner la période",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

        generate_viz = st.button(
            "🎨 Générer le graphique",
            use_container_width=True,
            type="primary"
        )

    with col2:
        st.subheader("📊 Résultat")

        if generate_viz:
            df_viz = df.copy()

            # Filtres
            if filter_region and "region" in df.columns:
                df_viz = df_viz[df_viz["region"].isin(filter_region)]

            if "date" in df.columns and use_date_filter and len(date_range) == 2:
                df_viz = df_viz[
                    (df_viz["date"] >= pd.to_datetime(date_range[0])) &
                    (df_viz["date"] <= pd.to_datetime(date_range[1]))
                ]

            # Vérification
            if len(df_viz) == 0:
                st.warning("⚠️ Aucune donnée après application des filtres.")
            else:
                st.info(f"📊 Graphique généré avec {len(df_viz):,} enregistrements")

                fig = plot_custom_visualization(
                    df_viz, chart_type, x_col, y_col, color_col, size_col
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats
                    with st.expander("📈 Statistiques du graphique"):
                        if y_col and y_col in df_viz.columns:
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                st.metric("Moyenne", f"{df_viz[y_col].mean():.2f}")
                            with col_b:
                                st.metric("Médiane", f"{df_viz[y_col].median():.2f}")
                            with col_c:
                                st.metric("Écart-type", f"{df_viz[y_col].std():.2f}")

                    # Téléchargement
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        st.download_button(
                            "💾 Télécharger (HTML)",
                            data=fig.to_html(),
                            file_name=f"chart_{chart_type.lower().replace(' ', '_')}_{x_col}.html",
                            mime="text/html",
                            use_container_width=True
                        )

                    with col_dl2:
                        try:
                            img_bytes = fig.to_image(format="png")

                            st.download_button(
                                "🖼️ Télécharger (PNG)",
                                data=img_bytes,
                                file_name=f"chart_{chart_type.lower().replace(' ', '_')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.caption("⚠️ Export PNG indisponible (installer kaleido)")
                else:
                    st.error("❌ Impossible de créer le graphique avec ces paramètres.")

        else:
            st.info("👈 Configurez votre graphique et cliquez sur 'Générer'")

            with st.expander("💡 Exemples de visualisations"):
                st.markdown("""
                **Suggestions selon vos besoins :**

                - **Comparer des valeurs** → Barre, Barre groupée
                - **Montrer une évolution** → Ligne, Area Chart
                - **Voir des proportions** → Camembert, Treemap, Donut
                - **Analyser une distribution** → Histogramme, Box Plot, Violin
                - **Trouver des corrélations** → Scatter, Heatmap
                - **Identifier des tendances** → Scatter avec tendance
                - **Comparer plusieurs dimensions** → Bubble Chart, Parallel Coordinates
                - **Visualiser des flux** → Waterfall, Funnel, Sunburst
                """)
# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🌦️ ANALYSE MÉTÉO
# ═══════════════════════════════════════════════════════════════════════

elif page == "🌦️ Analyse Météo":
    st.title("🌦️ Analyse de l'Impact Météorologique")

    st.markdown("""
    Enrichissez les données TER avec des données météo historiques
    (via **Open-Meteo**, gratuit et sans clé) puis analysez leur impact sur la régularité.
    """)

    if st.session_state.weather_analyzer is None:
        st.session_state.weather_analyzer = WeatherAnalyzer(df)

    weather_analyzer = st.session_state.weather_analyzer

    # ─── Étape 1 : Enrichissement ────────────────────────────────────

    st.markdown("---")
    st.subheader("📥 Étape 1 : Enrichissement avec données météo")

    with st.expander("⚙️ Configuration de l'enrichissement", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            sample_size = st.slider(
                "Nombre d'enregistrements à enrichir",
                min_value=100,
                max_value=min(5000, len(df)),
                value=min(1000, len(df)),
                step=100,
                help="Plus le nombre est élevé, plus l'enrichissement prend de temps."
            )

        with col2:
            use_openweather = st.checkbox("Utiliser OpenWeatherMap (optionnel)")
            openweather_key = None
            if use_openweather:
                openweather_key = st.text_input(
                    "Clé API OpenWeatherMap",
                    type="password",
                    help="Obtenez une clé gratuite sur https://openweathermap.org/api"
                )

        st.info("""
        💡 **Comment ça marche ?**
        1. L'app récupère les données météo historiques pour chaque date via Open-Meteo
        2. Les données sont associées aux régions via les grandes villes
        3. Un score de sévérité météo (0-100) est calculé
        4. Les corrélations avec la régularité sont analysées
        """)

        if st.button("🚀 Lancer l'enrichissement", type="primary", use_container_width=True):
            with st.spinner("🌦️ Enrichissement en cours..."):
                df_enriched = weather_analyzer.enrich_with_weather(
                    sample_size=sample_size,
                    use_api_key=openweather_key if use_openweather else None
                )
                if df_enriched is not None:
                    st.session_state.df_enriched = df_enriched
                    st.success("✅ Enrichissement terminé !")
                    st.balloons()

    # ─── Étape 2 : Analyse ────────────────────────────────────────────

    if st.session_state.df_enriched is not None:
        df_enriched = st.session_state.df_enriched

        st.markdown("---")
        st.subheader("📊 Étape 2 : Aperçu des données météo")

        col1, col2, col3, col4 = st.columns(4)
        if 'temperature_mean' in df_enriched.columns:
            col1.metric("🌡️ Température moy.", f"{df_enriched['temperature_mean'].mean():.1f}°C")
        if 'precipitation' in df_enriched.columns:
            col2.metric("🌧️ Précipitations", f"{df_enriched['precipitation'].sum():.0f} mm")
        if 'snow' in df_enriched.columns:
            col3.metric("❄️ Jours neige", f"{(df_enriched['snow'] > 0).sum()}")
        if 'wind_gusts' in df_enriched.columns:
            col4.metric("💨 Rafale max", f"{df_enriched['wind_gusts'].max():.0f} km/h")

        weather_cols_show = [c for c in df_enriched.columns if c in [
            'date', 'region', 'taux_regularite',
            'temperature_mean', 'precipitation', 'snow',
            'wind_speed', 'wind_gusts', 'weather_severity_score'
        ]]
        st.dataframe(df_enriched[weather_cols_show].head(20), use_container_width=True)

        st.markdown("---")
        st.subheader("🔬 Étape 3 : Analyse des corrélations")

        if st.button("📊 Analyser l'impact météo", type="primary", use_container_width=True):
            with st.spinner("🔬 Analyse en cours..."):
                results = weather_analyzer.analyze_weather_impact()
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.session_state.weather_results = results

        if st.session_state.weather_results:
            results = st.session_state.weather_results

            if 'correlation_regularite_meteo' in results:
                corr = results['correlation_regularite_meteo']
                r = corr['correlation']
                color = "red" if abs(r) > 0.6 else "orange" if abs(r) > 0.3 else "blue"
                st.markdown(f"""
                <div style='padding:1.5rem;background:#f0f2f6;border-radius:10px;border-left:5px solid {color}'>
                    <h3 style='margin:0;color:{color}'>Corrélation : {r:.3f}</h3>
                    <p>{corr['interpretation']} | {corr['significance']} (p={corr['p_value']:.4f})</p>
                </div>
                """, unsafe_allow_html=True)

            impacts = {
                'impact_neige': ("❄️ Impact de la Neige",
                                 'regularite_sans_neige', 'regularite_avec_neige'),
                'impact_vent': ("💨 Impact du Vent Fort (> 90 km/h)",
                                'regularite_vent_normal', 'regularite_vent_fort'),
                'impact_pluie': ("🌧️ Impact de la Pluie Forte (> 10 mm)",
                                 'regularite_pluie_faible', 'regularite_pluie_forte'),
            }

            for key, (title, col_ref, col_impact) in impacts.items():
                if key in results:
                    imp = results[key]
                    st.markdown(f"### {title}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sans perturbation", f"{imp[col_ref]:.2f}%")
                    c2.metric("Avec perturbation", f"{imp[col_impact]:.2f}%",
                              delta=f"-{imp['difference']:.2f}%", delta_color="inverse")
                    c3.metric("Perte", f"{imp['difference']:.2f}%")

            if 'retards_par_meteo' in results:
                df_plot = pd.DataFrame(
                    list(results['retards_par_meteo'].items()),
                    columns=['Condition', 'Retards moyens']
                )
                fig = px.bar(
                    df_plot, x='Condition', y='Retards moyens',
                    color='Condition',
                    color_discrete_map={'Bonne': 'green', 'Correcte': 'yellow',
                                        'Difficile': 'orange', 'Extrême': 'red'},
                    title="Trains en retard selon les conditions météo"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Étape 4 : Visualisations avancées")

        if st.button("🎨 Générer les visualisations", use_container_width=True):
            weather_analyzer.plot_weather_impact()

        st.markdown("---")
        st.subheader("💾 Étape 5 : Export des données enrichies")

        col1, col2 = st.columns(2)

        with col1:
            csv_e = df_enriched.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger CSV (avec météo)",
                data=csv_e,
                file_name=f"ter_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df_enriched.to_excel(writer, index=False, sheet_name='TER_Weather')
            buf.seek(0)
            st.download_button(
                "📥 Télécharger Excel (avec météo)",
                data=buf,
                file_name=f"ter_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    else:
        st.info("👆 Lancez d'abord l'enrichissement pour accéder aux analyses.")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 💬 CHAT IA
# ═══════════════════════════════════════════════════════════════════════

elif page == "💬 Chat IA":
    st.title("💬 Chat avec l'Assistant IA")

    st.markdown("""
    Posez vos questions en français. L'IA analyse les données TER et la météo
    et génère automatiquement des graphiques si nécessaire.
    """)

    if not Config.MISTRAL_API_KEY:
        st.error("❌ **Clé API Mistral non configurée**")
        st.markdown("""
        ### 🔑 Comment obtenir une clé API Mistral ?
        1. Créez un compte sur https://console.mistral.ai/
        2. Allez dans **"API Keys"** → **"Create API Key"**
        3. Copiez la clé

        ### ⚙️ Configuration Streamlit
        - Settings → Secrets → `MISTRAL_API_KEY = "votre_clé"`
        """)
        st.stop()

    # Initialisation de l'agent
    agent_df = st.session_state.df_enriched if st.session_state.df_enriched is not None else df
    df_hash = hash(str(agent_df.shape) + str(sorted(agent_df.columns.tolist())))

    if st.session_state.agent is None or st.session_state.current_df_hash != df_hash:
        try:
            with st.spinner("🤖 Initialisation de l'agent IA..."):
                st.session_state.agent = TERAnalysisAgent(agent_df)
                st.session_state.current_df_hash = df_hash
            st.success("✅ Agent IA prêt !", icon="🤖")
        except Exception as e:
            st.error(f"❌ **Erreur d'initialisation** : {str(e)}")
            st.stop()

    col_header1, col_header2 = st.columns([4, 1])
    with col_header2:
        if st.button("🔄 Recharger", help="Réinitialiser l'agent"):
            st.session_state.agent = None
            st.session_state.current_df_hash = None
            st.rerun()

    with st.expander("💡 Exemples de questions", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📊 Régularité & régions :**
            - Quelle est la régularité moyenne globale ?
            - Quelle région est la plus régulière ?
            - Quelles sont les 5 pires régions ?
            - Analyse complète de la Bretagne

            **📅 Filtrage temporel :**
            - Combien de trains annulés en avril 2020 en Bretagne ?
            - Compare avril 2020 et avril 2019
            - Régularité en 2023 en Normandie ?
            """)

        with col2:
            st.markdown("""
            **🌦️ Météo (après enrichissement) :**
            - Quel est l'impact de la neige sur les retards ?
            - Analyse l'impact du vent fort
            - Corrélation entre météo et régularité ?
            - Impact de la pluie forte ?

            **📈 Graphiques :**
            - Montre-moi un graphique par région
            - Évolution de la régularité dans le temps
            - Compare les 10 meilleures régions en barres
            """)

    if st.session_state.df_enriched is not None:
        st.success("✅ Données enrichies avec météo disponibles — l'IA y a accès.")
    else:
        st.info("ℹ️ Enrichissez les données via 'Analyse Météo' pour activer les questions météo.")

    st.markdown("---")

    # Historique du chat
    if not st.session_state.chat_history:
        st.info("👋 **Bonjour !** Posez-moi une question sur les données TER.", icon="🤖")
    else:
        for idx, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message['content'])
                    if message.get('figure') is not None:
                        st.plotly_chart(
                            message['figure'],
                            use_container_width=True,
                            key=f"chart_{idx}"
                        )

    # Zone de saisie
    user_question = st.chat_input("💬 Posez votre question ici...")

    if user_question:
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Analyse en cours..."):
                try:
                    response = st.session_state.agent.ask(user_question)
                    st.markdown(response)

                    # Génération automatique de graphique si demandé
                    plot_keywords = [
                        'graphique', 'graph', 'courbe', 'trace', 'dessine', 'montre',
                        'visualise', 'affiche', 'camembert', 'histogramme', 'barres',
                        'plot', 'chart', 'diagramme', 'évolution', 'compare', 'comparaison',
                        'heatmap', 'scatter', 'box plot', 'pie', 'tendance'
                    ]
                    should_plot = any(k in user_question.lower() for k in plot_keywords)

                    response_msg = {'role': 'assistant', 'content': response, 'figure': None}

                    if should_plot:
                        try:
                            fig = _generate_smart_chart(user_question, agent_df)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                response_msg['figure'] = fig
                        except Exception as plot_err:
                            st.warning(f"⚠️ Graphique non généré : {str(plot_err)}")

                    st.session_state.chat_history.append(response_msg)

                except Exception as e:
                    err = f"❌ **Erreur** : {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {'role': 'assistant', 'content': err, 'figure': None}
                    )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()

    with col2:
        nb_q = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        nb_c = len([m for m in st.session_state.chat_history if m.get('figure') is not None])
        st.metric("💬 Questions", nb_q)
        st.metric("📊 Graphiques", nb_c)

    with col3:
        if st.session_state.df_enriched is not None:
            st.success("✅ Données enrichies disponibles")
        else:
            st.info("ℹ️ Enrichissez via 'Analyse Météo'")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : ⚙️ PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════

elif page == "⚙️ Paramètres":
    st.title("⚙️ Paramètres de l'Application")

    st.subheader("🔧 Configuration actuelle")
    col1, col2 = st.columns(2)

    with col1:
        azure_ok = bool(Config.AZURE_CONNECTION_STRING)
        st.info(f"""
        **Azure Blob Storage**
        - Conteneur : `{Config.AZURE_CONTAINER_NAME}`
        - Fichier : `{Config.AZURE_BLOB_NAME}`
        - Connexion : {'✅ Configurée' if azure_ok else '❌ Non configurée (fallback API SNCF)'}
        """)

    with col2:
        mistral_ok = bool(Config.MISTRAL_API_KEY)
        st.success(f"""
        **Mistral AI**
        - Modèle : `{Config.MISTRAL_MODEL}`
        - API Key : {'✅ Configurée' if mistral_ok else '❌ Non configurée'}
        - Agent IA : {'✅ Actif' if st.session_state.agent else '❌ Inactif'}
        """)

    st.markdown("---")
    st.subheader("📊 Informations sur le dataset")

    st.dataframe(
        pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            '% Null': (df.isnull().sum() / len(df) * 100).round(2)
        }),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🔄 Actions de maintenance")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("🗑️ Vider le cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache vidé !")

    with col3:
        if st.button("🔄 Réinitialiser l'app", use_container_width=True):
            st.session_state.clear()
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.info("""
    **Assistant IA — Analyse TER SNCF**

    Version : 2.0.0
    Stack : Streamlit + LangGraph + Mistral AI (mistral-large-latest)
    Données : Azure Blob Storage / SNCF Open Data
    Météo : Open-Meteo (gratuit, sans clé)
    """)

# ═══════════════════════════════════════════════════════════════════════
# HELPER : GÉNÉRATION INTELLIGENTE DE GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════
@tools
def _generate_smart_chart(question: str, df: pd.DataFrame):
    """
    Génère automatiquement un graphique Plotly selon la question posée.
    Retourne None si aucun graphique pertinent n'est trouvé.
    """
    q = question.lower()

    if 'taux_regularite' not in df.columns:
        return None

    # Évolution temporelle
    if any(w in q for w in ['évolution', 'evolution', 'temps', 'tendance', 'courbe', 'mois']):
        if 'date' not in df.columns:
            return None
        time_stats = df.groupby('date')['taux_regularite'].mean().reset_index()
        fig = px.line(
            time_stats, x='date', y='taux_regularite',
            title="📈 Évolution de la régularité",
            labels={'date': 'Date', 'taux_regularite': 'Régularité (%)'},
            markers=True
        )
        if len(time_stats) >= 7:
            fig.add_scatter(
                x=time_stats['date'],
                y=time_stats['taux_regularite'].rolling(7).mean(),
                mode='lines', name='Tendance (7j)',
                line=dict(color='red', dash='dash')
            )
        return fig

    # Comparaison régions spécifiques
    if 'compare' in q or 'comparaison' in q:
        if 'region' not in df.columns:
            return None
        mentioned = [r for r in df['region'].unique() if r.lower() in q]
        if len(mentioned) >= 2:
            region_stats = df[df['region'].isin(mentioned)].groupby('region')['taux_regularite'].mean()
            fig = px.bar(
                x=region_stats.index, y=region_stats.values,
                title=f"📊 Comparaison : {' vs '.join(mentioned)}",
                labels={'x': 'Région', 'y': 'Régularité (%)'},
                color=region_stats.values, color_continuous_scale='RdYlGn',
                text=region_stats.values.round(2)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_coloraxes(showscale=False)
            return fig

    # Distribution / histogramme
    if any(w in q for w in ['distribution', 'histogramme', 'histogram', 'répartition']):
        fig = px.histogram(
            df, x='taux_regularite', nbins=40,
            title="📊 Distribution des taux de régularité",
            labels={'taux_regularite': 'Régularité (%)'}
        )
        mean_val = df['taux_regularite'].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne: {mean_val:.1f}%")
        return fig

    # Box plot
    if 'box' in q or 'boxplot' in q:
        if 'region' not in df.columns:
            return None
        top_r = df.groupby('region')['taux_regularite'].mean().nlargest(10).index
        fig = px.box(
            df[df['region'].isin(top_r)],
            x='region', y='taux_regularite',
            title="📦 Distribution par région (Top 10)",
            color='region'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        return fig

    # Impact météo (neige)
    if any(w in q for w in ['neige', 'météo', 'meteo', 'vent', 'pluie', 'température']):
        if 'weather_severity_score' in df.columns:
            d = df.copy()
            d['meteo_cat'] = pd.cut(
                d['weather_severity_score'],
                bins=[-1, 20, 40, 60, 100],
                labels=['Bonne', 'Correcte', 'Difficile', 'Extrême']
            )
            stats = d.groupby('meteo_cat')['taux_regularite'].mean()
            fig = px.bar(
                x=stats.index, y=stats.values,
                title="🌦️ Régularité selon la sévérité météo",
                labels={'x': 'Condition météo', 'y': 'Régularité (%)'},
                color=['green', 'yellow', 'orange', 'red'][:len(stats)],
                text=stats.values.round(2)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            return fig

    # Défaut : Top 10 régions par régularité
    if 'region' not in df.columns:
        return None

    n = 5
    if '10' in q or 'top 10' in q:
        n = 10
    if 'pire' in q or 'worst' in q:
        region_stats = df.groupby('region')['taux_regularite'].mean().nsmallest(n)
        title = f"📉 {n} pires régions"
    else:
        region_stats = df.groupby('region')['taux_regularite'].mean().nlargest(n)
        title = f"🏆 Top {n} régions"

    if 'camembert' in q or 'pie' in q:
        fig = px.pie(
            values=region_stats.values, names=region_stats.index,
            title=title, color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        fig = px.bar(
            x=region_stats.index, y=region_stats.values,
            title=title,
            labels={'x': 'Région', 'y': 'Régularité (%)'},
            color=region_stats.values, color_continuous_scale='RdYlGn',
            text=region_stats.values.round(2)
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        fig.update_coloraxes(showscale=False)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;padding:1rem;'>"
    "🚆 Assistant IA TER SNCF | Streamlit · LangGraph · Mistral AI · Open-Meteo"
    "</div>",
    unsafe_allow_html=True
)
