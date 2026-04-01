
# -*- coding: utf-8 -*-
"""
Module de génération de visualisations pour l'analyse TER
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from config import Config

# Style matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def plot_kpi_cards(df):
    """Affiche des cartes KPI en haut du dashboard"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # KPI 1 : Taux de régularité moyen
    if 'taux_regularite' in df.columns:
        avg_regularite = df['taux_regularite'].mean()
        delta_regularite = avg_regularite - 90  # Comparaison à un objectif de 90%
        
        with col1:
            st.metric(
                label="📊 Régularité Moyenne",
                value=f"{avg_regularite:.1f}%",
                delta=f"{delta_regularite:+.1f}% vs objectif",
                delta_color="normal" if avg_regularite >= 90 else "inverse"
            )
    
    # KPI 2 : Total trains
    if 'nombre_trains_prevus' in df.columns:
        total_trains = df['nombre_trains_prevus'].sum()
        with col2:
            st.metric(
                label="🚂 Trains Prévus",
                value=f"{total_trains:,.0f}"
            )
    
    # KPI 3 : Trains supprimés
    if 'nombre_trains_supprimes' in df.columns:
        total_supprimes = df['nombre_trains_supprimes'].sum()
        taux_suppression = (total_supprimes / total_trains * 100) if total_trains > 0 else 0
        
        with col3:
            st.metric(
                label="❌ Trains Supprimés",
                value=f"{total_supprimes:,.0f}",
                delta=f"{taux_suppression:.2f}%"
            )
    
    # KPI 4 : Nombre de régions
    if 'region' in df.columns:
        nb_regions = df['region'].nunique()
        with col4:
            st.metric(
                label="🗺️ Régions Analysées",
                value=f"{nb_regions}"
            )

def plot_regularite_evolution(df):
    """Graphique d'évolution de la régularité dans le temps"""
    
    if 'date' in df.columns and 'taux_regularite' in df.columns:
        st.subheader("📈 Évolution de la Régularité")
        
        # Agrégation mensuelle
        df_monthly = df.groupby(pd.Grouper(key='date', freq='M'))['taux_regularite'].mean().reset_index()
        
        # Graphique Plotly interactif
        fig = px.line(
            df_monthly,
            x='date',
            y='taux_regularite',
            title="Taux de Régularité Mensuel",
            labels={'date': 'Date', 'taux_regularite': 'Taux de Régularité (%)'},
            markers=True
        )
        
        # Ligne d'objectif
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="red",
            annotation_text="Objectif 90%"
        )
        
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

def plot_regularite_by_region(df):
    """Graphique de régularité par région"""
    
    if 'region' in df.columns and 'taux_regularite' in df.columns:
        st.subheader("🗺️ Régularité par Région")
        
        # Calcul de la moyenne par région
        df_region = df.groupby('region')['taux_regularite'].mean().sort_values(ascending=True)
        
        # Graphique horizontal
        fig = px.bar(
            x=df_region.values,
            y=df_region.index,
            orientation='h',
            title="Taux de Régularité Moyen par Région",
            labels={'x': 'Taux de Régularité (%)', 'y': 'Région'},
            color=df_region.values,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def plot_causes_retards(df):
    """Distribution des causes de retards (si disponible)"""
    
    # Cette fonction sera adaptée selon tes vraies colonnes
    st.subheader("🔍 Analyse des Perturbations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'nombre_trains_retard' in df.columns and 'nombre_trains_supprimes' in df.columns:
            # Pie chart des perturbations
            total_retards = df['nombre_trains_retard'].sum()
            total_supprimes = df['nombre_trains_supprimes'].sum()
            
            fig = go.Figure(data=[go.Pie(
                labels=['Trains en Retard', 'Trains Supprimés'],
                values=[total_retards, total_supprimes],
                hole=.3
            )])
            
            fig.update_layout(title="Répartition des Perturbations")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'mois' in df.columns and 'nombre_trains_retard' in df.columns:
            # Évolution mensuelle des retards
            df_monthly_retards = df.groupby('mois')['nombre_trains_retard'].sum().sort_index()
            
            fig = px.bar(
                x=df_monthly_retards.index,
                y=df_monthly_retards.values,
                title="Nombre de Trains en Retard par Mois",
                labels={'x': 'Mois', 'y': 'Nombre de Trains'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_regularite(df):
    """Heatmap régularité par région et mois"""
    
    if all(col in df.columns for col in ['region', 'mois', 'taux_regularite']):
        st.subheader("🔥 Heatmap : Régularité par Région et Mois")
        
        # Pivot table
        pivot = df.pivot_table(
            values='taux_regularite',
            index='region',
            columns='mois',
            aggfunc='mean'
        )
        
        # Heatmap avec Plotly
        fig = px.imshow(
            pivot,
            labels=dict(x="Mois", y="Région", color="Régularité (%)"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale='RdYlGn',
            aspect="auto"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def plot_custom_visualization(df, chart_type, x_col, y_col=None, color_col=None, size_col=None):
    """Générateur de visualisation personnalisée avec tous les types de graphiques"""

    try:
        # Graphiques de base
        if chart_type == "Ligne":
            fig = px.line(
                df, x=x_col, y=y_col, color=color_col, markers=True,
                title=f"Évolution de {y_col} selon {x_col}"
            )

        elif chart_type == "Barre":
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col,
                title=f"{y_col} par {x_col}"
            )

        elif chart_type == "Barre horizontale":
            fig = px.bar(
                df, x=y_col, y=x_col, color=color_col, orientation='h',
                title=f"{y_col} par {x_col}"
            )

        elif chart_type == "Barre empilée":
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col, barmode='stack',
                title=f"{y_col} par {x_col} (empilé)"
            )

        elif chart_type == "Barre groupée":
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col, barmode='group',
                title=f"{y_col} par {x_col} (groupé)"
            )

        # Graphiques de distribution
        elif chart_type == "Histogramme":
            fig = px.histogram(
                df, x=x_col, color=color_col, nbins=30,
                title=f"Distribution de {x_col}"
            )

        elif chart_type == "Box Plot":
            fig = px.box(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Distribution de {y_col} par {x_col}"
            )

        elif chart_type == "Violin Plot":
            fig = px.violin(
                df, x=x_col, y=y_col, color=color_col, box=True,
                title=f"Distribution de {y_col} par {x_col}"
            )

        # Graphiques de proportion
        elif chart_type == "Camembert (Pie)":
            if y_col:
                df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                df_agg.columns = [x_col, "values"]
                names = x_col
                values = "values"
            else:
                df_agg = df[x_col].value_counts().reset_index()
                df_agg.columns = ["index", x_col]
                names = "index"
                values = x_col

            fig = px.pie(
                df_agg,
                names=names,
                values=values,
                title=f"Répartition de {y_col or x_col} par {x_col}"
            )

        elif chart_type == "Donut":
            if y_col:
                df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                df_agg.columns = [x_col, "values"]
                names = x_col
                values = "values"
            else:
                df_agg = df[x_col].value_counts().reset_index()
                df_agg.columns = ["index", x_col]
                names = "index"
                values = x_col

            fig = px.pie(
                df_agg,
                names=names,
                values=values,
                hole=0.4,
                title=f"Répartition de {y_col or x_col} par {x_col}"
            )

        elif chart_type == "Treemap":
            if y_col:
                df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                df_agg.columns = [x_col, "values"]
                path = [x_col]
                values = "values"
            else:
                df_agg = df[x_col].value_counts().reset_index()
                df_agg.columns = ["index", x_col]
                path = ["index"]
                values = x_col

            fig = px.treemap(
                df_agg,
                path=path,
                values=values,
                title=f"Treemap de {y_col or x_col}"
            )

        elif chart_type == "Sunburst":
            if color_col:
                if y_col:
                    df_agg = df.groupby([x_col, color_col])[y_col].sum().reset_index()
                    values = y_col
                else:
                    df_agg = df.groupby([x_col, color_col]).size().reset_index(name="count")
                    values = "count"

                fig = px.sunburst(
                    df_agg,
                    path=[x_col, color_col],
                    values=values,
                    title=f"Sunburst de {values}"
                )
            else:
                return None

        # Graphiques de relation
        elif chart_type == "Scatter":
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col, size=size_col,
                title=f"Relation entre {x_col} et {y_col}"
            )

        elif chart_type == "Scatter avec tendance":
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                trendline="ols",
                title=f"Relation entre {x_col} et {y_col} (avec tendance)"
            )

        elif chart_type == "Bubble Chart":
            if not size_col:
                st.warning("⚠️ Le Bubble Chart nécessite une colonne de taille")
                return None

            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col, size=size_col,
                title=f"Bubble chart : {x_col} vs {y_col}"
            )

        # Graphiques matriciels
        elif chart_type == "Heatmap (Matrice de corrélation)":
            numeric_df = df.select_dtypes(include=["number"])
            corr_matrix = numeric_df.corr()

            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Matrice de corrélation"
            )

        elif chart_type == "Heatmap personnalisée":
            if color_col:
                pivot_df = df.pivot_table(
                    values=y_col, index=x_col, columns=color_col, aggfunc="mean"
                )

                fig = px.imshow(
                    pivot_df,
                    text_auto=".2f",
                    aspect="auto",
                    title=f"Heatmap : {y_col} par {x_col} et {color_col}"
                )
            else:
                return None

        # Graphiques avancés
        elif chart_type == "Area Chart":
            fig = px.area(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Évolution cumulée de {y_col}"
            )

        elif chart_type == "Funnel":
            if y_col:
                df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                df_agg.columns = [x_col, "values"]
                x_val, y_val = "values", x_col
            else:
                df_agg = df[x_col].value_counts().reset_index()
                df_agg.columns = ["index", x_col]
                x_val, y_val = x_col, "index"

            fig = px.funnel(
                df_agg,
                x=x_val,
                y=y_val,
                title=f"Funnel de {y_col or x_col}"
            )

        elif chart_type == "Waterfall":
            if y_col:
                fig = go.Figure(
                    go.Waterfall(
                        x=df[x_col],
                        y=df[y_col],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    )
                )
                fig.update_layout(title=f"Waterfall : {y_col} par {x_col}")
            else:
                return None

        elif chart_type == "Gauge (Jauge)":
            if y_col and len(df) > 0:
                value = df[y_col].mean()

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=value,
                        title={"text": f"Moyenne de {y_col}"},
                        gauge={
                            "axis": {"range": [None, df[y_col].max()]},
                            "bar": {"color": "darkblue"},
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": df[y_col].quantile(0.75),
                            },
                        },
                    )
                )
            else:
                return None

        elif chart_type == "Parallel Categories":
            cat_cols = [x_col]
            if color_col:
                cat_cols.append(color_col)

            if len(cat_cols) >= 2:
                fig = px.parallel_categories(
                    df,
                    dimensions=cat_cols,
                    color=y_col if y_col else None,
                    title="Diagramme de Sankey catégoriel"
                )
            else:
                return None

        elif chart_type == "Parallel Coordinates":
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if len(numeric_cols) >= 2:
                fig = px.parallel_coordinates(
                    df,
                    dimensions=numeric_cols[:5],
                    color=y_col if y_col else numeric_cols[0],
                    title="Coordonnées parallèles"
                )
            else:
                return None

        elif chart_type == "Density Heatmap":
            fig = px.density_heatmap(
                df, x=x_col, y=y_col,
                title=f"Densité : {x_col} vs {y_col}"
            )

        elif chart_type == "Density Contour":
            fig = px.density_contour(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Contours de densité : {x_col} vs {y_col}"
            )

        elif chart_type == "Strip Plot":
            fig = px.strip(
                df, x=x_col, y=y_col, color=color_col,
                title=f"Strip plot : {y_col} par {x_col}"
            )

        elif chart_type == "ECDF":
            fig = px.ecdf(
                df, x=x_col, color=color_col,
                title=f"Fonction de répartition empirique de {x_col}"
            )

        else:
            return None

        # Configuration commune
        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="closest"
        )

        return fig

    except Exception as e:
        st.error(f"❌ Erreur lors de la création du graphique : {str(e)}")
        return None
