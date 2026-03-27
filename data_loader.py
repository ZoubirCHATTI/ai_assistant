# data_loader.py
# -*- coding: utf-8 -*-
"""
Module de chargement des données TER depuis l'API SNCF
"""

import requests
import pandas as pd
from typing import Optional


class TERDataLoader:
    """Classe pour charger et préparer les données TER"""
    
    def __init__(self):
        """Initialise le loader avec l'URL de l'API SNCF"""
        self.base_url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-ter/records"
    
    def load_data(self, max_records: int = None) -> pd.DataFrame:
        """
        Charge les données TER depuis l'API SNCF
        
        Args:
            max_records: Nombre maximum d'enregistrements à charger (None = tous)
            
        Returns:
            DataFrame avec les données TER et le taux de régularité calculé
        """
        params = {
            'limit': 100,
            'offset': 0
        }
        
        all_records = []
        
        print("📥 Téléchargement des données TER depuis l'API SNCF...")
        
        while True:
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                records = data.get('results', [])
                
                if not records:
                    print("   ✅ Fin du téléchargement (pas de nouveaux enregistrements)")
                    break
                
                all_records.extend(records)
                print(f"   → {len(all_records)} enregistrements téléchargés...")
                
                # Vérifier si on a atteint le maximum demandé
                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    print(f"   ✅ Limite de {max_records} enregistrements atteinte")
                    break
                
                # Si on a reçu moins que la limite, c'est qu'on a tout
                if len(records) < params['limit']:
                    print("   ✅ Tous les enregistrements ont été téléchargés")
                    break
                
                # Passer au lot suivant
                params['offset'] += params['limit']
                
            except requests.exceptions.Timeout:
                print("   ⚠️ Timeout - Réessai...")
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Erreur lors du téléchargement : {e}")
                break
        
        if not all_records:
            raise ValueError("❌ Aucune donnée téléchargée depuis l'API")
        
        print(f"\n✅ Total téléchargé : {len(all_records)} enregistrements")
        
        # Créer le DataFrame
        df = pd.DataFrame(all_records)
        
        print(f"📊 DataFrame créé : {len(df)} lignes × {len(df.columns)} colonnes")
        
        # Nettoyage de base
        df = self._clean_data(df)
        
        # Calculer le taux de régularité
        df = self.calculate_regularite(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et prépare les données
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame nettoyé
        """
        print("\n🧹 Nettoyage des données...")
        
        df = df.copy()
        
        # Afficher les colonnes disponibles
        print(f"   Colonnes présentes : {', '.join(df.columns.tolist()[:15])}...")
        
        # Convertir les dates
        date_columns = ['date', 'mois', 'annee']
        for col in date_columns:
            if col in df.columns:
                if col == 'date':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"   ✅ Colonne '{col}' convertie en datetime")
        
        # Convertir les colonnes numériques
        numeric_columns = [
            'nombre_trains_prevus', 
            'nombre_trains_circules',
            'nombre_trains_supprimes', 
            'nombre_trains_annules',
            'nombre_trains_retard',
            'nombre_trains_a_l_heure',
            'taux_regularite'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"   ✅ Colonne '{col}' convertie en numérique")
        
        # Supprimer les lignes avec dates invalides si la colonne date existe
        if 'date' in df.columns:
            avant = len(df)
            df = df.dropna(subset=['date'])
            apres = len(df)
            if avant != apres:
                print(f"   ⚠️ {avant - apres} lignes supprimées (dates invalides)")
        
        print(f"   ✅ Nettoyage terminé : {len(df)} lignes conservées")
        
        return df
    
    def calculate_regularite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le taux de régularité à partir des colonnes disponibles
        
        Le taux de régularité représente le pourcentage de trains à l'heure
        
        Args:
            df: DataFrame TER
            
        Returns:
            DataFrame avec la colonne taux_regularite ajoutée ou mise à jour
        """
        print("\n📊 Calcul du taux de régularité...")
        
        df = df.copy()
        
        # Vérifier si la colonne existe déjà
        if 'taux_regularite' in df.columns:
            # Vérifier si elle contient des valeurs valides
            valeurs_valides = df['taux_regularite'].notna().sum()
            
            if valeurs_valides > len(df) * 0.8:  # Si plus de 80% des valeurs sont présentes
                print(f"   ✅ Colonne 'taux_regularite' déjà présente ({valeurs_valides}/{len(df)} valeurs)")
                avg = df['taux_regularite'].mean()
                print(f"   📊 Taux de régularité moyen : {avg:.2f}%")
                return df
            else:
                print(f"   ⚠️ Colonne 'taux_regularite' présente mais incomplète ({valeurs_valides}/{len(df)})")
        
        # Méthode 1 : Si on a "nombre_trains_a_l_heure"
        if 'nombre_trains_a_l_heure' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                df['nombre_trains_a_l_heure'] / df['nombre_trains_circules'] * 100
            ).fillna(0)
            print("   ✅ Taux calculé à partir de 'nombre_trains_a_l_heure' / 'nombre_trains_circules'")
        
        # Méthode 2 : Si on a "nombre_trains_retard"
        elif 'nombre_trains_retard' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                100 - (df['nombre_trains_retard'] / df['nombre_trains_circules'] * 100)
            ).fillna(0)
            print("   ✅ Taux calculé à partir de 'nombre_trains_retard' / 'nombre_trains_circules'")
        
        # Méthode 3 : Chercher une colonne avec "regularite" ou "ponctualite"
        else:
            regularite_cols = [
                col for col in df.columns 
                if 'regularite' in col.lower() or 'ponctualite' in col.lower()
            ]
            
            if regularite_cols:
                df['taux_regularite'] = pd.to_numeric(df[regularite_cols[0]], errors='coerce')
                print(f"   ✅ Colonne '{regularite_cols[0]}' utilisée comme taux_regularite")
            else:
                print("   ⚠️ ATTENTION : Impossible de calculer le taux de régularité automatiquement")
                print("   Colonnes disponibles dans le dataset :")
                for col in df.columns:
                    print(f"      - {col}")
                print("   → Création d'une colonne avec valeur par défaut 0")
                df['taux_regularite'] = 0
        
        # S'assurer que les valeurs sont entre 0 et 100
        df['taux_regularite'] = df['taux_regularite'].clip(0, 100)
        
        # Calculer et afficher les statistiques
        avg = df['taux_regularite'].mean()
        median = df['taux_regularite'].median()
        mini = df['taux_regularite'].min()
        maxi = df['taux_regularite'].max()
        
        print(f"\n   📈 Statistiques du taux de régularité :")
        print(f"      - Moyenne  : {avg:.2f}%")
        print(f"      - Médiane  : {median:.2f}%")
        print(f"      - Minimum  : {mini:.2f}%")
        print(f"      - Maximum  : {maxi:.2f}%")
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Retourne des informations sur le DataFrame
        
        Args:
            df: DataFrame TER
            
        Returns:
            Dictionnaire avec les informations
        """
        info = {
            'nb_lignes': len(df),
            'nb_colonnes': len(df.columns),
            'colonnes': df.columns.tolist(),
            'has_regularite': 'taux_regularite' in df.columns,
            'has_region': 'region' in df.columns,
            'has_date': 'date' in df.columns
        }
        
        if 'date' in df.columns:
            info['date_min'] = df['date'].min()
            info['date_max'] = df['date'].max()
        
        if 'region' in df.columns:
            info['nb_regions'] = df['region'].nunique()
            info['regions'] = df['region'].unique().tolist()
        
        if 'taux_regularite' in df.columns:
            info['regularite_moyenne'] = df['taux_regularite'].mean()
            info['regularite_min'] = df['taux_regularite'].min()
            info['regularite_max'] = df['taux_regularite'].max()
        
        return info


# ═══════════════════════════════════════════════════════════════════════
# TEST DU MODULE (si exécuté directement)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test du module data_loader.py\n")
    
    try:
        loader = TERDataLoader()
        df = loader.load_data(max_records=500)  # Limiter pour le test
        
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DU CHARGEMENT")
        print("="*70)
        
        info = loader.get_data_info(df)
        
        print(f"\n✅ Nombre de lignes : {info['nb_lignes']:,}")
        print(f"✅ Nombre de colonnes : {info['nb_colonnes']}")
        
        if info['has_regularite']:
            print(f"\n📊 Taux de régularité :")
            print(f"   - Moyenne : {info['regularite_moyenne']:.2f}%")
            print(f"   - Min     : {info['regularite_min']:.2f}%")
            print(f"   - Max     : {info['regularite_max']:.2f}%")
        
        if info['has_region']:
            print(f"\n🗺️ Régions : {info['nb_regions']} régions uniques")
            print(f"   Exemples : {', '.join(info['regions'][:5])}")
        
        if info['has_date']:
            print(f"\n📅 Période couverte :")
            print(f"   - Du {info['date_min']} au {info['date_max']}")
        
        print("\n" + "="*70)
        print("✅ TEST RÉUSSI")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
