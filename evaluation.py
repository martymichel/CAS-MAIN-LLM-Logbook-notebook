#!/usr/bin/env python3
"""
Evaluierungsmodul für Logbook Analyzer
CAS Data Science - Text Analysis Projekt

Datengetriebene Ground Truth Erstellung basierend auf echten Logbuch-Einträgen.

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import re
import time
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class LogbookEvaluator:
    """
    Verbesserte Evaluierungsklasse mit datengetriebener Ground Truth.
    """
    
    def __init__(self, analyzer):
        """
        Initialisiert den verbesserten Evaluator.
        
        Args:
            analyzer: Konfigurierter LogbookAnalyzer
        """
        self.analyzer = analyzer
        self.df = analyzer.df
        self.subsystem_keywords = {}
        self.problem_patterns = {}
        self.temporal_patterns = {}
        
        # Analysiere Daten für bessere Ground Truth
        self._analyze_data_patterns()
    
    def _analyze_data_patterns(self):
        """
        Analysiert die echten Daten um Muster zu identifizieren.
        """
        if self.df is None:
            return
        
        print("Analysiere Datenmuster für Ground Truth Erstellung...")
        
        # 1. Subsystem-spezifische Keywords extrahieren
        self._extract_subsystem_keywords()
        
        # 2. Problem-Muster identifizieren
        self._identify_problem_patterns()
        
        # 3. Temporale Muster analysieren
        self._analyze_temporal_patterns()
        
        print(f"Datenanalyse abgeschlossen:")
        print(f"  - {len(self.subsystem_keywords)} Subsystem-Keyword-Sets")
        print(f"  - {len(self.problem_patterns)} Problem-Muster")
        print(f"  - Temporale Muster für {len(self.temporal_patterns)} Zeiträume")
    
    def _extract_subsystem_keywords(self):
        """
        Extrahiert charakteristische Keywords für jedes Subsystem.
        """
        subsystem_texts = {}
        
        # Sammle alle Texte pro Subsystem
        for subsystem in self.df['Subsystem'].unique():
            if pd.isna(subsystem) or subsystem == 'N/A':
                continue
            
            subsystem_entries = self.df[self.df['Subsystem'] == subsystem]
            combined_text = ' '.join(subsystem_entries['Ereignis & Massnahme'].astype(str))
            subsystem_texts[subsystem] = combined_text
        
        # TF-IDF für charakteristische Begriffe
        if len(subsystem_texts) > 1:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=2
            )
            
            subsystems = list(subsystem_texts.keys())
            texts = list(subsystem_texts.values())
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Top Keywords pro Subsystem
                for i, subsystem in enumerate(subsystems):
                    scores = tfidf_matrix[i].toarray()[0]
                    top_indices = scores.argsort()[-15:][::-1]  # Top 15
                    
                    keywords = []
                    for idx in top_indices:
                        if scores[idx] > 0.1:  # Mindest-Score
                            keywords.append(feature_names[idx])
                    
                    self.subsystem_keywords[subsystem] = keywords[:10]  # Top 10
                    
            except Exception as e:
                logger.warning(f"TF-IDF Analyse fehlgeschlagen: {e}")
                # Fallback: Häufigste Wörter
                self._extract_keywords_fallback(subsystem_texts)
    
    def _extract_keywords_fallback(self, subsystem_texts):
        """
        Fallback-Methode für Keyword-Extraktion.
        """
        for subsystem, text in subsystem_texts.items():
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter häufige deutsche Wörter
            stop_words = {
                'und', 'der', 'die', 'das', 'ist', 'mit', 'von', 'auf', 'für', 
                'zu', 'in', 'bei', 'nach', 'vor', 'werden', 'wurde', 'wird'
            }
            
            filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
            word_counts = Counter(filtered_words)
            
            # Top 10 häufigste Wörter
            top_words = [word for word, count in word_counts.most_common(10)]
            self.subsystem_keywords[subsystem] = top_words
    
    def _identify_problem_patterns(self):
        """
        Identifiziert Problem-Muster aus den echten Daten.
        """
        # Definiere Problem-Indikatoren
        problem_indicators = [
            'problem', 'fehler', 'defekt', 'ausfall', 'störung', 'reparatur',
            'wartung', 'instandhaltung', 'service', 'austausch', 'erneuern',
            'reinigen', 'einstellen', 'justieren', 'kalibrieren',
            'undicht', 'blockiert', 'verstopft', 'verschleiß', 'abnutzung',
            'überhitzung', 'unterkühlung', 'druckabfall', 'leckage'
        ]
        
        # Analysiere welche Indikatoren tatsächlich vorkommen
        text_column = self.df['Ereignis & Massnahme'].astype(str).str.lower()
        
        for indicator in problem_indicators:
            matches = text_column.str.contains(indicator, na=False)
            match_count = matches.sum()
            
            if match_count >= 5:  # Mindestens 5 Vorkommen
                # Sammle Indizes der Einträge mit diesem Problem
                matching_indices = self.df[matches].index.tolist()
                self.problem_patterns[indicator] = {
                    'count': match_count,
                    'indices': matching_indices,
                    'percentage': (match_count / len(self.df)) * 100
                }
        
        # Sortiere nach Häufigkeit
        self.problem_patterns = dict(
            sorted(self.problem_patterns.items(), 
                  key=lambda x: x[1]['count'], reverse=True)
        )
    
    def _analyze_temporal_patterns(self):
        """
        Analysiert zeitliche Muster in den Daten.
        """
        if 'Datum' not in self.df.columns:
            return
        
        # Versuche Datum zu parsen
        try:
            dates = pd.to_datetime(self.df['Datum'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                return
            
            # Gruppiere nach Monaten
            monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
            
            # Finde aktivste Zeiträume
            top_months = monthly_counts.nlargest(5)
            
            for month, count in top_months.items():
                month_data = self.df[dates.dt.to_period('M') == month]
                self.temporal_patterns[str(month)] = {
                    'count': count,
                    'indices': month_data.index.tolist(),
                    'percentage': (count / len(self.df)) * 100
                }
                
        except Exception as e:
            logger.warning(f"Temporale Analyse fehlgeschlagen: {e}")
    
    def create_data_driven_ground_truth(self) -> Dict[str, Set[int]]:
        """
        Erstellt datengetriebene Ground Truth basierend auf echten Mustern.
        
        Returns:
            Dictionary: Query -> Set relevanter Indizes
        """
        ground_truth = {}
        
        print("\nErstelle datengetriebene Ground Truth...")
        print("-" * 50)
        
        # 1. Subsystem-basierte Queries
        print("1. Subsystem-spezifische Queries:")
        for subsystem, keywords in self.subsystem_keywords.items():
            if len(keywords) < 3:
                continue
            
            # Erstelle Query aus Top-Keywords
            query = f"Probleme {subsystem} {' '.join(keywords[:3])}"
            
            # Finde relevante Einträge
            relevant_indices = set()
            
            # Direkte Subsystem-Matches
            subsystem_matches = self.df[self.df['Subsystem'] == subsystem].index
            relevant_indices.update(subsystem_matches)
            
            # Keyword-basierte Matches
            text_lower = self.df['Ereignis & Massnahme'].astype(str).str.lower()
            for keyword in keywords[:5]:  # Top 5 Keywords
                keyword_matches = self.df[text_lower.str.contains(keyword, na=False)].index
                relevant_indices.update(keyword_matches)
            
            if len(relevant_indices) >= 10:  # Mindestens 10 relevante Einträge
                ground_truth[query] = relevant_indices
                print(f"   '{subsystem}': {len(relevant_indices)} relevante Einträge")
        
        # 2. Problem-basierte Queries
        print("\n2. Problem-spezifische Queries:")
        for problem, data in list(self.problem_patterns.items())[:8]:  # Top 8 Probleme
            if data['count'] < 10:
                continue
            
            query = f"Alle {problem} in der Anlage"
            relevant_indices = set(data['indices'])
            
            # Erweitere um ähnliche Probleme
            similar_problems = self._find_similar_problems(problem)
            for similar_problem in similar_problems:
                if similar_problem in self.problem_patterns:
                    relevant_indices.update(self.problem_patterns[similar_problem]['indices'])
            
            ground_truth[query] = relevant_indices
            print(f"   '{problem}': {len(relevant_indices)} relevante Einträge")
        
        # 3. Kombinierte Subsystem-Problem Queries
        print("\n3. Kombinierte Queries:")
        for subsystem in list(self.subsystem_keywords.keys())[:5]:  # Top 5 Subsysteme
            for problem in list(self.problem_patterns.keys())[:3]:  # Top 3 Probleme
                query = f"{problem} im {subsystem}"
                
                # Finde Einträge die beide Kriterien erfüllen
                subsystem_entries = self.df[self.df['Subsystem'] == subsystem]
                text_lower = subsystem_entries['Ereignis & Massnahme'].astype(str).str.lower()
                problem_matches = subsystem_entries[text_lower.str.contains(problem, na=False)]
                
                if len(problem_matches) >= 5:  # Mindestens 5 Einträge
                    relevant_indices = set(problem_matches.index)
                    ground_truth[query] = relevant_indices
                    print(f"   '{problem} im {subsystem}': {len(relevant_indices)} relevante Einträge")
        
        # 4. Lot-spezifische Queries (falls Lot-Nummern vorhanden)
        if 'Lot-Nr.' in self.df.columns:
            print("\n4. Lot-spezifische Queries:")
            lot_counts = self.df['Lot-Nr.'].value_counts()
            top_lots = lot_counts.head(5)  # Top 5 häufigste Lots
            
            for lot_nr, count in top_lots.items():
                if count >= 5 and not pd.isna(lot_nr):
                    query = f"Alle Ereignisse Lot {lot_nr}"
                    lot_entries = self.df[self.df['Lot-Nr.'] == lot_nr]
                    relevant_indices = set(lot_entries.index)
                    ground_truth[query] = relevant_indices
                    print(f"   'Lot {lot_nr}': {len(relevant_indices)} relevante Einträge")
        
        # 5. Zeitbasierte Queries
        if self.temporal_patterns:
            print("\n5. Zeitbasierte Queries:")
            for period, data in list(self.temporal_patterns.items())[:3]:  # Top 3 Zeiträume
                query = f"Alle Ereignisse im {period}"
                relevant_indices = set(data['indices'])
                ground_truth[query] = relevant_indices
                print(f"   '{period}': {len(relevant_indices)} relevante Einträge")
        
        print(f"\nGround Truth erstellt: {len(ground_truth)} Queries")
        return ground_truth
    
    def _find_similar_problems(self, problem: str) -> List[str]:
        """
        Findet ähnliche Problem-Begriffe.
        """
        similarity_groups = {
            'fehler': ['defekt', 'problem', 'störung'],
            'wartung': ['service', 'instandhaltung', 'reparatur'],
            'ausfall': ['störung', 'defekt', 'problem'],
            'undicht': ['leckage', 'tropfen'],
            'überhitzung': ['temperatur', 'heiß'],
            'verstopft': ['blockiert', 'verstopfung']
        }
        
        for key, similar_list in similarity_groups.items():
            if problem in similar_list or problem == key:
                return [p for p in similar_list if p != problem and p in self.problem_patterns]
        
        return []
    
    def calculate_precision_recall_improved(self, query: str, expected_relevant: Set[int], top_k: int = 10) -> Dict:
        """
        Verbesserte Precision/Recall Berechnung mit detaillierten Metriken.
        """
        # Suche durchführen
        indices, scores, _ = self.analyzer.semantic_search(query, max_results=50)
        
        if not indices:
            return {
                "precision": 0, "recall": 0, "f1": 0, "retrieved_count": 0,
                "precision_at_5": 0, "precision_at_15": 0, "ndcg": 0
            }
        
        # Verschiedene K-Werte für Precision@K
        results = {}
        for k in [5, 10, 15, 20]:
            if k <= len(indices):
                top_k_indices = set(indices[:k])
                true_positives = len(top_k_indices.intersection(expected_relevant))
                precision_at_k = true_positives / k
                results[f"precision_at_{k}"] = precision_at_k
        
        # Standard Metriken für k=10
        top_k_indices = set(indices[:top_k])
        true_positives = len(top_k_indices.intersection(expected_relevant))
        false_positives = len(top_k_indices - expected_relevant)
        false_negatives = len(expected_relevant - top_k_indices)
        
        precision = true_positives / len(top_k_indices) if top_k_indices else 0
        recall = true_positives / len(expected_relevant) if expected_relevant else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # NDCG berechnen
        ndcg = self._calculate_ndcg(indices, expected_relevant, scores, k=top_k)
        
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "retrieved_count": len(top_k_indices),
            "relevant_count": len(expected_relevant),
            "ndcg": ndcg
        })
        
        return results
    
    def _calculate_ndcg(self, indices: List[int], relevant_set: Set[int], scores: List[float], k: int = 10) -> float:
        """
        Berechnet Normalized Discounted Cumulative Gain.
        """
        if not indices or not relevant_set:
            return 0.0
        
        # DCG berechnen
        dcg = 0.0
        for i, idx in enumerate(indices[:k]):
            if idx in relevant_set:
                # Relevanz = 1 für relevante Dokumente
                relevance = 1
                dcg += relevance / np.log2(i + 2)  # i+2 weil Log2(1) = 0
        
        # IDCG berechnen (perfekte Rangfolge)
        num_relevant_in_top_k = min(len(relevant_set), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(num_relevant_in_top_k))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def evaluate_with_improved_ground_truth(self) -> Dict:
        """
        Evaluiert das System mit verbesserter Ground Truth.
        """
        print("EVALUIERUNG MIT DATENGETRIEBENER GROUND TRUTH")
        print("=" * 65)
        
        # Erstelle verbesserte Ground Truth
        ground_truth = self.create_data_driven_ground_truth()
        
        if not ground_truth:
            return {"error": "Keine Ground Truth erstellt"}
        
        results = {}
        all_metrics = {
            'precision': [], 'recall': [], 'f1': [], 'ndcg': [],
            'precision_at_5': [], 'precision_at_15': []
        }
        
        print(f"\nEvaluiere {len(ground_truth)} datengetriebene Queries:")
        print("-" * 60)
        
        for query, expected_relevant in ground_truth.items():
            if len(expected_relevant) < 5:  # Skip zu kleine Sets
                continue
            
            metrics = self.calculate_precision_recall_improved(query, expected_relevant, top_k=10)
            results[query] = metrics
            
            # Sammle Metriken
            for metric in all_metrics:
                if metric in metrics:
                    all_metrics[metric].append(metrics[metric])
            
            print(f"'{query[:50]}...':")
            print(f"  P@5: {metrics.get('precision_at_5', 0):.3f} | P@10: {metrics['precision']:.3f} | P@15: {metrics.get('precision_at_15', 0):.3f}")
            print(f"  Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | NDCG: {metrics['ndcg']:.3f}")
            print(f"  Relevante: {metrics['true_positives']}/{metrics['relevant_count']}")
            print()
        
        # Berechne aggregierte Metriken
        aggregate_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                aggregate_metrics[f'mean_{metric}'] = np.mean(values)
                aggregate_metrics[f'std_{metric}'] = np.std(values)
                aggregate_metrics[f'median_{metric}'] = np.median(values)
        
        aggregate_metrics['queries_evaluated'] = len(results)
        aggregate_metrics['total_queries_created'] = len(ground_truth)
        
        print("AGGREGIERTE ERGEBNISSE:")
        print("-" * 30)
        print(f"Evaluierte Queries: {aggregate_metrics['queries_evaluated']}")
        print(f"Mean Precision@5:  {aggregate_metrics.get('mean_precision_at_5', 0):.3f} ± {aggregate_metrics.get('std_precision_at_5', 0):.3f}")
        print(f"Mean Precision@10: {aggregate_metrics.get('mean_precision', 0):.3f} ± {aggregate_metrics.get('std_precision', 0):.3f}")
        print(f"Mean Recall@10:    {aggregate_metrics.get('mean_recall', 0):.3f} ± {aggregate_metrics.get('std_recall', 0):.3f}")
        print(f"Mean F1-Score:     {aggregate_metrics.get('mean_f1', 0):.3f} ± {aggregate_metrics.get('std_f1', 0):.3f}")
        print(f"Mean NDCG@10:      {aggregate_metrics.get('mean_ndcg', 0):.3f} ± {aggregate_metrics.get('std_ndcg', 0):.3f}")
        
        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "ground_truth_stats": {query: len(relevant) for query, relevant in ground_truth.items()},
            "data_analysis": {
                "subsystems_analyzed": len(self.subsystem_keywords),
                "problem_patterns_found": len(self.problem_patterns),
                "temporal_patterns": len(self.temporal_patterns)
            }
        }


def run_improved_evaluation(analyzer) -> Dict:
    """
    Führt verbesserte Evaluierung mit datengetriebener Ground Truth durch.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        
    Returns:
        Umfassende Evaluierungsergebnisse
    """
    evaluator = LogbookEvaluator(analyzer)
    
    # Hauptevaluierung
    improved_results = evaluator.evaluate_with_improved_ground_truth()
    
    # Zusätzliche Analysen
    if "error" not in improved_results:
        print(f"\nDATENANALYSE-ZUSAMMENFASSUNG:")
        data_analysis = improved_results["data_analysis"]
        print(f"  Analysierte Subsysteme: {data_analysis['subsystems_analyzed']}")
        print(f"  Identifizierte Problem-Muster: {data_analysis['problem_patterns_found']}")
        print(f"  Temporale Muster: {data_analysis['temporal_patterns']}")
        
        # Qualitätsbewertung
        agg = improved_results["aggregate_metrics"]
        overall_score = (
            agg.get('mean_precision', 0) * 0.3 +
            agg.get('mean_recall', 0) * 0.3 +
            agg.get('mean_f1', 0) * 0.4
        )
        
        improved_results["overall_evaluation"] = {
            "overall_score": overall_score,
            "recommendation": "Sehr gut" if overall_score > 0.8 else "Gut" if overall_score > 0.6 else "Befriedigend" if overall_score > 0.4 else "Verbesserungsbedarf",
            "data_driven": True,
            "ground_truth_quality": "Hoch - basiert auf echten Datenmustern"
        }
    
    return improved_results