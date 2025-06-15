#!/usr/bin/env python3
"""
Utilities Modul für Logbook Analyzer - VOLLSTÄNDIGE VERSION
CAS Data Science - Text Analysis Projekt

Hilfsfunktionen für Visualisierung, Interaktion und Datenexport.

Autor: [Ihr Name]
Datum: Juni 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

def create_evaluation_visualizations(analyzer, evaluation_results: Dict, stats: Dict):
    """
    Erstellt umfassende Visualisierungen für die Evaluierung.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        evaluation_results: Ergebnisse der Evaluierung
        stats: Datenstatistiken
    """
    # Setup für 2x2 Subplot-Grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Logbuch Analyzer - Evaluierungs-Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Subsystem-Verteilung (Top 8)
    if 'subsystem_distribution' in stats and stats['subsystem_distribution']:
        subsystems = list(stats['subsystem_distribution'].keys())[:8]
        counts = list(stats['subsystem_distribution'].values())[:8]
        
        bars = axes[0, 0].bar(range(len(subsystems)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Top 8 Subsysteme', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(range(len(subsystems)))
        axes[0, 0].set_xticklabels(subsystems, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Anzahl Einträge')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Werte auf Balken anzeigen
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                           f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 2. Performance vs Qualität Scatter
    if "performance" in evaluation_results:
        perf_data = evaluation_results["performance"]["individual_results"]
        search_times = [r["search_time"] for r in perf_data]
        quality_scores = [r["mean_score"] for r in perf_data]
        
        scatter = axes[0, 1].scatter(search_times, quality_scores, 
                                   alpha=0.7, s=60, c='orange', edgecolors='red')
        axes[0, 1].set_xlabel('Suchzeit (Sekunden)')
        axes[0, 1].set_ylabel('Qualitätsscore')
        axes[0, 1].set_title('Performance vs Qualität', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Durchschnittslinie hinzufügen
        avg_time = np.mean(search_times)
        avg_quality = np.mean(quality_scores)
        axes[0, 1].axvline(avg_time, color='red', linestyle='--', alpha=0.5, label=f'Avg Zeit: {avg_time:.3f}s')
        axes[0, 1].axhline(avg_quality, color='red', linestyle='--', alpha=0.5, label=f'Avg Qualität: {avg_quality:.3f}')
        axes[0, 1].legend()
    
    # 3. Relevanz-Score Verteilung für Beispiel-Query
    test_query = "Probleme mit der Produktion"
    indices, scores, _ = analyzer.semantic_search(test_query, max_results=50)
    
    if scores:
        axes[1, 0].hist(scores, bins=15, alpha=0.7, color='lightgreen', 
                       edgecolor='darkgreen', density=True)
        axes[1, 0].set_xlabel('Relevanz-Score')
        axes[1, 0].set_ylabel('Dichte')
        axes[1, 0].set_title(f'Score-Verteilung\n"{test_query[:30]}..."', fontsize=12, fontweight='bold')
        
        # Statistiken hinzufügen
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        axes[1, 0].axvline(mean_score, color='red', linestyle='-', 
                          label=f'μ = {mean_score:.3f}')
        axes[1, 0].axvline(mean_score + std_score, color='red', linestyle='--', alpha=0.7,
                          label=f'μ + σ = {mean_score + std_score:.3f}')
        axes[1, 0].axvline(mean_score - std_score, color='red', linestyle='--', alpha=0.7,
                          label=f'μ - σ = {mean_score - std_score:.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Evaluierungs-Metriken Balkendiagramm
    if evaluation_results:
        metrics_names = []
        metrics_values = []
        colors = []
        
        # Regel-basierte Metriken
        if "rule_based" in evaluation_results and "aggregate_metrics" in evaluation_results["rule_based"]:
            rb = evaluation_results["rule_based"]["aggregate_metrics"]
            metrics_names.extend(['Precision@10', 'Recall@10', 'F1-Score'])
            metrics_values.extend([rb['mean_precision'], rb['mean_recall'], rb['mean_f1']])
            colors.extend(['lightblue', 'lightgreen', 'lightcoral'])
        
        # Gesamtscore
        if "overall_evaluation" in evaluation_results:
            overall_score = evaluation_results["overall_evaluation"]["overall_score"]
            metrics_names.append('Gesamtscore')
            metrics_values.append(overall_score)
            colors.append('gold')
        
        if metrics_names:
            bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, 
                                 edgecolor='black', alpha=0.8)
            axes[1, 1].set_title('Evaluierungs-Metriken', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Score (0-1)')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            # Werte auf Balken anzeigen
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_query_comparison_chart(analyzer, queries: List[str]):
    """
    Erstellt Vergleichsdiagramm für verschiedene Queries.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        queries: Liste von Queries zum Vergleichen
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Query-Vergleichsanalyse', fontsize=14, fontweight='bold')
    
    # Daten sammeln
    query_data = []
    for query in queries[:6]:  # Maximal 6 Queries für Übersichtlichkeit
        indices, scores, count = analyzer.semantic_search(query)
        
        query_data.append({
            'query': query[:25] + "..." if len(query) > 25 else query,
            'result_count': count,
            'mean_score': np.mean(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'high_relevance': sum(1 for s in scores if s >= 0.7) if scores else 0
        })
    
    # Chart 1: Ergebnisanzahl und Durchschnittsscore
    queries_short = [d['query'] for d in query_data]
    result_counts = [d['result_count'] for d in query_data]
    mean_scores = [d['mean_score'] for d in query_data]
    
    x = np.arange(len(queries_short))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, result_counts, width, label='Anzahl Ergebnisse', 
                   color='lightblue', alpha=0.8)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, mean_scores, width, label='Durchschn. Score', 
                        color='orange', alpha=0.8)
    
    ax1.set_xlabel('Queries')
    ax1.set_ylabel('Anzahl Ergebnisse', color='blue')
    ax1_twin.set_ylabel('Durchschnittsscore', color='orange')
    ax1.set_title('Ergebnisanzahl vs Durchschnittsscore')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries_short, rotation=45, ha='right')
    
    # Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Chart 2: Hochrelevante Ergebnisse
    high_relevance = [d['high_relevance'] for d in query_data]
    
    bars = ax2.bar(queries_short, high_relevance, color='green', alpha=0.7)
    ax2.set_xlabel('Queries')
    ax2.set_ylabel('Anzahl hochrelevanter Ergebnisse (Score ≥ 0.7)')
    ax2.set_title('Hochrelevante Ergebnisse pro Query')
    ax2.set_xticklabels(queries_short, rotation=45, ha='right')
    
    # Werte auf Balken
    for bar, value in zip(bars, high_relevance):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def interactive_query_demo(analyzer):
    """
    Interaktive Demo für eigene Suchanfragen.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
    """
    if analyzer is None:
        print("Analyzer nicht verfügbar für interaktive Demo.")
        return
    
    print("INTERAKTIVE SUCHDEMO")
    print("=" * 40)
    print("Geben Sie eigene Suchanfragen ein!")
    print("Tipps:")
    print("- 'Probleme mit Spritzgussmaschine'")
    print("- 'Qualitätsfehler letzte Woche'")
    print("- 'Wartung Kühlsystem'")
    print("(Leere Eingabe beendet die Demo)")
    print("-" * 40)
    
    query_counter = 0
    
    while True:
        query_counter += 1
        user_query = input(f"\n[{query_counter}] Ihre Suchanfrage: ").strip()
        
        if not user_query:
            print("Demo beendet.")
            break
        
        print(f"Analysiere: '{user_query}'...")
        
        # Zeitmessung
        start_time = time.time()
        results = analyzer.analyze_query(user_query)
        search_time = time.time() - start_time
        
        if "error" not in results:
            print(f"\nERGEBNISSE (in {search_time:.3f}s):")
            print(f"  Gefunden: {results['result_count']} Einträge")
            print(f"  Qualität: {results['search_quality']} (Durchschn.: {results['average_relevance']:.3f})")
            
            # Top 3 Ergebnisse detailliert anzeigen
            print(f"\nTOP 3 RELEVANTE EINTRÄGE:")
            for i, (_, row) in enumerate(results["relevant_entries"].head(3).iterrows(), 1):
                score = row.get('similarity_score', 0)
                subsystem = row.get('Subsystem', 'N/A')
                ereignis = str(row.get('Ereignis & Massnahme', 'N/A'))
                datum = row.get('Datum', 'N/A')
                zeit = row.get('Zeit', 'N/A')
                
                print(f"\n{i}. RELEVANZ: {score:.3f} | SYSTEM: {subsystem}")
                print(f"   ZEIT: {datum} {zeit}")
                print(f"   EREIGNIS: {ereignis[:100]}{'...' if len(ereignis) > 100 else ''}")
            
            # Kurze KI-Analyse falls verfügbar
            if "llm_analysis" in results and results["llm_analysis"]:
                analysis = results["llm_analysis"]
                if len(analysis) > 200:
                    analysis = analysis[:200] + "..."
                print(f"\nKI-KURZANALYSE:\n{analysis}")
        else:
            print(f"FEHLER: {results['error']}")

def analyze_specific_subsystem(analyzer, subsystem_name: str):
    """
    Analysiert spezifisches Subsystem detailliert.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        subsystem_name: Name des zu analysierenden Subsystems
    """
    if analyzer is None or analyzer.df is None:
        print("Analyzer oder Daten nicht verfügbar.")
        return
    
    print(f"\nDETAILANALYSE SUBSYSTEM: {subsystem_name}")
    print("=" * 50)
    
    # Filter nach Subsystem (case-insensitive)
    subsystem_entries = analyzer.df[
        analyzer.df['Subsystem'].str.contains(subsystem_name, case=False, na=False)
    ]
    
    if len(subsystem_entries) == 0:
        print(f"Keine Einträge für Subsystem '{subsystem_name}' gefunden.")
        
        # Ähnliche Subsysteme vorschlagen
        available_subsystems = analyzer.df['Subsystem'].value_counts().head(10)
        print(f"\nVerfügbare Subsysteme:")
        for subsys, count in available_subsystems.items():
            print(f"  - {subsys} ({count} Einträge)")
        return
    
    print(f"Einträge gefunden: {len(subsystem_entries)}")
    
    # Zeitraum-Analyse
    if 'Datum_Parsed' in subsystem_entries.columns:
        valid_dates = subsystem_entries['Datum_Parsed'].dropna()
        if len(valid_dates) > 0:
            print(f"Zeitraum: {valid_dates.min().date()} bis {valid_dates.max().date()}")
            
            # Einträge pro Monat
            monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
            if len(monthly_counts) > 1:
                print(f"Aktivste Monate:")
                for month, count in monthly_counts.head(3).items():
                    print(f"  {month}: {count} Einträge")
    
    # Häufigste Begriffe in Ereignissen
    print(f"\nHÄUFIGSTE BEGRIFFE:")
    all_text = ' '.join(subsystem_entries['Ereignis & Massnahme'].astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Filter häufige deutsche Wörter
    stop_words = {'und', 'der', 'die', 'das', 'ist', 'mit', 'von', 'auf', 'für', 'zu', 'in', 'bei', 'nach', 'vor'}
    word_freq = {}
    for word in words:
        if len(word) > 3 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, count in top_words:
        print(f"  {word}: {count}x")
    
    # Problem-Indikatoren
    problem_indicators = ['problem', 'fehler', 'defekt', 'ausfall', 'störung', 'reparatur']
    problem_counts = {}
    for indicator in problem_indicators:
        count = all_text.lower().count(indicator)
        if count > 0:
            problem_counts[indicator] = count
    
    if problem_counts:
        print(f"\nPROBLEM-INDIKATOREN:")
        for indicator, count in sorted(problem_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {indicator}: {count}x")
    
    # Neueste Einträge
    print(f"\nNEUESTE 3 EINTRÄGE:")
    if 'Datum_Parsed' in subsystem_entries.columns:
        latest_entries = subsystem_entries.sort_values('Datum_Parsed', ascending=False).head(3)
    else:
        latest_entries = subsystem_entries.head(3)
    
    for i, (_, entry) in enumerate(latest_entries.iterrows(), 1):
        datum = entry.get('Datum', 'N/A')
        zeit = entry.get('Zeit', 'N/A')
        ereignis = str(entry.get('Ereignis & Massnahme', 'N/A'))[:80]
        print(f"{i}. {datum} {zeit}: {ereignis}...")

def export_analysis_results(analyzer, query: str, filename: str = None) -> Optional[pd.DataFrame]:
    """
    Exportiert Analyseergebnisse als DataFrame.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        query: Suchanfrage
        filename: Dateiname für Export (optional)
        
    Returns:
        DataFrame mit Ergebnissen oder None bei Fehler
    """
    if analyzer is None:
        print("Analyzer nicht verfügbar.")
        return None
    
    print(f"Exportiere Analyseergebnisse für: '{query}'")
    
    results = analyzer.analyze_query(query)
    
    if "error" in results:
        print(f"Exportfehler: {results['error']}")
        return None
    
    # Ergebnisse vorbereiten
    export_df = results["relevant_entries"].copy()
    
    # Metadaten hinzufügen
    metadata = {
        'QUERY': query,
        'EXPORT_TIMESTAMP': datetime.now().isoformat(),
        'RESULT_COUNT': results['result_count'],
        'AVERAGE_RELEVANCE': results['average_relevance'],
        'SEARCH_QUALITY': results['search_quality'],
        'TOTAL_AVAILABLE_ENTRIES': results['total_available']
    }
    
    # Metadaten als erste Zeile hinzufügen
    for key, value in metadata.items():
        if key not in export_df.columns:
            export_df[key] = ''
        export_df.loc[0, key] = str(value)
    
    # Filename generieren falls nicht angegeben
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:30]
        filename = f"logbuch_analyse_{safe_query}_{timestamp}.csv"
    
    print(f"Exportdaten vorbereitet:")
    print(f"  Einträge: {len(export_df)}")
    print(f"  Spalten: {len(export_df.columns)}")
    print(f"  Dateiname: {filename}")
    
    # In produktiver Umgebung würde hier gespeichert werden:
    # export_df.to_csv(filename, index=False, encoding='utf-8-sig')
    # print(f"Erfolgreich exportiert: {filename}")
    
    return export_df

def print_system_summary(analyzer, stats: Dict, evaluation_results: Dict):
    """
    Druckt umfassende Systemzusammenfassung.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        stats: Datenstatistiken
        evaluation_results: Evaluierungsergebnisse
    """
    print("\n" + "="*70)
    print("SYSTEM-ZUSAMMENFASSUNG")
    print("="*70)
    
    # Datensatz-Info
    print(f"\nDATENSATZ:")
    print(f"  Gesamteinträge: {stats.get('total_entries', 'N/A')}")
    if stats.get('date_range'):
        dr = stats['date_range']
        print(f"  Zeitraum: {dr['start']} bis {dr['end']} ({dr['days']} Tage)")
    print(f"  Top Subsysteme: {len(stats.get('subsystem_distribution', {}))}")
    
    # Technische Spezifikationen
    print(f"\nTECHNISCHE SPEZIFIKATIONEN:")
    print(f"  Embedding-Modell: {analyzer.model_name}")
    if hasattr(analyzer, 'embeddings') and analyzer.embeddings is not None:
        print(f"  Embedding-Dimension: {analyzer.embeddings.shape[1]}")
        print(f"  Index-Größe: {analyzer.embeddings.shape[0]} Vektoren")
    
    # Performance-Metriken
    if "performance" in evaluation_results:
        perf = evaluation_results["performance"]
        print(f"\nPERFORMANCE:")
        print(f"  Durchschnittliche Suchzeit: {perf['average_search_time']:.3f}s")
        print(f"  Suchen pro Sekunde: {perf['searches_per_second']:.1f}")
        print(f"  Qualitätsscore: {perf['average_quality_score']:.3f}")
    
    # Evaluierungs-Metriken
    if "rule_based" in evaluation_results and "aggregate_metrics" in evaluation_results["rule_based"]:
        rb = evaluation_results["rule_based"]["aggregate_metrics"]
        print(f"\nEVALUIERUNG (Regel-basiert):")
        print(f"  Precision@10: {rb['mean_precision']:.3f} ± {rb['std_precision']:.3f}")
        print(f"  Recall@10: {rb['mean_recall']:.3f} ± {rb['std_recall']:.3f}")
        print(f"  F1-Score: {rb['mean_f1']:.3f}")
    
    # Gesamtbewertung
    if "overall_evaluation" in evaluation_results:
        overall = evaluation_results["overall_evaluation"]
        print(f"\nGESAMTBEWERTUNG:")
        print(f"  Score: {overall['overall_score']:.3f}/1.000")
        print(f"  Bewertung: {overall['recommendation']}")
        
        print(f"\nKOMPONENTEN:")
        for component, score, weight in overall['components']:
            print(f"  {component}: {score:.3f} (Gewicht: {weight:.1f})")
    
    # Stabilität
    if "stability" in evaluation_results:
        stab = evaluation_results["stability"]
        print(f"\nSTABILITÄT:")
        print(f"  Query-Ähnlichkeit: {stab.get('mean_similarity', 0):.3f}")
        print(f"  Bereich: {stab.get('min_similarity', 0):.3f} - {stab.get('max_similarity', 0):.3f}")
    
    print("\n" + "="*70)

def create_demo_queries() -> List[str]:
    """
    Erstellt Liste von Demo-Queries für verschiedene Anwendungsfälle.
    
    Returns:
        Liste von beispielhaften Suchanfragen
    """
    return [
        "Probleme mit der Spritzgussmaschine",
        "Hydraulikfehler und Druckprobleme", 
        "Qualitätsfehler in der Produktion",
        "Temperaturprobleme im Kühlsystem",
        "Materialprobleme und Verunreinigungen",
        "Wartung und Instandhaltung",
        "Werkzeugverschleiß und Austausch",
        "Produktionsstillstände",
        "Defekte Sensoren",
        "Reinigung und Hygiene"
    ]

def create_performance_comparison(analyzer, queries: List[str]):
    """
    Erstellt Performance-Vergleichsdiagramm.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        queries: Liste von Test-Queries
    """
    performance_data = []
    
    for query in queries:
        start_time = time.time()
        indices, scores, count = analyzer.semantic_search(query)
        search_time = time.time() - start_time
        
        performance_data.append({
            'query': query[:20] + "..." if len(query) > 20 else query,
            'search_time': search_time,
            'result_count': count,
            'avg_score': np.mean(scores) if scores else 0
        })
    
    # Visualisierung erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    queries_short = [d['query'] for d in performance_data]
    search_times = [d['search_time'] for d in performance_data]
    avg_scores = [d['avg_score'] for d in performance_data]
    
    # Suchzeiten
    ax1.bar(queries_short, search_times, color='lightcoral', alpha=0.7)
    ax1.set_ylabel('Suchzeit (Sekunden)')
    ax1.set_title('Suchzeiten pro Query')
    ax1.tick_params(axis='x', rotation=45)
    
    # Durchschnittliche Scores
    ax2.bar(queries_short, avg_scores, color='lightblue', alpha=0.7)
    ax2.set_ylabel('Durchschnittlicher Relevanz-Score')
    ax2.set_title('Qualität pro Query')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return performance_data