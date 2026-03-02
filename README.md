# I/O-effiziente Algorithmen für kürzeste Wege im SEM

## Allgemeine Erklärung des Projektes

Dieses Repository enthält die **Implementierung und experimentelle Evaluation** der zugehörigen Masterarbeit *„I/O-effiziente Algorithmen für kürzeste Wege im External-Memory-Modell“*. Untersucht wird **SSSP auf gerichteten, positiv gewichteten Graphen** im **Semi-External-Memory-Modell (SEM)**, in dem Knotendaten im RAM liegen, die Kanten jedoch extern sind und I/O der Engpass ist.

Im Kern steht ein **hybrides Dijkstra–Bellman–Ford-Verfahren**, das die zwei komplementären Relaxationsprimitive kombiniert:
- **Dijkstra (knotenweise, PQ-getrieben)**: modelliert als 1 I/O pro geladener Adjazenzliste,
- **Bellman–Ford/SPFA (scanbasiert)**: modelliert als Graph-Scan-Kosten pro Runde.

Die Hybrid-Abfolge wird formal als **Schedule-Optimierungsproblem auf dem Shortest-Path-DAG (SPDAG)** betrachtet. Daraus ergeben sich Offline-Referenzen (Set-Cover/IP, Greedy) sowie praktikable Online-Heuristiken (Multi-Switch) und eine Landmark-basierte Hybridvariante.

## Wie lässt sich der Code starten

1. Repository klonen oder herunterladen  
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
3. Experimente starten:
- **`main.py`**: startet die Experimentpipeline (Single-Instanzen / Reihenexperimente) und schreibt Ergebnisse nach `results/`. 
- **Experimente ausführen:** Über `main.py` lassen sich Experimente auf beliebigen Instanzen starten. Dazu werden im Ordner **`asym_input/`** (bzw. den entsprechenden Input-Ordnern) **`.toml`-Konfigurationsdateien** abgelegt und beim Lauf eingelesen. Neue Experimentreihen entstehen somit einfach durch Hinzufügen weiterer `.toml`-Files.

Über einen Schnellstart kann eine beispielhafte Einzelbetrachtung ausgeführt und der Code schnell getestet werden:

~~~bash
git clone https://gitlab.ae.cs.uni-frankfurt.de/telsalim/io-efficient-shortest-paths-master-thesis.git
cd io-efficient-shortest-paths-master-thesis
pip install -r requirements.txt
python main.py
~~~


## Module (Kurzüberblick über den Code)

- **`graph_io.py`**: Orchestrierung der Experimente (Instanzen/Parameter, Ausführung, Checks, JSON + PDF).
- **`modified_dijkstra.py`**: Dijkstra-Varianten (u.a. Warmstart), Logging, SPT/Level-Statistiken.
- **`bellman_ford.py`**: rundenbasierte SPFA-Bellman–Ford-Variante für eine ganzen Durchlauf und ausgehend von einem Dijkstra-Warmstart.
- **`best_single_switch.py`**: **Best-Single-Switch** (optimaler einmaliger Wechselpunkt DJ→BF unter SEM-Kosten).
- **`spdag.py`**: Aufbau/Utilities rund um **SPDAG** und schedulebezogene Auswertungen.
- **`opt_hybrid_with_set_cover.py`**: **IP/Set-Cover** zur optimalen Schedule-Bestimmung (Offline-Referenz, NP-schwer).
- **`multi_switch_hybrid.py`**: **Multi-Switch** Online-Hybrid mit schwellwertbasierter Phasensteuerung über \(e_B\).
- **`landmark_sssp.py`**: Landmark-basierter Hybrid (Vorrelaxation + Projektion + Warmstart-Dijkstra).
- **`graph_generators.py`**: synthetische Testgraphen (**G(n,p)**, **GEO**, **BA**, **DSF**) + CSR/packing.
- **`graph_properties.py`**: WCC/BFS-Reachability/Quellwahl-Heuristiken.
- **`graph_plots.py`**: headless PDF-Plots (Phasen, I/O-Kurven, Level-Histogramme, Matrizen).
- **`heap_numba.py`**: numba-freundlicher Lazy-Min-Heap.

## Ergebnisse

- Die Evaluation erfolgt über mehrere synthetische Graphklassen und Gewichtsregime (u.a. uniform/exponentiell) sowie variierte Blockgrößen \(B\). Im Fokus steht, **wann** hybride Schedules gegenüber den Baselines I/O sparen. Ergebnisse liegen unter **`results/`**.
- In der SEM-Evaluation sind mit hybriden Dijkstra–Bellman–Ford-Schedules **deutliche I/O-Einsparungen** möglich, deren Stärke jedoch von **Blockgröße \(B\)** und **Graphstruktur** abhängt. Die Vorteile konzentrieren sich auf ein **intermediäres Parameterregime**, in dem keine der beiden Baselines klar dominiert und erreichen **bis zu 72%** gegenüber der jeweils besseren Baseline für Blockgröße 512:

| Graphtyp | Multi-Switch | Greedy-Hybrid | Best Single-Switch |
|:--|--:|--:|--:|
| **BA**  | 21.7%  | **66.2%** | 3.9%  |
| **GEO** | -33.6% | **44.8%** | 0.0%  |
| **GNP** | 50.6%  | **64.8%** | 44.7% |
| **DSF** | 45.9%  | **72.2%** | 41.1% |

- Weitere inhaltliche Erklärungen sowie alle relevanten Ergebnisse, Beweise und Definitionen sind in der zugehörigen Masterarbeit zu finden.
- Zu große .pdf Dateien oder zu lange Listen mit Ergebnissen sind nicht in results abgespeichert, können aber jederzeit über die .toml deterministisch reproduziert werden.