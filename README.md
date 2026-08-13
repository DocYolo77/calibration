# YOLO Calibration Engine — Phase 1

Deterministische, point-in-time Research- und Calibration-Infrastruktur für
das YOLO-Momentum-System. Dieses Repository ist **kein** Produktions-
Dashboard, hat **keine** Web-App/Frontend/DB-Server/LLM-Abhängigkeit und
verändert **nicht** `DocYolo77/yolo-dashboard`.

> **Phase 1 Scope**: Daten + Feature/Outcome-Infrastruktur bauen. **Keine**
> Leader-/Fresh-Leader-/Constructive-Reset-/QQQ-Health-/Regime-Schwellen
> werden hier ausgewählt — siehe [Stop Condition](#stop-condition-phase-1).

---

## Inhalt

1. [Architektur](#architektur)
2. [Massive-Endpunkte](#massive-endpunkte)
3. [Point-in-time Universe-Methode](#point-in-time-universe-methode)
4. [Point-in-time QQQ-Constituent-Methode](#point-in-time-qqq-constituent-methode)
5. [Market-Cap-Methode](#market-cap-methode)
6. [Feature-Definitionen](#feature-definitionen)
7. [Datenhaltung](#datenhaltung)
8. [Quickstart / CLI](#quickstart--cli)
9. [Kompletter Backfill starten](#kompletter-backfill-starten)
10. [Tests](#tests)
11. [Datenqualität & Sanity-Reports](#datenqualität--sanity-reports)
12. [Bekannte Limitierungen](#bekannte-limitierungen)
13. [Geklärt: Split-Adjustierung der historischen Preise](#geklärt-split-adjustierung-der-historischen-preise-2026-08-12)
14. [Geklärt am 2026-08-13](#geklärt-am-2026-08-13-vormals-offene-fragen-vor-phase-2)
15. [Offen für Phase 2](#offen-für-phase-2)
16. [Stop Condition (Phase 1)](#stop-condition-phase-1)
17. [Phase 1 Abschlussbericht (Stock-Track)](#phase-1-abschlussbericht-stock-track)

---

## Architektur

```
config/                     zentrale YAML-Configs (Zeitsplits, Filter, Feature-Definitionen, API)
src/yolo_calibration/
  config.py                 Config-Loader + Reproduzierbarkeits-Metadaten
  cli.py, __main__.py       python -m yolo_calibration <command>
  data/
    massive_client.py       REST-Client (Bearer-Auth, Retry/Backoff, Rate-Limit)
    fetch_raw.py             checkpointfähiger Fetch-Orchestrator
    storage.py                Parquet-Layer (raw checkpoints + processed, partitioniert nach Jahr)
    loaders.py                raw -> tidy DataFrame
    cache.py                   Disk-Cache für Referenz-/Fundamentaldaten
  universe/build_universe.py  Asset-Type -> ADR20 -> Market-Cap Filterkette
  features/technical.py       ATR/EMA/SMA/Returns/RS/Thrust (reine Funktionen)
  features/build_features.py  stock_features_daily Orchestrierung
  outcomes/build_outcomes.py  stock_outcomes_daily (5/10/20 Tage forward)
  outcomes/build_market_breadth.py  market_breadth_daily — D0-Kohorte Forward-Performance +
                                     D+H tatsächlich-eligible-Universe RS-Breadth (unabhängig
                                     von QQQ-Health-Verfügbarkeit, s. unten)
  qqq_health/
    constituents.py            Point-in-time QQQ-Holdings, harter Stop bei fehlenden Daten
    breadth.py                  A/D, RANA, MCO, MCO-Z, MCSI, MCSI-Z, %>MA, H/L-Oszillator
    price_structure.py          QQQ eigene Preisstruktur + ATR
    build_qqq_health.py         qqq_health_daily Orchestrierung
    build_qqq_outcomes.py       qqq_health_outcomes_daily (nur noch INDEX-Outcomes — die
                                 Momentum-Environment-Outcomes wurden nach
                                 outcomes/build_market_breadth.py extrahiert, s. dort)
  data/raw_manifest.py          raw_manifest — Checkpoint-Coverage pro Rohdatenquelle/Datum
  reports/
    data_quality.py             maschinen-/menschenlesbarer Datenqualitätsreport
    descriptive.py               Phase-1-Sanity-Check-Reports (keine Threshold-Optimierung)
tests/                        pytest-Suite (57 Tests, synthetische Daten, keine Netzwerkzugriffe)
.github/workflows/
  ci.yml                       Tests bei jedem Push/PR
  historical-build.yml         workflow_dispatch, checkpointfähig via actions/cache
```

Alle research-relevanten Zahlen (Zeiträume, Filter, ATR/EMA/RS/Thrust-
Definitionen, Outcome-Horizonte) liegen **ausschließlich** in `config/*.yaml`
— siehe `config/README_PHASE1_SCOPE.md`.

---

## Massive-Endpunkte

Massive ist der (Okt. 2025) Rebrand von Polygon.io; `api.massive.com` ist
API-kompatibel mit `api.polygon.io`. Tatsächlich verwendete Endpunkte
(siehe `config/massive_api.yaml`):

| Zweck | Endpoint |
|---|---|
| Bulk Daily OHLCV (alle US-Stocks, 1 Request/Tag) | `GET /v2/aggs/grouped/locale/us/market/stocks/{date}` |
| Point-in-time Ticker-Referenzliste | `GET /v3/reference/tickers` (Parameter `date`, `type=CS`, `market=stocks`) |
| Asset-Type-Codes validieren | `GET /v3/reference/tickers/types` |
| Point-in-time Shares Outstanding (Market Cap) | `GET /v3/reference/tickers/{ticker}` (Parameter `date`) |
| Point-in-time QQQ-Holdings | `GET /etf-global/v1/constituents` (Parameter `composite_ticker=QQQ`, `effective_date`) |

Auth: `Authorization: Bearer $MASSIVE_API_KEY` (nur aus Environment-Variable
gelesen, nie geloggt, nie in einer Datei gespeichert — s. `config/massive_api.yaml`
`auth:` Block und `src/yolo_calibration/config.py::get_massive_api_key`).

## Point-in-time Universe-Methode

Filterreihenfolge (Kostenoptimierung, `config/universe.yaml` `filter_order`):

1. **Asset-Type**: pro Handelstag wird die Referenz-Ticker-Liste **mit
   `date=D`** abgerufen (nicht `active=true` von heute) — ein Ticker, der
   heute delisted ist, erscheint also weiterhin an alten Handelstagen vor
   seinem Delisting. Das ist die zentrale Maßnahme gegen Survivorship Bias
   (getestet in `tests/test_universe_point_in_time.py` und
   `tests/test_qqq_pit_membership.py`).
2. **ADR20**: lokal aus bereits geladenen OHLCV-Daten berechnet, keine
   zusätzlichen API-Calls.
3. **Market Cap**: nur noch für die (deutlich kleinere) Menge an
   Ticker-Tagen, die Schritt 1+2 überstanden haben — ein `ticker-overview`
   Call pro Ticker, monatsweise gecacht (s. unten).

## Point-in-time QQQ-Constituent-Methode

Quelle **A** aus der Spezifikation ist verfügbar und wird verwendet: die
Massive/ETF-Global `constituents`-Endpoint liefert historische QQQ-Holdings
über `effective_date`, mit Datenabdeckung seit 2017-04-03 (laut Dokumentation)
— deckt damit den gesamten Untersuchungszeitraum 2023–2026 ab.

Quelle **B** (verifizierte historische NDX-100-Quelle) wurde **nicht**
implementiert — es wurde keine zusätzliche, unabhängig verifizierte Quelle
identifiziert, und da Quelle A verfügbar ist, war sie nicht nötig.

**Harter Stop (Quelle C)**: Fehlt für einen Handelstag ein
Constituents-Snapshot, wirft `QQQConstituentsUnavailable` — der
QQQ-Health-Track wird für den betroffenen Range NICHT gebaut, sondern das
CLI-Kommando `build-qqq-health` gibt Exit-Code 2 zurück und der
Datenqualitätsreport markiert `qqq_health_status: "unavailable"` mit
Fehlertext. Kein stiller Fallback auf die heutige Ticker-Liste. Das wird
explizit getestet in `tests/test_qqq_pit_membership.py::test_hard_stop_when_pit_data_missing_no_silent_fallback`.

Die tatsächliche Komponentenanzahl pro Tag wird gespeichert
(`qqq_health.constituents.component_counts`) und fließt in den
Datenqualitätsreport ein.

**Waren QQQ-Constituents 2023–2026 vollständig verfügbar?** Das kann in
dieser Umgebung nicht abschließend beantwortet werden, da kein
`MASSIVE_API_KEY` zur Verfügung stand, um den Live-Endpoint gegen den vollen
Zeitraum zu prüfen (s. [Bekannte Limitierungen](#bekannte-limitierungen)).
Der Code ist so gebaut, dass ein Backfill dies beim ersten echten Lauf
automatisch aufdeckt (harter Stop bei Lücken) und im Datenqualitätsreport
sichtbar macht.

## Market-Cap-Methode

`market_cap(T, D) = close_UNADJUSTED(T, D) * weighted_shares_outstanding(T, D)`,
wobei `weighted_shares_outstanding` über `ticker-overview?date=D` abgefragt
wird (point-in-time nach letztem SEC-Filing vor D), monatsweise gecacht
(erster Tag des Monats). **Niemals** wird today's Market Cap oder today's
Shares Outstanding auf ein historisches Datum zurückgerechnet. Vollständig
dokumentiert in `config/market_cap_methodology.yaml`.

**Entscheidung für Phase 1** (`decided_for_phase_1` in dieser Config):
die monatliche Point-in-Time-Approximation wird **beibehalten** — keine
filing-genaue Rekonstruktion (z.B. direkt aus SEC-EDGAR-Filing-Terminen)
wird ergänzt, da der Aufwand gegenüber der gemessenen Auswirkung
unverhältnismäßig wäre. Stattdessen läuft bei jedem `verify` ein
permanenter Sensitivity-Diagnose-Check
(`reports/data_quality.py::_market_cap_eligibility_sensitivity`,
Config-Parameter `config/universe.yaml` `market_cap.sensitivity`):

- **near-threshold**: Ticker-Tage, deren `market_cap` innerhalb von
  `near_threshold_band_pct` (Standard 15%) der $1-Mrd.-Grenze liegt —
  Eligibility, die durch einen plausiblen Vendor-Datenfehler allein kippen
  könnte.
- **volatile**: davon Ticker, deren monatlicher `market_cap`-Durchschnitt
  mindestens einmal um mehr als `volatility_ratio_threshold` (Standard 3x)
  zum Vormonat springt — das Muster, das beim echten 2023-Backfill bei
  Tickern mit häufigen Reverse-Splits (z.B. `MULN`) gefunden wurde: die
  vom Vendor gemeldete `weighted_shares_outstanding` selbst schwankt dort
  stark zwischen benachbarten Monats-Buckets, auch wenn der (korrekt
  unadjustierte) Preis sich glatt bewegt. Kein Preis-Adjustierungs-Bug
  (der ist separat und bereits gefixt, s. unten) — ein
  Vendor-Point-in-Time-Filing-Artefakt.

Beide Zahlen sind rein deskriptiv (verändern `eligible`/`market_cap_ok`
nicht) und fließen in `data_quality_report.md` sowie den
Phase-1-Abschlussbericht ein.

## Feature-Definitionen

Kanonische Formeln (unveränderlich, `config/features.yaml`):

- **True Range** = `max(H-L, |H-PrevClose|, |L-PrevClose|)`
- **ATR14** = einfacher arithmetischer Mittelwert der letzten 14 TR-Werte
  (**kein** Wilder-Smoothing)
- **ATR%** = `ATR14 / Close * 100`
- **ATR Extension** = `((Close-SMA50)/SMA50*100) / ATR%`
- **ADR20** = Mittelwert der letzten 20 abgeschlossenen Handelstage von
  `(High/Low - 1) * 100`
- **RS-Perzentile** (1D/1W/1M/**3M/6M/12M**, Fenster 1/5/21/**63/126/252**
  Handelstage): `rank(pct=True)*100`, **ausschließlich** innerhalb des an
  diesem Tag eligible Universe
- **Thrust** = `EMA(short) - EMA(long)` der täglichen Return-Serie
  (1D: 2/5, 1W: 5/15, 1M: 10/25 Tage, `adjust=False`), plus Perzentilrang
  im eligible Universe
- **SMA50-Slope** (`sma50_slope_pct`) = prozentuale Veränderung des SMA50
  gegenüber sich selbst vor `lookback_days` (Default 5) Handelstagen:
  `(SMA50[t] - SMA50[t-N]) / SMA50[t-N] * 100`
- **SMA50-Persistence** (`sma50_persistence_days`) = vorzeichenbehaftete
  Streak-Länge in Handelstagen: wie viele Tage in Folge (bis einschließlich
  heute) steht Close kontinuierlich über (positiv) bzw. unter (negativ)
  SMA50. Reset auf ±1 am Tag des Wechsels; NaN während des SMA50-Warmups.
- **Outcome-Horizonte**: 5/10/20 Handelstage; MFE/MAE als Fensterextrema
  (nicht nur Endpunkt); ATR-Multiples nutzen die **Signal-Tag-ATR**, nie
  eine zukünftige ATR

SMA50-Slope und SMA50-Persistence sind **Feature-Definitionen** (wie ATR/EMA
oben), keine Trading-Schwellen — es wird kein "trending"/"persistent ab X
Tagen"-Cutoff festgelegt, nur der rohe Wert berechnet.

Alle Formeln sind 1:1 als reine, getestete Funktionen in
`features/technical.py` und `outcomes/build_outcomes.py` implementiert.

### Dokumentierte methodische Annahme (kein Trading-Threshold, aber
statistik-relevant)

`reached_plus_X_before_minus_X`: Werden beide Schwellen (±X%) am selben
Tag im Fenster berührt, wird konservativ angenommen, dass der Rückgang
zuerst passierte (Tie-Break via striktem `<`-Vergleich, s.
`outcomes/build_outcomes.py` Docstring). **Entscheidung für Phase 1**:
diese Konvention wird **beibehalten** (keine Intraday-Pipeline ergänzt),
aber jeder so entschiedene Fall wird jetzt explizit gezählt und markiert —
eine begleitende Boolean-Spalte
`reached_plus_X_before_minus_X_tie_{H}d` ist `True` genau dann, wenn die
wahre Intraday-Reihenfolge aus Tages-OHLC nicht bestimmbar war (beide
Schwellen am selben Tag berührt). `reports/data_quality.py` aggregiert
diese Spalten zu einer Gesamtzahl + Prozentsatz je Schwelle/Horizont im
Datenqualitätsreport.

**"Zukünftige Market Breadth" (`future_rs{bucket}plus_share_{H}d`,
jetzt in `market_breadth_daily`, s. `outcomes/build_market_breadth.py`)**:
**Gefixt am 2026-08-13.** Berechnet jetzt korrekt auf der zu D+H
**tatsächlich eligible** Menge (deren eigene RS-Verteilung an D+H), NICHT
mehr auf der an D0 eligible Kohorte, die nur auf Existenz (nicht
Eligibility) ihrer D+H-Zukunftszeile geprüft wurde. Die D0-Kohorten-
Forward-Performance (`median_forward_return_{H}d`, `mfe`/`mae`,
`share_positive_{H}d`, `cohort_size_{H}d`) bleibt davon getrennt und
unverändert auf der D0-Kohorten-Basis. Diese Tabelle ist zudem jetzt
unabhängig von der QQQ-Health-Verfügbarkeit baubar (sie hing vorher
fälschlich am `qqq_health_daily`-Erfolg, obwohl sie nie QQQ-Constituents
oder -Preisdaten brauchte) — s. `outcomes/build_market_breadth.py`
Docstring für die vollständige Herleitung.

## Datenhaltung

- **Raw-Checkpoints** (`data/raw/...`, gitignored): ein Parquet-File pro
  gefetchtem Handelstag (grouped-daily, reference-tickers, qqq-constituents)
  — Basis für Resumability.
- **Processed** (`data/processed/<table>/year=YYYY/part.parquet`,
  gitignored): `market_universe_daily`, `stock_features_daily`,
  `stock_outcomes_daily`, `qqq_health_daily`, `qqq_health_outcomes_daily`
  (Index-Outcomes only, s. oben), `market_breadth_daily`,
  `reference_tickers`, `raw_manifest`. Jede Tabelle hat ein
  `_manifest.json` mit Build-Timestamp, Zeitraum, Config-Version,
  Git-Commit-Hash, Quelle, Feature-Definitions-Version
  (`config.make_build_metadata`).
- **Reports** (`reports/`, versioniert): `data_quality_report.{json,md}`,
  `reports/phase1_sanity_checks/*.csv`.
- **`reference_tickers`** (materialisiert seit 2026-08-13, CLI
  `build-reference-tickers`): konsolidiert die täglichen Rohdaten-Checkpoints
  (`data/raw/reference_tickers/`) zu einer versionierten `processed`-Tabelle
  (date, ticker, type, market, primary_exchange, active, ...) — dieselbe
  Point-in-Time-Semantik wie die Rohdaten, jetzt aber mit Build-Metadata und
  ohne dass Konsumenten selbst über Tages-Parquets iterieren müssen.
- **`raw_manifest`** (materialisiert seit 2026-08-13, CLI
  `build-raw-manifest`): eine Zeile pro (Quelle, Datum) für jede der vier
  Rohdatenquellen (`grouped_daily`, `grouped_daily_unadjusted`,
  `qqq_constituents`, `reference_tickers`) mit `row_count` — reine
  Coverage-/Reproduzierbarkeits-Metadaten (welche Checkpoints existieren
  tatsächlich, wie viele Zeilen), kein Inhalts-Korrektheitscheck. Ein
  fehlender Checkpoint erscheint als fehlende Zeile, ein leerer (z.B.
  Markt-Feiertag) als vorhandene Zeile mit `row_count=0` — beides bleibt
  unterscheidbar.

## Quickstart / CLI

```bash
pip install -e ".[dev]"
export MASSIVE_API_KEY=...   # nie in eine Datei schreiben

python -m yolo_calibration fetch-raw --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration fetch-qqq-constituents --start 2022-01-01 --end 2026-08-11

python -m yolo_calibration build-reference-tickers --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration build-raw-manifest --start 2022-01-01 --end 2026-08-11

python -m yolo_calibration build-universe --start 2022-01-01 --end 2026-08-11 --no-fetch
python -m yolo_calibration build-stock-features --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration build-stock-outcomes --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration build-market-breadth --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration build-qqq-health --start 2022-01-01 --end 2026-08-11 --no-fetch
python -m yolo_calibration build-qqq-outcomes --start 2022-01-01 --end 2026-08-11

python -m yolo_calibration verify --start 2022-01-01 --end 2026-08-11
```

`history_buffer_start` (`config/time_splits.yaml`, Standard `2022-01-01`)
liegt bewusst vor `calibration.start` (`2023-01-01`), damit Rolling-Features
mit langem Lookback (SMA200, MCO/MCSI 200-Tage-Z-Score mit 80-Tage-Warmup)
zum Start des Calibration-Fensters bereits valide Werte haben, ohne
zukünftige Daten zu verwenden.

## Kompletter Backfill starten

Über GitHub Actions: **Actions → Historical Build → Run workflow**, mit
`start_date`/`end_date` (z.B. `2022-01-01` / `2026-08-11`) und optional
`rebuild: true` um Checkpoints zu ignorieren. `MASSIVE_API_KEY` muss als
Repository-Secret hinterlegt sein. Der Job ist via `actions/cache`
checkpointfähig — ein erneuter manueller Lauf mit demselben Datumsbereich
setzt bei den zuletzt gespeicherten Rohdaten fort statt neu zu laden.

## Tests

```bash
pytest -q          # 57 Tests, ausschließlich synthetische Daten, kein Netzwerkzugriff
```

Abgedeckt: ADR20-Exaktheit, True Range/ATR14 (kein Wilder-Smoothing),
ATR Extension, EMA (`adjust=False`), RS-Perzentile inkl. 3M/6M/12M (nur
eligible Universe, tagesweise unabhängig), SMA50-Slope-Formel,
SMA50-Persistence (Streak-Zählung inkl. Flip- und Ticker-Grenzfällen),
Thrust + Thrust-Perzentile, MFE/MAE/Forward-Return (inkl. Window-Extrema
statt nur Endpunkt), kein Future-Leakage (Mutations- und
Truncation-Äquivalenztests, jetzt auch für SMA50-Slope/Persistence),
"eligible vor RS-Ranking", delistete Ticker werden nicht wegen heutiger
Inaktivität entfernt, OTC-Ausschluss, A/D + RANA, MCO (EMA19/39,
`adjust=False`), MCO-Z (200T-Fenster, Min-Periods 80), MCSI (Cumsum),
MCSI-Z, %>MA, Point-in-time QQQ-Membership (inkl. des von der
Spezifikation explizit geforderten Tests, der bei einer statischen
heutigen Komponentenliste fehlschlägt), harter Stop bei fehlenden
QQQ-Daten, **Skaleninvarianz der Ratio-Features gegenüber dem
Split-Adjustierungs-Faktor** (`test_scale_invariance.py` — market_cap
nutzt nachweislich den unadjustierten Preis, alle anderen Features sind
beweisbar unabhängig vom Skalierungsfaktor), Market-Cap-Enrichment-Batching
nach Ticker-Monat statt Ticker-Tag, Market-Cap-Eligibility-Sensitivity-
Diagnose (near-threshold + volatilitätsgetriebene Ticker,
`test_data_quality_report.py`), `future_market_breadth` auf Basis der zu
D+H tatsächlich eligible Universe statt der D0-Kohorte
(`test_market_breadth.py` — inkl. eines expliziten Regressionstests, der
beweist, dass das alte kohortenbasierte Ergebnis vom neuen abweicht),
`reached_plus_X_before_minus_X`-Tie-Diagnose (`test_outcomes.py`,
`test_data_quality_report.py`), Materialisierung von `reference_tickers`/
`raw_manifest` aus Rohdaten-Checkpoints (`test_reference_tickers_and_manifest.py`),
sowie ein End-to-End-Integrationstest der gesamten Pipeline
(raw → universe → features → outcomes → market_breadth) mit synthetischen
Daten und gestubtem API-Client.

**In dieser Session konnte kein Live-Lauf gegen die echte Massive-API
durchgeführt werden** (kein `MASSIVE_API_KEY` verfügbar). Die Testsuite
verifiziert die Berechnungslogik vollständig mit synthetischen Daten; die
End-to-End-Korrektheit gegen echte historische Daten (Datenlücken,
Sonderfälle, tatsächliche QQQ-Coverage) muss beim ersten echten Backfill
verifiziert werden — der `verify`-Report ist genau dafür gebaut.

## Datenqualität & Sanity-Reports

`verify` erzeugt `reports/data_quality_report.{json,md}` (Handelstage,
Ticker/Jahr, Median-Universengröße, fehlende Market-Cap-/OHLC-Anteile,
QQQ-Komponentenzahlen, Warnungen) sowie deskriptive Sanity-Checks unter
`reports/phase1_sanity_checks/` (RS-Dezil/ATR-Extension-Bucket/EMA20-
Distance-Bucket vs. 10D-MFE/MAE; MCO-Z-/MCSI-Z-/%>EMA21-/H-L-Oszillator-
Dezil vs. Momentum-Environment-Outcomes). Diese Reports optimieren
**keine** Schwelle — rein deskriptiv.

## Bekannte Limitierungen

- Kein `MASSIVE_API_KEY` in dieser Build-Umgebung → kein Live-Backfill,
  keine Verifikation realer Datenlücken/Plan-Limits/Rate-Limits.
- Massive-API-Dokumentation wurde per Web-Fetch (zusammenfassend) gelesen,
  nicht als vollständige OpenAPI-Spec — kleinere Feldnamen-Abweichungen
  sind beim ersten echten Lauf möglich und sollten dort auffallen
  (HTTP-Fehler/leere Ergebnisse werden geloggt, nicht verschluckt).
- **Bestätigt durch den ersten Live-Testlauf (Jan. 2023, 2026-08-12):**
  der `/etf-global/v1/constituents`-Endpoint (Quelle A für QQQ-Holdings)
  liefert `403 "You are not entitled to this data"` auf dem aktuell
  gebuchten Plan — der QQQ-Health-Track ist damit vorerst blockiert
  (korrekt als "unavailable" markiert, kein stiller Fallback). Die
  Stock-Pipeline (Universe/Features/Outcomes) lief im selben Test
  vollständig fehlerfrei durch.
- Rate Limit: laut Massive-Doku bis zu 100 Requests/Sekunde, bevor
  serverseitiges Throttling/429 einsetzt. `requests_per_minute_soft_limit`
  in `config/massive_api.yaml` steht auf 3000 (=50/s, Sicherheitsmarge
  unter dem dokumentierten Ceiling). Der ursprüngliche konservative
  Default von 90/min hätte einen vollen 2022–2026-Backfill auf
  schätzungsweise 10+ Stunden gestreckt (Hochrechnung aus dem
  Market-Cap-Enrichment-Schritt des Testlaufs).

## Geklärt: Split-Adjustierung der historischen Preise (2026-08-12)

Beim Explorieren der echten 2023-Testdaten fielen 46 Ticker mit unplausiblen
ADJUSTED Close-Preisen auf (bis zu mehrere hundert Milliarden USD/Aktie,
5.924 Zeilen ≈ 0,22%, davon 5.141 als eligible markiert — z.B. `MULN` am
2023-06-15 mit $14,1 Mrd./Aktie statt real ca. $1–3). **Root Cause
bestätigt**: der grouped-daily Endpoint wird mit `adjusted=true` abgefragt;
das adjustiert historische Preise anhand ALLER Splits, die bis zum
BUILD-Zeitpunkt (heute) bekannt sind, nicht nur bis zum jeweils historischen
Datum — bei Penny Stocks mit mehreren Reverse-Splits zwischen 2023 und heute
kumuliert sich das zu absurden absoluten Werten.

**Analyse der tatsächlichen Auswirkung**: Jede aktuell implementierte
Feature-/Outcome-Formel (ADR20, Returns, ATR%, ATR-Extension, Thrust,
RS-Perzentile, Distanz-zu-MA, MFE/MAE, ATR-Multiples) ist ein **Verhältnis**
innerhalb derselben Ticker-Serie — der fehlerhafte, aber pro Ticker
*konstante* Skalierungsfaktor kürzt sich algebraisch heraus und ist daher
**nicht** betroffen (empirisch bestätigt: MULNs ADR20 im Juni 2023 liegt
bei plausiblen 12–20%, obwohl der absolute Preis absurd ist). Die einzige
Größe, die **tatsächlich betroffen war**: `market_cap` (Preis × Shares
Outstanding — kein Verhältnis, daher nicht skaleninvariant), inkl. der
falschen Eligibility bei 5.141 Zeilen.

**Fix implementiert**: `market_cap` verwendet jetzt einen separat via
`adjusted=false` gefetchten UNADJUSTED Close-Preis
(`data/fetch_raw.py::fetch_grouped_daily_unadjusted_range`,
`universe/build_universe.py`), alle anderen Features bleiben unverändert auf
der `adjusted=true`-Serie (korrekt und Standardpraxis für
Verhältnis-basierte technische Analyse). Die Skaleninvarianz-Eigenschaft ist
jetzt als Regressionstest abgesichert
(`tests/test_scale_invariance.py`). Vollständig dokumentiert in
`config/market_cap_methodology.yaml` (`why_unadjusted_close_specifically`).
Ein `implausible_price_row_count`-Diagnose-Feld im Data-Quality-Report
bleibt als Frühwarnsystem für künftige Builds erhalten.

## Geklärt am 2026-08-13 (vormals "Offene Fragen vor Phase 2")

1. **Monats-Cache-Granularität für Market-Cap**: **beibehalten**, keine
   filing-genaue Rekonstruktion in Phase 1 (s.
   [Market-Cap-Methode](#market-cap-methode) und
   `config/market_cap_methodology.yaml` `decided_for_phase_1`). Ein
   permanenter Sensitivity-Diagnose-Check macht die Auswirkung in jedem
   Build sichtbar statt sie stillschweigend zu tragen.
2. **Tie-Break-Konvention** für `reached_plus_X_before_minus_X`:
   **beibehalten** (konservativ, Rückgang zuerst), keine Intraday-Pipeline.
   Jeder nicht-bestimmbare Fall wird jetzt gezählt und im
   Datenqualitätsreport ausgewiesen (`_tie`-Spalten, s.
   [Feature-Definitionen](#feature-definitionen)).
3. **"Zukünftige Market Breadth"**: **gefixt** — berechnet jetzt auf der zu
   D+H tatsächlich eligible Menge, nicht der D0-Kohorte (s.
   `outcomes/build_market_breadth.py`).
4. **`reference_tickers` und `raw_manifest`**: **materialisiert** als
   eigene versionierte `processed`-Tabellen (s. [Datenhaltung](#datenhaltung)).
5. **QQQ-ETF-Constituents-Verfügbarkeit 2022–2026**: **beantwortet** — der
   Endpoint liefert auf dem aktuell gebuchten Plan durchgehend `403 "You
   are not entitled to this data"`, unabhängig vom angefragten Datum (eine
   Plan-Entitlement-Antwort, keine datumsabhängige Datenlücke). Explizit
   per `data_quality_report.md` bestätigt für die real gebauten Jahre 2023,
   2025 und 2026 (Teiljahr); 2022/2024 zeigen strukturell dasselbe
   Verhalten (derselbe plankonstante 403, nicht erneut einzeln verifiziert).
   QQQ-Health bleibt entsprechend permanent `"unavailable"`; kein Fallback
   implementiert (Spec-Section-10-Hard-Stop bleibt aktiv).

## Offen für Phase 2

6. Welche RS-Horizont-Wahl (jetzt: 1D/1W/1M/3M/6M/12M) ist für spätere
   Leader-Definitionen vorgesehen? Phase 1 berechnet alle gleichwertig und
   trifft bewusst **keine** Vorauswahl — diese Entscheidung wird hier nicht
   getroffen.

## Stop Condition (Phase 1)

Dieses Repository definiert **keine** Leader-, Fresh-Leader-, Constructive-
Reset-, Extended-, Narrative-Lifecycle-, QQQ-Health-State- oder
Market-Regime-Schwellen. Diese werden separat in Phase 2 auf Basis von
2023–2024 kalibriert, auf 2025 validiert (ohne Nachjustierung) und
einmalig auf 2026 getestet (`config/time_splits.yaml`). Es wurde und wird
kein Code in `DocYolo77/yolo-dashboard` verändert.

---

## Phase 1 Abschlussbericht (Stock-Track)

**Stand: 2026-08-13.** Deckt den Stock-Track ab (`market_universe_daily`,
`stock_features_daily`, `stock_outcomes_daily`, `market_breadth_daily`,
`reference_tickers`, `raw_manifest`). Der QQQ-Health-Track ist explizit
**nicht** Teil dieses Berichts, da er keine Daten produziert (permanent
`"unavailable"`, s. Punkt 5 unten) — es gibt dort nichts abzuschließen.

### 1. Was gebaut wurde

Eine deterministische, point-in-time-korrekte Daten-Pipeline von
Massive-Rohdaten bis zu fertigen Feature-/Outcome-Tabellen für 2022–2026
(Teiljahr bis 08-11), in fünf separaten, checkpointfähigen GitHub-Actions-
Läufen (einer pro Jahr — siehe Abschnitt 4). Architektur, Formeln und
Konfiguration sind vollständig oben im README sowie in `config/*.yaml`
dokumentiert; dieser Abschnitt fasst nur den Stand zum Phase-1-Abschluss
und die in dieser Session (2026-08-13) getroffenen Entscheidungen zusammen.

### 2. In dieser Session geklärt/gefixt (2026-08-13)

Alle vier Punkte sind Code- bzw. Dokumentationsänderungen, die vor diesem
Abschluss explizit angefordert und umgesetzt wurden — Details, Code-Stellen
und Tests jeweils oben verlinkt:

1. **Market-Cap-Approximation**: monatliches Point-in-Time-Raster wird für
   Phase 1 **beibehalten**, keine filing-genaue Rekonstruktion. Neu: ein
   permanenter Sensitivity-Diagnose-Check (`market_cap_eligibility_sensitivity`
   im Data-Quality-Report) macht sichtbar, wie viele Ticker-Tage nahe der
   $1-Mrd.-Grenze liegen und davon wie viele durch eine unplausible
   Shares-Outstanding-Schwankung (Reverse-Split-Muster wie bei `MULN`)
   getrieben sind.
2. **`future_market_breadth`**: war faktisch nie lauffähig (hing hinter dem
   permanent nicht verfügbaren QQQ-Health-Track) UND rechnete auf der
   falschen Basis (D0-Kohorte statt D+H-tatsächlich-eligible-Universe).
   Beides gefixt: neue eigenständige Tabelle `market_breadth_daily`
   (`outcomes/build_market_breadth.py`), unabhängig von QQQ-Verfügbarkeit,
   mit korrigierter Semantik und Regressionstest, der explizit beweist,
   dass altes und neues Ergebnis divergieren.
3. **`reference_tickers`/`raw_manifest`**: als versionierte `processed`-
   Tabellen materialisiert (`build-reference-tickers`, `build-raw-manifest`
   CLI-Kommandos), mit Build-Metadata.
4. **Tie-Break-Konvention** (`reached_plus_X_before_minus_X`): unverändert
   beibehalten (keine Intraday-Pipeline), aber jeder nicht-bestimmbare Fall
   wird jetzt über eine `_tie`-Begleitspalte pro Zeile gezählt und im
   Data-Quality-Report aggregiert.

**Wichtiger Vorbehalt**: Punkte 1 (Sensitivity-Zahlen), 2
(`market_breadth_daily`-Werte) und 4 (Tie-Counts) sind neuer Code, der
noch **nicht** gegen die bereits gebauten realen 2022–2026-Datensätze
gelaufen ist — diese wurden vor der Implementierung gebaut. Die
Korrektheit ist über die Testsuite (synthetische Daten, s. unten)
abgesichert; die tatsächlichen Zahlen für den realen Datensatz liegen erst
nach einem erneuten `verify`- bzw. `build-market-breadth`-Lauf vor. Das
ist eine bewusste Reihenfolge-Entscheidung dieser Session (Fix zuerst
dokumentieren/implementieren, Neu-Lauf danach), keine offene Lücke in der
Umsetzung selbst.

### 3. Empirisch bestätigt (echte Daten, vor dieser Session gebaut)

Aus der Exploration der echten Backfill-Ergebnisse (DuckDB-Abfragen gegen
heruntergeladene Artefakte, s. Abschnitt "Geklärt: Split-Adjustierung"):

| Jahr | Trading Days | Feature-Zeilen | Ticker gesamt | Median eligible | Implausible-Price-Zeilen |
|---|---|---|---|---|---|
| 2023 | 250 | 2.663.316 | 13.048 | 403 | 2.698 (17 Ticker) |
| 2025 | 250 | 2.814.310 | 13.423 | 550 | 138 (4 Ticker) |
| 2026 (bis 08-11) | 152 | 1.839.539 | 13.953 | 986 | 0 |

Alle drei liefen fehlerfrei durch (2022/2024 vorab mit identischem
Code-Stand, s. Abschnitt 4). `market_cap` nutzt nachweislich den
unadjustierten Preis (MULN-Beispiel: `close` läuft bis auf ~$858M
(Adjustierungs-Artefakt, erwartet), `market_cap` bleibt im Bereich
Zehntausende bis niedrige Millionen — plausibel für einen Penny Stock).

**Vorab-Befund zur Sensitivity-Diagnose (manuelle Ad-hoc-Analyse, motivierte
Punkt 1 oben)**: im 2023-Datensatz zeigen 209 Ticker mindestens einen
Monat-zu-Monat-`market_cap`-Sprung >3x gefolgt von einem Rückgang >2,5x —
das MULN-Reverse-Split-Muster. Davon führen 356 Ticker-Tage (von ~99.000
eligible Zeilen insgesamt, ≈0,36%) zu einer spurios über die $1-Mrd.-Grenze
gehobenen Eligibility. Das ist die konkrete Zahl, die zur permanenten
Diagnose in Punkt 1 geführt hat — ein erneuter `verify`-Lauf mit dem neuen
Code liefert diese Zahl künftig automatisch statt durch manuelle Analyse.

### 4. Abdeckung 2022–2026

| Jahr | Status | Code-Stand (Commit) |
|---|---|---|
| 2022 | ✅ gebaut | `7eba4fa` (Market-Cap-Fix + Structural-Features, vor OOM-Fix — funktional identisch, s. unten) |
| 2023 | ✅ gebaut (Rebuild) | `7684ff2` (aktuell) |
| 2024 | ✅ gebaut | `7eba4fa` |
| 2025 | ✅ gebaut | `7684ff2` (aktuell) |
| 2026 (bis 08-11) | ✅ gebaut (Teiljahr) | `7684ff2` (aktuell) |

Der einzige Unterschied zwischen `7eba4fa` und dem aktuellen `7684ff2` ist
ein reiner Speicher-Fix in `outcomes/build_outcomes.py` (O(n) statt
O(n·Horizont) Matrizen, behebt einen OOM-Crash bei einem versuchten
zusammenhängenden Mehrjahres-Lauf) — zum Zeitpunkt dieses Fixes durch die
damalige volle Testsuite als bit-identisch zum Vorher-Verhalten abgesichert
(exakte Wert-Tests in `test_outcomes.py`). 2022/2024 sind inhaltlich NICHT
stale und müssen nicht neu gebaut werden. Die vier in dieser Session
umgesetzten Fixes (Abschnitt 2) sind jedoch neuer als alle fünf Jahres-Läufe
— siehe Vorbehalt oben.

### 5. Bekannte, akzeptierte Limitierungen (nicht Teil von Phase 2)

- **QQQ-Health-Track dauerhaft unavailable**: `/etf-global/v1/constituents`
  liefert plan-bedingt `403` für jedes Datum. Kein Fallback implementiert
  (Spec-Section-10-Hard-Stop). Ein Plan-Upgrade ist die einzige Abhilfe;
  außerhalb des Scopes dieses Berichts.
- **Market-Cap-Monatsraster**: akzeptierte Phase-1-Entscheidung (Abschnitt
  2, Punkt 1) — dauerhaft überwacht, nicht behoben.
- **Tie-Break bei `reached_plus_X_before_minus_X`**: akzeptierte Phase-1-
  Entscheidung (Abschnitt 2, Punkt 4) — dauerhaft gezählt, nicht behoben.

### 6. Explizit NICHT Teil dieses Abschlusses

- **Keine Visualisierung** der Daten wurde begonnen.
- **Keine Leader-/Fresh-Leader-/Constructive-Reset-/Regime-Schwellen**
  wurden gewählt oder kalibriert (Stop Condition bleibt in Kraft).
- **Keine RS-Horizont-Entscheidung** für eine spätere Leader-Definition
  (Punkt 6 unter "Offen für Phase 2").
- Kein Code in `DocYolo77/yolo-dashboard` wurde berührt.

### 7. Empfohlener nächster Schritt (nicht in dieser Session ausgeführt)

Ein erneuter `verify`- (und, wo relevant, `build-market-breadth`-)Lauf
gegen die bestehenden 2022–2026-Checkpoints, um die in Abschnitt 2
genannten neuen Diagnose-Zahlen (Sensitivity, Tie-Counts,
`market_breadth_daily`-Werte) für den vollständigen realen Datensatz
einmalig zu erzeugen, bevor Phase 2 (Leader-/Threshold-Kalibrierung)
beginnt.
