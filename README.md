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
13. [Offene Fragen vor Phase 2](#offene-fragen-vor-phase-2)
14. [Stop Condition (Phase 1)](#stop-condition-phase-1)

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
  qqq_health/
    constituents.py            Point-in-time QQQ-Holdings, harter Stop bei fehlenden Daten
    breadth.py                  A/D, RANA, MCO, MCO-Z, MCSI, MCSI-Z, %>MA, H/L-Oszillator
    price_structure.py          QQQ eigene Preisstruktur + ATR
    build_qqq_health.py         qqq_health_daily Orchestrierung
    build_qqq_outcomes.py       qqq_health_outcomes_daily (Index- + Momentum-Environment-Outcomes)
  reports/
    data_quality.py             maschinen-/menschenlesbarer Datenqualitätsreport
    descriptive.py               Phase-1-Sanity-Check-Reports (keine Threshold-Optimierung)
tests/                        pytest-Suite (36 Tests, synthetische Daten, keine Netzwerkzugriffe)
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

`market_cap(T, D) = close(T, D) * weighted_shares_outstanding(T, D)`,
wobei `weighted_shares_outstanding` über `ticker-overview?date=D` abgefragt
wird (point-in-time nach letztem SEC-Filing vor D). **Niemals** wird
today's Market Cap oder today's Shares Outstanding auf ein historisches
Datum zurückgerechnet. Vollständig dokumentiert in
`config/market_cap_methodology.yaml`, inklusive der Caching-Optimierung
(Monatsraster) und bekannter Limitierungen.

## Feature-Definitionen

Kanonische Formeln (unveränderlich, `config/features.yaml`):

- **True Range** = `max(H-L, |H-PrevClose|, |L-PrevClose|)`
- **ATR14** = einfacher arithmetischer Mittelwert der letzten 14 TR-Werte
  (**kein** Wilder-Smoothing)
- **ATR%** = `ATR14 / Close * 100`
- **ATR Extension** = `((Close-SMA50)/SMA50*100) / ATR%`
- **ADR20** = Mittelwert der letzten 20 abgeschlossenen Handelstage von
  `(High/Low - 1) * 100`
- **RS-Perzentile** (1D/1W/1M): `rank(pct=True)*100`, **ausschließlich**
  innerhalb des an diesem Tag eligible Universe
- **Thrust** = `EMA(short) - EMA(long)` der täglichen Return-Serie
  (1D: 2/5, 1W: 5/15, 1M: 10/25 Tage, `adjust=False`), plus Perzentilrang
  im eligible Universe
- **Outcome-Horizonte**: 5/10/20 Handelstage; MFE/MAE als Fensterextrema
  (nicht nur Endpunkt); ATR-Multiples nutzen die **Signal-Tag-ATR**, nie
  eine zukünftige ATR

Alle Formeln sind 1:1 als reine, getestete Funktionen in
`features/technical.py` und `outcomes/build_outcomes.py` implementiert.

### Dokumentierte methodische Annahme (kein Trading-Threshold, aber
statistik-relevant)

`reached_plus_X_before_minus_X`: Werden beide Schwellen (±X%) am selben
Tag im Fenster berührt, wird konservativ angenommen, dass der Rückgang
zuerst passierte (Tie-Break via striktem `<`-Vergleich, s.
`outcomes/build_outcomes.py` Docstring). **Als offene Frage markiert** —
sollte vor Phase 2 bei Bedarf überprüft werden.

`qqq_health_outcomes_daily` "zukünftige Market Breadth" / RS80+/90+/95+-
Anteile: berechnet auf der **gleichen Kohorte** (heute eligible Ticker),
ausgewertet an deren eigener D+H-Zukunftszeile — nicht auf der zu D+H
tatsächlich eligible Menge. **Ebenfalls als offene Frage markiert** (s.
`qqq_health/build_qqq_outcomes.py` Docstring).

## Datenhaltung

- **Raw-Checkpoints** (`data/raw/...`, gitignored): ein Parquet-File pro
  gefetchtem Handelstag (grouped-daily, reference-tickers, qqq-constituents)
  — Basis für Resumability.
- **Processed** (`data/processed/<table>/year=YYYY/part.parquet`,
  gitignored): `market_universe_daily`, `stock_features_daily`,
  `stock_outcomes_daily`, `qqq_health_daily`, `qqq_health_outcomes_daily`.
  Jede Tabelle hat ein `_manifest.json` mit Build-Timestamp, Zeitraum,
  Config-Version, Git-Commit-Hash, Quelle, Feature-Definitions-Version
  (`config.make_build_metadata`).
- **Reports** (`reports/`, versioniert): `data_quality_report.{json,md}`,
  `reports/phase1_sanity_checks/*.csv`.
- `reference_tickers` und `raw_manifest` als eigenständige Tabellen wurden
  in Phase 1 nicht befüllt (die Referenzdaten werden als Rohdaten pro Tag
  gehalten, s. `data/raw/reference_tickers/`); bei Bedarf leicht als
  zusätzliche `processed`-Tabelle ergänzbar — **offene Frage für Phase 2**,
  ob das für Debugging/Audit-Zwecke gewünscht ist.

## Quickstart / CLI

```bash
pip install -e ".[dev]"
export MASSIVE_API_KEY=...   # nie in eine Datei schreiben

python -m yolo_calibration fetch-raw --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration fetch-qqq-constituents --start 2022-01-01 --end 2026-08-11

python -m yolo_calibration build-universe --start 2022-01-01 --end 2026-08-11 --no-fetch
python -m yolo_calibration build-stock-features --start 2022-01-01 --end 2026-08-11
python -m yolo_calibration build-stock-outcomes --start 2022-01-01 --end 2026-08-11
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
pytest -q          # 36 Tests, ausschließlich synthetische Daten, kein Netzwerkzugriff
```

Abgedeckt: ADR20-Exaktheit, True Range/ATR14 (kein Wilder-Smoothing),
ATR Extension, EMA (`adjust=False`), RS-Perzentile (nur eligible Universe,
tagesweise unabhängig), Thrust + Thrust-Perzentile, MFE/MAE/Forward-Return
(inkl. Window-Extrema statt nur Endpunkt), kein Future-Leakage (Mutations-
und Truncation-Äquivalenztests), "eligible vor RS-Ranking", delistete
Ticker werden nicht wegen heutiger Inaktivität entfernt, OTC-Ausschluss,
A/D + RANA, MCO (EMA19/39, `adjust=False`), MCO-Z (200T-Fenster,
Min-Periods 80), MCSI (Cumsum), MCSI-Z, %>MA, Point-in-time
QQQ-Membership (inkl. des von der Spezifikation explizit geforderten
Tests, der bei einer statischen heutigen Komponentenliste fehlschlägt),
harter Stop bei fehlenden QQQ-Daten, sowie ein End-to-End-Integrationstest
der gesamten Pipeline (raw → universe → features → outcomes) mit
synthetischen Daten und gestubtem API-Client.

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
- ADR/Plan-Limits (Rate Limits, Endpoint-Verfügbarkeit für ETF-Global-
  Constituents auf dem konkret gebuchten Plan) sind unbekannt — der Client
  hat konfigurierbares Rate-Limiting/Retry (`config/massive_api.yaml`
  `http:`), aber die konkreten Limits müssen beim ersten Lauf beobachtet
  und ggf. angepasst werden.

## Offene Fragen vor Phase 2

1. Ist die Monats-Cache-Granularität für Market-Cap-Enrichment
   (`compute_point_in_time_market_cap`) präzise genug, oder wird eine
   feingranularere (Filing-Datum-genaue) Rekonstruktion benötigt?
2. Tie-Break-Konvention für `reached_plus_X_before_minus_X` bei
   Gleichzeitigkeit — konservativ (Rückgang zuerst) korrekt genug, oder
   sollte Intraday-Reihenfolge (falls verfügbar) genutzt werden?
3. Interpretation von "zukünftige Market Breadth" / RS80+/90+/95+-Anteile
   in `qqq_health_outcomes_daily`: gleiche Kohorte in der Zukunft
   ausgewertet (aktuelle Implementierung) vs. tatsächlich zu D+H eligible
   Menge?
4. Sollen `reference_tickers` und `raw_manifest` als eigene versionierte
   `processed`-Tabellen materialisiert werden (aktuell nur als Rohdaten
   pro Tag vorhanden)?
5. War die QQQ-ETF-Constituents-Quelle für den GESAMTEN Zeitraum
   2023–2026 lückenlos verfügbar? Muss beim ersten echten Backfill via
   `verify`-Report geprüft werden.
6. Welche RS-Horizont-Wahl (1D/1W/1M) ist für spätere Leader-Definitionen
   vorgesehen? Phase 1 berechnet alle drei gleichwertig und trifft keine
   Vorauswahl.

## Stop Condition (Phase 1)

Dieses Repository definiert **keine** Leader-, Fresh-Leader-, Constructive-
Reset-, Extended-, Narrative-Lifecycle-, QQQ-Health-State- oder
Market-Regime-Schwellen. Diese werden separat in Phase 2 auf Basis von
2023–2024 kalibriert, auf 2025 validiert (ohne Nachjustierung) und
einmalig auf 2026 getestet (`config/time_splits.yaml`). Es wurde und wird
kein Code in `DocYolo77/yolo-dashboard` verändert.
