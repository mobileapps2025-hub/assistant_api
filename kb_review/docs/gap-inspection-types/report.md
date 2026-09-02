# USER DOCUMENTATION

## Special (ad-hoc) vs. routine (recurring) inspections in MCL

Two separate things in MCL sound like "special vs. routine". They are set in different places, by different people, and they do different jobs. It helps to keep them apart.

### 1. The "Reason" you pick when you start an inspection — Routine or Special

Every time you open a checklist in the mobile app, the start screen (the one with the store, the checklist name and the **Start checklist** button) has a **Reason** choice with two options: **Routine** and **Special**. **Routine** is pre-selected; tap the other option to switch to **Special**, then start.

This is a label you attach to that one inspection. It is chosen on the mobile app only — the web dashboard has no place to set it.

### 2. The checklist's frequency — set once in the web dashboard

Separately, each checklist has a **Periodicity** setting in the dashboard's checklist setup, chosen by whoever is allowed to create and edit checklists at your company. The options are:

- **N/A** — no recurrence (an on-demand checklist)
- **Daily**, **Weekly** (optionally tied to one weekday), **Monthly**, **Quarterly**, **Biannual**, **Yearly**

The same setup screen also holds the checklist's valid-from / valid-to dates and, optionally, a time-of-day window.

This frequency setting is the one that actually decides whether MCL treats a checklist as recurring or as on-demand.

---

## What this means for scheduling

**Picking "Special" instead of "Routine" changes nothing about scheduling.** It does not create an extra inspection, does not move a due date, does not exempt you from anything, and does not make the checklist appear or disappear. It is purely a label on the record you are about to create.

In particular:

- A **Special** inspection still counts as having done the checklist for the current period. If a weekly checklist is run once this week and the person chose "Special", the checklist shows as done for that week — on the mobile checklist list and in the dashboard's planned-vs-completed and not-completed views alike. MCL does not separate the two reasons when it decides whether a scheduled checklist was completed.
- Time-of-day restrictions apply either way. If a checklist has a time window and you try to start it outside that window, the app refuses and shows a message telling you it can only be run during that window. This happens before you get to the Reason choice, so choosing "Special" cannot get you past it.
- Validity dates apply either way. A checklist only appears in the mobile app on days inside its valid-from / valid-to range.

**What does change scheduling is the frequency setting:**

- A checklist with a real frequency (Daily … Yearly) shows that frequency under its name in the mobile checklist list, together with a marker you can tap to see the deadline — "the checklist needs to be completed by <date>", or for a weekday-specific weekly checklist, "the checklist can only be completed on <weekday>". Recurring checklists are also the ones counted in the dashboard's overdue / not-completed tracking.
- A checklist set to **N/A** shows no frequency and no deadline marker in the mobile list, and it is left out of the dashboard's planned-vs-completed and not-completed tracking entirely. It is never "overdue". This is MCL's real ad-hoc checklist: run it when you need it.

So: if what you want is a genuinely ad-hoc check that nobody is chased about, that is a checklist with frequency N/A — not a checklist run with Reason "Special".

## What this means for reports

The Routine / Special reason follows the finished inspection into reporting:

- **On the finished report.** The report has a **Format** line showing Routine or Special, alongside the report ID, dates, inspector and rating. In the downloadable/printed report this is translated into the report's language. In the dashboard's on-screen report view it may appear in German even when the rest of the page is in your language.
- **In the mobile app's history.** The list of past reports shows the reason on each entry, and it appears again when you open a single past report.
- **In the dashboard's Data Analysis charts.** The filter bar has a **Report type** dropdown offering **Routine check** and **Special inspection**. You must pick one; the charts show only inspections of that type. It opens on *Routine check*, so special inspections are hidden from those charts until you switch the dropdown. If a chart looks emptier than you expect, this dropdown is the first thing to check.
- **Not in the report list.** The dashboard's main list of completed reports does not show a Routine/Special column, and it cannot be filtered by it. To see the reason for a given inspection, open the report itself.
- **Not in the notification e-mails.** The e-mail sent out when an inspection is finished names the store and the checklist and links to the report; it does not state whether the inspection was routine or special.

## Things that depend on your setup

- Which frequencies your checklists use, and whether any are set to N/A, is entirely your company's configuration — MCL does not impose a pattern.
- Whether you can change a checklist's frequency, and which stores and charts you can see, depends on your role.
- Reason wording follows the app or dashboard language (for example Routine/Special, Rutina/Especial, Routinekontrolle/Sonderkontrolle) but always means the same two options.

# REVIEWER EVIDENCE

- Status: documented
- Platform: both
- Title: Special vs. routine inspections: what MCL actually distinguishes, in scheduling and in reports
- Summary: MCL has two unrelated mechanisms that map onto the question. (a) A per-run **Reason** label (Routine/Special) chosen on the mobile start screen, stored on the run header, normalised server-side to canonical German values on finish, and surfaced in the report header ("Format"), in mobile history/report detail, and as a mandatory single-select filter in the dashboard's Data Analysis charts. (b) A per-checklist **Periodicity** (N/A, Daily…Yearly) set in the dashboard wizard, which is what actually drives due windows, the mobile deadline marker, and planned-vs-executed / not-completed tracking — checklists with periodicity `NONE` are explicitly excluded from all of that. I verified that no scheduling, completion or compliance query filters on the run type: `crh_type` appears in the dashboard's statistics layer only once, as a passthrough projection, and all type filtering is confined to the indicators/chart queries. Confidence is high on every claim below; each is a direct read of this commit.
- Evidence:
  - `app-android/MCLNET/Resources/values-en/strings.xml` (L120-L126) — the Reason label and its two option strings ("REASON:", "Routine", "Special"); confirms user-facing wording and that Weather is a sibling field.
  - `app-android/MCLNET/Activities/RunCheckListActivityPhone.cs` (L372-L480) — the start screen builds the Reason label plus a two-option segmented control; `SelectOne` defaults to 0 (Routine); no visibility gate on the Reason control (contrast: Weather is set `Invisible` on phone at L501, L528).
  - `app-android/MCLNET/Activities/RunCheckListActivityPhone.cs` (L860-L935) — on **Start checklist**, `SelectOne` maps to `checkListRun.Type = "Routinekontrolle" | "Sonderkontrolle"`; nothing else in the run header depends on it. Same pattern in `app-android/MCLNET/Activities/RunCheckListActivity.cs` (L729-L740).
  - `app-ios/CheckList.iOS/ViewControllers/RunCheckListVC.cs` (L226-L252, L317, L342-L347, L511-L563) — iOS equivalent: iPad uses a `UISegmentedControl` (segment 0 preselected), iPhone uses two tappable images with `type = true` (Routine) as the default; iOS sends the *localized* string as the type.
  - `api/CheckListService/Controllers/ChekListControllerV8.cs` (L686-L706) — `SetHeaderDefaultFormat` normalises de/es/en spellings of Routine/Special to the canonical German values; called on finish at L620. Note: an unrecognised value normalises to empty string.
  - `web/CheckListWebV3/Utils/ReportUtilsV4.cs` (L88-L122, L146-L158) — printed/exported report: `Report_Format` caption plus `GetFormat(lang, crh_type)` translating to Routinekontrolle/Rutina/Routine and Sonderkontrolle/Especial/Special. V4 is the live path (`web/CheckListWebV3/Pages/CheckListReportPage.aspx.cs` L338; `web/CheckListWebV3/Pages/CheckListReportWithFilterPage.aspx.cs` L520).
  - `web/CheckListWebV3/Pages/CheckListReportWithFilterPage.aspx.cs` (L157-L158) — on-screen report detail assigns `crh_type` to the Format label **raw**, without `GetFormat` translation → the German-value caveat in the draft.
  - `web/CheckListWebV3/Pages/CheckListReportPage.aspx` (L334-L342) — the completed-reports grid has a `Format` column but it is `Visible="false"`; hence "not in the report list".
  - `web/CheckListWebV3/Pages/ChartType1.aspx` (L186-L199) — Data Analysis "Report type" combo with exactly two items, values `Routinekontrolle` / `Sonderkontrolle`; same control in ChartType2–5 and Chart1 (Chart1's caption is hardcoded "Audit Type").
  - `web/CheckListWebV3/Pages/ChartType1.aspx.cs` (L41-L53, L145-L203) — `cmbAuditType.SelectedIndex = 0` on first load (Routine is first item), and the selected value is passed as `TypeId` into the chart query on every load.
  - `web/CheckListWebV3/Database/Query.Indicators.cs` (L459-L482) — `GetChart1Ranking` hard-filters `crh.crh_type == TypeId`; 39 such occurrences in this file, i.e. all Data Analysis charts are single-type.
  - `web/CheckListWebV3/App_GlobalResources/Resources.resx` (L706-L720) — user-facing captions "Report type", "Routine check", "Special inspection".
  - `web/CheckListWebV3/Database/Query.Estadisticas.cs` (L25-L34) — executed-checklist counting filters on company/date/market only, never on type.
  - `web/CheckListWebV3/Database/Query.Estadisticas.cs` (L1276-L1292, L1516-L1518, L905-L909, L934-L938, L1697-L1700, L2716-L2720, L3515-L3516) — planned/not-executed logic repeatedly and explicitly excludes checklists with `chl_period_id` null or `"NONE"`; this is the real recurring-vs-ad-hoc split.
  - `web/CheckListWebV3/Database/Query.CheckLists.cs` (L722-L769) — the full periodicity option list offered in the UI: NONE (shown as "N/A"/"N.Z."), DAILY, WEEKLY, MONTHLY, QUARTERLY, BIANNUAL, YEARLY.
  - `web/CheckListWebV3/Pages/CheckListWizardPage.aspx.cs` (L49-L54, L128-L137, L344-L391) — checklist setup: periodicity is required to save, plus valid-from/valid-to, optional from/to times, and a weekday for WEEKLY.
  - `api/CheckListService/Controllers/ChekListControllerV8.cs` (L3213-L3290) — the mobile checklist feed: filtered by valid-from/valid-to against today; for periodicity ≠ `NONE` it computes the current period window, for `NONE` it blanks the periodicity and uses the checklist's own validity dates; `Executed` is set if **any** run exists in the window — no type condition.
  - `app-android/MCLNET/Common/AdapterCheckList.cs` (L107-L206, L251-L296) — the mobile list only renders a frequency label and the tappable deadline marker when `Periodicity` is non-empty (so N/A checklists show neither); the marker text comes from the "needs to be completed by"/"can only be completed on" strings.
  - `app-android/MCLNET/Resources/values-en/strings.xml` (L63-L71) — those deadline message strings and the Daily/Weekly/Monthly/Quarterly/Yearly labels.
  - `app-android/MCLNET/Fragments/HomeFragment.cs` (L248-L270) — starting a checklist is blocked outside its from/to time window with a message; this gate runs before the Reason screen, so it is type-independent.
  - `app-android/MCLNET/Common/AdapterRecyclerViewReports.cs` (L56-L64) and `app-android/MCLNET/Activities/ReportDetailActivityPhone.cs` (L265-L273) — Android history list and past-report detail both display the reason.
  - `app-ios/CheckList.iOS/ViewControllers/CheckListHistoryVC.cs` (L496-L519, L769-L795) and `app-ios/CheckList.iOS/ViewControllers/ReportDetailVM.cs` (L246-L250) — iOS equivalents.
  - `api/CheckListService/Controllers/ChekListControllerV8.cs` (L1477-L1513, L2370-L2440) — completion e-mail: `crh_type` is selected into the mail view model but the subject/body composed by `GetMailInfoSendGrid` never uses it (the only type-in-subject line is commented out, `api/CheckListService/Controllers/ChekListControllerV7.cs` L1745) → "not in the notification e-mails".
  - `web/CheckListWebV3/Pages/ChartType1.aspx.cs` (L18-L24, L223-L276) — role-gated company/store scoping in the analysis pages, supporting the role/permission caveat.
- Uncertainties:
  - Whether the Reason can be changed after an inspection is finished: I found no edit path in the app, dashboard or API, but that is an argument from absence, so the draft only states where the choice is made rather than asserting it is immutable.
  - Whether the mobile Reason control is ever hidden by company configuration: no flag gating it in the paths I read (phone and tablet, plain and department flows), but I did not exhaustively read the department-checklist start screens (`RunCheckListActivityPhoneDep.cs`, `DepartmentChecklistPhone.cs`) — they declare the same `SelectOne` field, so the pattern looks identical.
  - Reason wording is localised per device/dashboard language and the API normalises to canonical German values; a company using an unexpected language or an old client could in principle store a value that normalises to empty. Android's history display treats anything that is not the routine value as "Special" — an empty value would therefore read as Special. Too edge-case and internal to put in the draft.
  - The dashboard has several report-viewing pages (with and without filters) and several report-rendering generations; I verified the current one (V4) and the two pages that call it, but did not audit older/unreferenced report pages, so exactly which page a given customer's menu opens is not something I assert.
  - How a given company actually uses the Routine/Special label operationally (e.g. whether they treat "Special" as an incident check) is policy, not code, and is left to the reader.
  - Abstain reason: n/a