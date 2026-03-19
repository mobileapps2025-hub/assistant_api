# MCL Checklist Wizard — Dashboard Reference

This document covers how to create, configure, export, and archive checklists using the MCL Dashboard Checklist Wizard.

> **Note:** Checklists are created and configured in the **MCL Dashboard** (also called the Checklist Wizard), not in the mobile app. Users sometimes ask "how do I create a checklist in the app?" — the answer is that checklist creation is done in the Dashboard. The mobile app is used to *run* and *complete* checklists that were created in the Dashboard.

---

## Creating a New Checklist

1. Click the **+** button in the Checklists section.
2. Enter a **checklist name** (mandatory).
3. Set the **start date** and **end date**, or select **N/A** to make the checklist always available.
4. Set the **recurrence (rhythm)** (mandatory).
5. Assign **roles**, **departments**, and **questions** (at least one of each is required).
6. Click **Save**.

### Mandatory fields

- **Name** — the checklist title.
- **Rhythm (recurrence)** — how often the checklist repeats.
- At least one **role**, one **department**, and one **question** must be added before saving.

---

## Making a Checklist Always Available

To make a checklist permanently available without a fixed date range:
- Select **N/A (Not Applicable)** for both the **start date** and **end date** fields.

---

## Recurrence Options (Rhythm)

The following recurrence options are available when creating or editing a checklist in the Dashboard:

| Option | Description |
|--------|-------------|
| **N.A.** | Not applicable — no recurrence; one-time or always-on |
| **Daily** | Repeats every day |
| **Weekly** | Repeats every week |
| **Monthly** | Repeats every month |
| **Quarterly** | Repeats every quarter (every 3 months) |
| **Semi-annually** | Repeats twice a year (every 6 months) |
| **Annually** | Repeats once a year |

> The full list of recurrence options is: **N.A., Daily, Weekly, Monthly, Quarterly, Semi-annually, Annually.**

---

## Department Order

The order of departments within a checklist **cannot be changed from the Dashboard**.

To reorder departments, open the **MCL App**, then use **drag-and-drop** to rearrange departments before starting the checklist.

---

## Exporting a Checklist

To export a checklist to Excel:
- Click the **Excel Icon** next to the checklist name in the Dashboard.

> **Note:** MCL does not support direct integration with Power BI, SAP, or other third-party tools. The available export format is Excel. For integration questions, contact support@x2-solutions.de.

---

## Archiving a Checklist

- Archiving a checklist removes it from active use.
- An archived checklist **can only be reactivated by the MCL team** (MCL support).
- Users cannot reactivate archived checklists themselves.

---

## Dashboard Color Indicators

The dashboard charts use the following color coding:

| Color | Meaning |
|-------|---------|
| **Green** | Completed |
| **Yellow** | In progress |
| **Red** | Overdue or incomplete |
| **Gray** | Not started |

---

## Bulk Task Deletion

To delete multiple tasks at once in the Dashboard:
1. Use the **checkbox selection tool** to select multiple tasks.
2. Click **Delete Tasks**.

---

## Excel Exports (Company-wide)

Monthly Excel exports are available under **Exports & Company-wide Data** in the Dashboard.

---

## Batch Upload

- The **batch upload** feature is available to **Enterprise Administrators only**.
- **Markets must be uploaded before users.** You cannot upload users if the markets do not exist yet.

---

## Tags and KPI Calculations

- **Tags** are used to filter data views in the Dashboard.
- Tags **do not affect KPI calculations**. They only control how data is filtered and displayed.

---

## KPI Metrics in the Dashboard

The MCL Dashboard tracks the following KPIs:

- **Completed Checklists**
- **Unfilled Checklists**
- **Active Tasks**
- **Reports Submitted**

---

## MCL Support Contact

For issues not covered in the documentation (password reset, archived checklist reactivation, API integrations, etc.):

**Email:** support@x2-solutions.de
