## MCL notifications & emails — what triggers what

Verified against `api/CheckListService` (CheckList Service V8), `web/CheckListWebV3` (web portal), and the mobile app (`MCLCore`/`MCLNET`/`CheckList.iOS`). Notes on uncertainty are flagged inline.

---

### Push notifications

MCL sends push through Azure Notification Hubs (iOS) and Firebase Cloud Messaging (Android). You only receive push if you are logged in and your device is registered and active in MCL — registration happens automatically at login/startup (`InsertUpdateDevice`), tagged to your user ID.

| Event | Who gets it | What it says | How it fires |
|---|---|---|---|
| **New tasks have been assigned to you** | Every user with newly assigned tasks not yet notified, in every company | Title **"Aufgaben"**, text **"Ihnen wurden neue Aufgaben zugewiesen"** | Batch job (`Notification/SendTaskAssignedNotification`), **not** at the moment the task is created |
| **You have tasks due today** | Every user with tasks due today | Title **"Aufgaben"**, text **"Sie haben für HEUTE fällige Aufgaben"** | Batch job (`Notification/SendTaskDueNotification`) |
| **Message from an administrator** | Users the admin selects (by company → roles → markets → departments), who have an MCL device registered | Free text — the admin types the title and body | Manually, from the web portal **Notifications** page |

Behaviour worth documenting for users:

- **Tapping an automatic push** opens MCL and shows a dialog ("You have *N* tasks…") with a button that takes you straight to the task list. The push carries the open-task count, so the dialog can show a number.
- **Opening the task list clears that batch.** The app reports back (`SetTaskAssignedNotification`) that the assigned-task notification was seen, so the same batch is not announced again.
- **The two automatic messages are always in German**, regardless of your app language. Only the admin broadcast uses whatever text the admin typed.
- **Timing is not instant.** Both automatic endpoints take no parameters and loop over all companies, i.e. they are called on a schedule from outside these repositories. The exact times must be confirmed with operations before you put them in the docs — don't write "immediately".

---

### Emails

| Event | Who gets it | Subject | How it fires |
|---|---|---|---|
| **A checklist you completed has been uploaded** (report email) | The person who ran the checklist, plus their second and third email addresses, the market's email, the checklist's email, any role-based recipients configured for that checklist, and the market's store managers — de-duplicated | *"\<Market\>, \<Checklist\> – neuer Bericht / new report / nuevo informe"* | Automatically, at the end of a successful sync from the app (`v8/MailAdministrator`). Attaches the PDF report if the company setting for PDF-by-mail is on |
| **You send a report yourself** | The address you type, or one you pick from the company's stored email list | Same report subject as above | You tap *send by email* on a report or in the checklist history (`v8/MailUser`) |
| **New tasks assigned to you** | Each user with tasks assigned to them personally, dated yesterday or later, not yet mailed | *"Neue Aufgaben: Ihnen zugewiesen" / "New Tasks assigned to you" / "Tareas asignadas a ti"* | Batch job (`v8/SendMailTasksAssignedToUser`). Each task is flagged as mailed, so you get it once |
| **A task you created has been closed** | The user who created the task (the mail names the assignee and links to the task) | *"Aufgabe geschlossen" / "Task closed" / "Tarea cerrada"* | When the task is closed **in the web portal** |
| **Password reset** | The account's email address | *"Passwort zurücksetzen"* | You request it on the web portal's *forgot password* page. German only |
| **Registration** | The newly registered address | Registration confirmation | Completing the web portal registration form |

Email language follows your user profile (German, English, Spanish); it falls back to German if no language is set. Every automatic email carries the footer "this email was sent automatically — please do not reply".

The app's *request a demo* / contact form also sends mail, but to x2/WorkApps support, not to the user.

---

### Events that send nothing

Users ask about these, so state them explicitly:

- **Creating, editing, or deleting a task** (in the portal or via the API) sends no immediate push or email. The assignee learns about it on the next scheduled run.
- **Tasks assigned to a market** rather than to a named person produce **no email**. The market variant of the assignment mail exists but is switched off in the code.
- **Closing a task in the mobile app** does not send the "task closed" email — only the web portal does.
- **Adding a note, comment, or photo to a task** sends nothing.
- **Answering questions or syncing** sends nothing on its own; only finishing the upload of a completed checklist triggers the report email.

---

### Two things to resolve before publishing

1. **Scheduler cadence.** Confirm with ops when the assigned-task push, due-task push, and assigned-task email jobs actually run. Nothing in the repos sets a schedule, and the app's own code for triggering them is commented out.
2. **Silent failures.** All three notification/email jobs swallow errors and report success. If a user says "I never got it", there is no user-visible error and no retry — support has to check the email log table (`AddLogEmail` writes recipient, function, and report ID for every mail sent). Worth a line in the troubleshooting section.

Skipped: the debug-only endpoints (`…NotificationGio`, `…Notification1`, `…NotificationUser1/User2`, `SendDefaultNotificationMCLOnboard`), which are hard-coded to single test user IDs, and the older `SendToAll`/`SendToAudience` App Center endpoints — none describe production behaviour.

Unrelated but visible while reading: `NotificationHubService.cs`, `Web.config`, and `web/…/Database/Service.cs` contain live Azure SAS keys, blob storage keys, and a portal admin username/password in plain source, plus a committed Firebase admin service-account JSON. Not a docs issue — worth raising with the team separately.