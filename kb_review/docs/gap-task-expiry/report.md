## What happens to a task after its due date passes

Short answer: **nothing changes automatically.** The due date in MCL is a target, not a deadline the system enforces.

### Status
The task keeps the status it already had. MCL tasks only ever have these statuses:

| Status | Meaning |
|---|---|
| Not started | Assigned, no work recorded yet |
| In progress | Someone has added a note, photo, or partial update |
| Completed | Closed by the assignee |
| Draft | Created in the web portal but not yet sent out |

There is **no "Overdue" or "Expired" status.** A task that was *Not started* yesterday is still *Not started* today. Only a person can change a task's status.

### Notifications
- **On the due date itself**, assignees with the mobile app get a push notification: *"You have tasks due TODAY."* Tapping it opens the task list filtered to today's due, not-yet-completed tasks.
- **After the due date passes, no further notification is sent.** There is no overdue reminder, no repeat notice, and no escalation to a manager or task creator.
- The only other task emails are sent at the start and end of a task's life: one when the task is assigned to you, and one to the task's creator when you complete it.

### Visibility
The task **stays fully visible and fully actionable**, indefinitely:

- It remains in your open task list. Task lists are never filtered by due date — only *completing* a task moves it out of the open list and into the completed/closed view.
- Lists are sorted by due date, oldest first, so past-due tasks sit at the top. (Tasks with no due date sort last and show as *No date*.)
- You can still add notes and photos, and still close the task, exactly as before. Nothing locks or expires.

### How to find overdue tasks
Because there is no automatic flag, you spot overdue work yourself:

- **Mobile app:** the task list is already sorted oldest-due-first; anything above today's date is overdue. You can also filter by a due-date range.
- **Web portal:** the task grid sorts and filters by due date the same way.

> **Note:** the web portal's *Delays / Expired* report covers **checklists**, not tasks. There is currently no equivalent overdue-tasks report.

---

Two things worth flagging to the product owner before this ships:

- The backend already calculates a past-due flag for every task and sends it to the mobile app, but no screen uses it — the row highlighting that would have shown it is commented out in the iOS app and was never built in Android. If you want overdue tasks to look different, the data is already there.
- The "due today" push list is built by a database stored procedure (`sp_Get_TaskUserDueNotifications`) that isn't in the application code, so I could not verify from source whether it *only* picks up same-day tasks or also re-notifies for dates already passed. The notification wording says "TODAY", and the app's tap-through filters strictly on today — but someone with DB access should confirm the procedure matches before we publish "no further notification is sent" as fact.