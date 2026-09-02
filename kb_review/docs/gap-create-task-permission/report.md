## What decides whether a user can create a task in MCL

Three things, checked in this order. All three must pass in the web portal.

**1. Your company must have the Tasks module.**
Tasks is a licensed module. If your company isn't subscribed, the **Tasks** menu doesn't appear at all and the task pages return "Sie sind nicht berechtigt, auf diese Seite zuzugreifen". Only a System Administrator bypasses this check. Nothing a company admin can toggle — it's set when the module is added to the company.

**2. Your role must have "Task creation" switched on.**
This is a per-company, per-role checkbox on **Configuration → Roles** ("Task creation" column). It's the actual permission. Defaults when a company is created:

| Role | Task creation by default |
|---|---|
| System Administrator | on |
| Company Administrator | on |
| Store Manager, District Manager, Consultant, Department Manager, Auditor, Team Member | off |

It's a **role-level** setting, not per-user. To give one person the right, either turn it on for their whole role in that company, or move them to a role that has it.

With it off, the **New task** entry in the menu and the **Add user / Add market / Add role** buttons on the task list are hidden. If you still have a role permission to *receive* tasks (assigned-to-user or assigned-to-market), you keep the Tasks section and see your own tasks — you just can't create.

**3. Your role decides where you can create it, not just whether.**
The market picker on the new-task form is filled from your role:

- System Administrator, Company Administrator, Corporate Manager → every market in the company
- Store Manager, District Manager, Consultant, Department Manager → only markets assigned to you
- Everyone else → only the "All" and "All markets assigned to me" entries

If you don't pick recipients, MCL assigns the task to the store manager of the target market automatically.

### The mobile app is different

Raising a task from a checklist run in the MCL app is **not** gated by the "Task creation" role permission. Any signed-in user with a valid session can create one; the service works out who receives it from the creator's role using the same scope rules as above. The role permission only governs the web portal.

---

Two things worth flagging to whoever owns the spec:

- The hard block on the Add-task page only fires when the page is opened without parameters (`act=createnoparams`). Other entry paths fall through to a weaker check that only confirms a role-permission row exists for your role — it doesn't re-test the Task creation flag. In practice enforcement rests on the hidden menu and buttons, so a user with a direct link may reach the form.
- The mobile gap above means "Task creation = off" is not a company-wide guarantee if users have the app.