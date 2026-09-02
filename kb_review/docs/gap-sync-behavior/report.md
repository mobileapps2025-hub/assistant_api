Grounded in `app-android/MCLCore/ViewModels/BaseVM.cs` (shared by both platforms — `app-ios/MCLCore` has the same file), `MCLNET/Fragments/HomeFragment.cs`, `MCLCore/ViewModels/ReportVM.cs`/`UserVM.cs`, and the server endpoints `v8/Synchronization` + `v8/ServicesToDownload`.

# Synchronization

## What gets synchronized

Sync runs in two directions, and they are separate operations.

**Download (server → device)** — one call, `Synchronization`, which returns your working set:

| Downloaded | Used for |
|---|---|
| Your role | What you're allowed to see and do |
| Stores / markets assigned to you | Market list on the home screen |
| Checklists (with their periodicity window: daily, weekly, monthly, quarterly, biannual, yearly) | Which checklists are currently due |
| Departments | Department-based checklists |
| Questions and their answer options | The checklist itself |
| Users assigned to your markets | Assigning tasks to colleagues |

This is a **full replace**, not a merge: each local table is emptied and rewritten (`BaseVM.getData`). The device is only marked as synced if *every* table is rewritten successfully — otherwise nothing is marked and the app tries again next time.

**Upload (device → server)** — finished checklist runs stored on the device with status *Finished, not yet sent*. For each run, strictly in order (`BaseVM.sincronizar`):

1. Answer photos — up to 3 per question
2. Task photos — up to 3 per task created during the run
3. The signature image
4. The run header (store, dates, signature)
5. Tasks created during the run, plus who each is assigned to
6. The answers
7. Notification email to the administrator

Photos are deduplicated: before uploading, the app asks the server whether that file already exists, so a partially-uploaded run doesn't re-send what already arrived. Each upload gets one automatic retry after 1.5 seconds.

**Not synchronized — these need a live connection and nothing is queued:** signing in, report history and report PDFs/emails, the Tasks tab (creating, editing, completing standalone tasks), and notes. Offline, those screens tell you to connect instead of storing anything.

## When it runs

There is no background sync and no timer. Upload is attempted:

- every time the market/home screen loads — i.e. on app start and whenever you return to it
- the moment you finish a checklist, if you're online (`ReportVM.FinishCheckList`)
- when you tap **Synchronize Now** on the User tab
- when you log out, if you're online and have pending work

After each upload attempt the app asks the server for its last-modified timestamp (`ServicesToDownload`) and downloads only if either:

- the last successful full download wasn't today, or
- the server's timestamp is newer than the device's last sync

In practice: at least once a day, plus whenever an administrator changes checklists, questions, or your store assignments.

If a download is needed but doesn't complete, the market list is replaced by the screen reading *"To continue, please connect your device to the internet…"* with a **Synchronize data** button. Until that download succeeds, no checklist can be started.

## If sync fails

The design rule is that **nothing local is deleted until the server has confirmed everything.**

- Local photos and database rows for a run are removed only when all images uploaded *and* the header, tasks, and answers all posted successfully. Any failure at any step leaves the run intact on the device, complete with photos and signature, for the next attempt.
- It's all-or-nothing per run: if one photo still fails after its retry, the app deliberately skips the data upload for that run rather than create a partial record on the server.
- One failing run doesn't block the others — the app continues through the queue.
- You get a dialog listing the checklists still waiting, with a count. The User tab shows a **Reports not yet synchronized** counter and a **Synchronize Now** button.
- Finishing a checklist offline saves it and tells you it will be sent automatically the next time you open the app, or manually from the User tab. You can keep running more checklists in the meantime.
- Failures are reported to the server as diagnostic log entries when a connection is available, so support can see which run and which step failed.

**One case where offline work is lost: logging out.** Logout attempts a sync only if you're online, then deletes the local database whether or not that sync succeeded (`UserVM.LogOutData`). Anything not yet on the server is gone and cannot be recovered. Users must confirm the pending-reports counter reads 0 before logging out — this belongs in the docs as an explicit warning, not a footnote.

---

Two things I could not confirm from the code and would need to check before publishing: whether re-posting a run header after a mid-sequence failure creates a duplicate server-side or updates in place, and whether a run left pending across a reference-data download can still be sent if its checklist was meanwhile deleted by an administrator.