# Ragie spike — side-by-side evaluation

- Partition: `mcl_spike`  |  Answer model: `gpt-4o`  |  rerank=on
- Compared against current pipeline: no (ragie-only)

Judge each pair: is the Ragie answer at least as correct, grounded, and complete as the current one? Note any multimodal wins (image links retrieved).

## 1. How do I create a checklist?
_topic=checklists · lang=en · expects_visual=True · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Creating
Checklists
In this tutorial, you will learn how to use the MCL Checklist Wizard to create new check...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
How do I analyze specific questions or answers in the dashboard?
Go to Data Analysis. - Filter by market, checklist, and time peri...
    score=0.091 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
How do I create a new user in the system in the dashboard?
To add users: - Open the Users tab and click + to create a new profile....
    score=0.083 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Open the Checklists section.
Click + to create a new checklist or click the Pencil Icon to change one.
Link checklists to specific...
```

**Ragie answer:**

To create a checklist using the MCL Checklist Wizard, follow these steps:

1. Click on **Checklists** on the left side of your screen to view your existing checklists and their configurations [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
2. Click the **+** icon to launch the Checklist Wizard [Source: MCL Visual Guide Checklist.pdf].
3. In the first panel, name your checklist and set the start-and-end dates for when the checklist should be active. You can also set the periodicity (e.g., Daily, Weekly) or select "N/A" for it to be always active [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
4. Configure the start-time and end-time for checklist completion, if needed. You can also require an electronic signature and allow users to select/deselect departments [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
5. In the second panel, select which user roles will have access to this checklist. Roles not selected will not see the checklist in their MCL Apps [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
6. In the third panel, choose the departments the checklist will cover. Note that selecting a department does not automatically include all its questions in the checklist [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
7. Preview your checklist in the final panel. Once satisfied, click **Complete** to finalize the checklist [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

These steps will guide you through creating a new checklist in the MCL system.

---

## 2. How do I edit an existing checklist?
_topic=checklists · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
    score=0.100 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
Confirm that each recurrence period aligns with business needs.
How can I restrict or allow department selection for checklists?
I...
    score=0.091 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
    score=0.083 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Creating
Checklists
In this tutorial, you will learn how to use the MCL Checklist Wizard to create new check...
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
If checklists, tasks, or reports are not visible:
Go to the Checklists tab.
The app should automatically sync.
3.
What are the di...
```

**Ragie answer:**

To edit an existing checklist in MCL, follow these steps:

1. Navigate to the **Checklists** section on the left side of your screen.
2. Locate the checklist you wish to edit.
3. Click the **Pencil Icon** next to the checklist name to open it in the Checklist Wizard [Source: MCL Visual Guide Checklist.pdf].
4. Once in the Checklist Wizard, you can modify various settings such as the checklist name, periodicity, start and end dates, and role assignments [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
5. After making your changes, ensure to save or complete the checklist to apply the updates [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

Remember, any changes you make will affect how the checklist appears and functions in the MCL App.

---

## 3. How do I delete a checklist?
_topic=checklists · lang=en · expects_visual=False · ragie_chunks=3 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
    score=0.091 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
    score=0.083 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
Confirm that each recurrence period aligns with business needs.
How can I restrict or allow department selection for checklists?
I...
```

**Ragie answer:**

To delete a checklist in MCL, follow these steps:

1. Navigate to the Checklist Overview.
2. Locate the checklist you wish to delete.
3. Click the **Trash Icon** next to the checklist name to remove it permanently [Source: MCL Visual Guide Checklist.pdf].

> Please note that deleting a checklist cannot be undone [Source: MCL Visual Guide Checklist.pdf].

---

## 4. How do I export a checklist to Excel?
_topic=checklists · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.138 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Export checklist, task, and report data to Excel or PDF.
Who can access and customize the dashboard?
Store Managers: Access to ass...
    score=0.100 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
A checklist appears only in stores where both the department and checklist settings match the assigned
markets. - If departments a...
    score=0.091 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
Confirm that each recurrence period aligns with business needs.
How can I restrict or allow department selection for checklists?
I...
    score=0.083 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
```

**Ragie answer:**

To export a checklist to Excel, follow these steps:

1. Navigate to the **Checklists** menu in the MCL Dashboard.
2. Locate the checklist you wish to export.
3. Click the **Export** button next to the checklist. This will generate an Excel spreadsheet of the checklist for review or offline storage [Source: MCL Visual Guide Checklist.pdf].

---

## 5. How do I set how often a checklist repeats (its periodicity)?
_topic=checklists · lang=en · expects_visual=True · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.167 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
Confirm that each recurrence period aligns with business needs.
How can I restrict or allow department selection for checklists?
I...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
lists.
How do I edit a task in the dashboard after it’s created?
Open a task marked as Not Started and select Edit.
Only the origi...
    score=0.100 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
MCL Checklist Wizard – FAQ Guide
How do I access the Checklist Wizard in MCL?
To access the Checklist Wizard, open the Checklists...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Use Recurring Checklist Status to monitor due dates and completions for daily or scheduled checklists.
What KPIs does the dashboar...
    score=0.091 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Creating
Checklists
In this tutorial, you will learn how to use the MCL Checklist Wizard to create new check...
    score=0.077 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Select N.A. (Not Applicable) to make it always available.
Set signature and department options.
What does “Start scoring the check...
```

**Ragie answer:**

To set how often a checklist repeats, you need to configure its recurrence or periodicity during the checklist setup. Here’s how you can do it:

1. **Access the Checklist Wizard**: Open the Checklists tab from the main menu and click the **+** icon to create a new checklist [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

2. **Set Recurrence**: In the Scheduling section, choose a recurrence option such as Daily, Weekly, or Custom. This determines how often the checklist must be completed [Source: Mcl Checklist Faq Revised.pdf].

3. **Always Available Option**: If you want the checklist to be available at all times, select N.A. (Not Applicable) for the start and end dates [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

Make sure that the recurrence period aligns with your business needs to ensure the checklist is completed as required [Source: Mcl Checklist Faq Revised.pdf].

---

## 6. How do I configure the email settings in the Checklist Wizard?
_topic=checklists · lang=en · expects_visual=True · ragie_chunks=3 · image_links=yes_

**Retrieved chunks:**
```
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
How do I configure checklist recurrence in the dashboard?
In General → Recurring, choose from: N.A., Daily, Weekly, Monthly, Quart...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Select N.A. (Not Applicable) to make it always available.
Set signature and department options.
What does “Start scoring the check...
    score=0.077 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
```

**Ragie answer:**

To configure the email settings in the Checklist Wizard, follow these steps:

1. During checklist creation, navigate to **E-Mail Settings**.
2. Enter up to three email addresses, separated by commas, to receive the checklist reports.
3. Assign roles that should automatically receive the reports by changing the setting from “No” to “Yes” under Assigned [Source: Mcl Dashboard Faq Revised.pdf].

This setup allows you to specify who will receive the reports generated from the checklist, ensuring that the right individuals or roles are informed.

---

## 7. Where do I find the checklists section in the dashboard?
_topic=navigation · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.162 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
A checklist appears only in stores where both the department and checklist settings match the assigned
markets. - If departments a...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Open the Checklists section.
Click + to create a new checklist or click the Pencil Icon to change one.
Link checklists to specific...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
optional photos or PDFs. - Ensure the sender is configured as a market contact to publish successfully.
How do I assign reports to...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Use Recurring Checklist Status to monitor due dates and completions for daily or scheduled checklists.
What KPIs does the dashboar...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
markets, departments, or users. - Save the configuration for future checklists.
How do I create checklist groups in the dashboard...
```

**Ragie answer:**

To find the Checklists section in the MCL Dashboard, select the **Checklists** tab from the main menu. This section allows you to edit, delete, copy, export, or archive checklists [Source: Mcl Dashboard Faq Revised.pdf].

---

## 8. Where can I see completed checklists and delivered tasks?
_topic=navigation · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.174 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
highlighting due dates, completions, and open cycles.
What do you see in the Task Overview section of the dashboard?
Total open an...
    score=0.143 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Use Recurring Checklist Status to monitor due dates and completions for daily or scheduled checklists.
What KPIs does the dashboar...
    score=0.100 doc=Visual Guide Checklist Tasks MCL APP.pdf links=yes
      Visual Guide Checklist Tasks MCL APP.pdf
Photo is optional unless required by task configuration
Once the responsible user, start date, due date, and task are s...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
optional photos or PDFs. - Ensure the sender is configured as a market contact to publish successfully.
How do I assign reports to...
    score=0.083 doc=MCL Visual Guide Dashboard Entry page.pdf links=yes
      MCL Visual Guide Dashboard Entry page.pdf
Recurring Checklists by Status: Each bar shows per store the status of recurring checklists in this period (e.g. this...
```

**Ragie answer:**

To view completed checklists and delivered tasks in the MCL Dashboard, you can follow these steps:

1. **Completed Checklists:**
   - Navigate to the **Checklists** tab from the main menu. Here, you can view all completed checklist submissions. You can also filter these by user, date, or checklist type [Source: Mcl Dashboard Faq Revised.pdf].

2. **Delivered Tasks:**
   - Switch to the **Tasks** tab to view task-related charts and statistics. This section will display the status of tasks, including those that are completed. The **Status of Tasks with Due Date** graph shows tasks that are due within the selected period, with completed tasks indicated in green [Source: MCL Visual Guide Dashboard Entry page.pdf].

These sections will help you track and manage your completed checklists and tasks effectively.

---

## 9. How do I navigate the MCL Dashboard?
_topic=navigation · lang=en · expects_visual=False · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.200 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
MCL Dashboard – FAQ Guide
What is displayed in the MCL Dashboard and how do I navigate it?
The MCL dashboard provides an overview...
    score=0.091 doc=MCL Visual Guide Tasks.pdf links=yes
      MCL Visual Guide Tasks.pdf
The following MCL Visual Guide is based on MCL Web UI, the following is not repeatable through MCL APP UI.
Main Menu → Task Overview...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Open the Checklists section.
Click + to create a new checklist or click the Pencil Icon to change one.
Link checklists to specific...
    score=0.083 doc=MCL Visual Guide Data Analysis.pdf links=yes
      MCL Visual Guide Data Analysis.pdf
The following MCL Visual Guide is based on MCL Web UI, the following is not repeatable through MCL APP UI.
Main Menu → Data A...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
optional photos or PDFs. - Ensure the sender is configured as a market contact to publish successfully.
How do I assign reports to...
    score=0.077 doc=MCL Visual Guide Dashboard Entry page.pdf links=yes
      MCL Visual Guide Dashboard Entry page.pdf
The following MCL Visual Guide is based on MCL Web UI, the following is not repeatable through MCL APP UI.
Main Menu (...
```

**Ragie answer:**

To navigate the MCL Dashboard, you can use the side menu, which allows you to switch between different modules and monitor key performance indicators such as completed checklists or assigned tasks [Source: Mcl Dashboard Faq Revised.pdf]. When you log in, you are taken directly to the MCL Dashboard. If you are using a smaller screen, you can access the full navigation menu by clicking the hamburger icon (three horizontal lines) in the top-left corner of the screen [Source: MCL Visual Guide Tasks.pdf].

The navigation panel on the right provides access to all management and analysis modules of the MCL system. Depending on your role, certain dashboard elements and menu sections may be partially available or completely restricted [Source: MCL Visual Guide Data Analysis.pdf].

---

## 10. How do I complete a task on a tablet?
_topic=tasks · lang=en · expects_visual=False · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.191 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
How do I complete a task?
There are two methods:
Swipe Method: Complete directly if no photo/comment is required.
Note Method: Op...
    score=0.183 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Yes. You can create a task directly from any checklist question.
1.
Tap the Add Task icon (bottom of the question view).
2.
Enter...
    score=0.168 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Alternatively, use the Microphone Icon to dictate a comment (online mode required).
Red-bordered fields indicate mandatory commen...
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
If checklists, tasks, or reports are not visible:
Go to the Checklists tab.
The app should automatically sync.
3.
What are the di...
    score=0.077 doc=Visual Guide Tasks MCL APP.pdf links=yes
      Visual Guide Tasks MCL APP.pdf
LEVEL 3C – Task Closing
[Screen Title]: Task Closing [Parent Screen]: Tasks Overview [Child Screen]: Task Overview
Task Closing
T...
    score=0.071 doc=Visual Guide Tasks MCL APP.pdf links=yes
      Visual Guide Tasks MCL APP.pdf
Creating new tasks
LEVEL 3B – CREATE TASK
[Screen Title]: Task creation [Parent Screen]: Tasks Overview [Child Screen]: Task Over...
```

**Ragie answer:**

To complete a task on a tablet using the MCL Mobile App, follow these steps:

1. Go to the **Tasks Overview**.
2. Tap **Note** next to the task you want to complete.
3. Tap the **Note Icon** located at the top right of the screen.
4. Set **Complete Task** to **Yes**.
5. Tap **Add** to finalize the completion.

If you see red borders around any fields, it indicates that a photo or comment is required before you can complete the task [Source: Mcl Mobile App Faq Revised.pdf].

---

## 11. What is the difference between creating a task in a checklist and in the Task Menu?
_topic=tasks · lang=en · expects_visual=False · ragie_chunks=1 · image_links=yes_

**Retrieved chunks:**
```
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Alternatively, use the Microphone Icon to dictate a comment (online mode required).
Red-bordered fields indicate mandatory commen...
```

**Ragie answer:**

Creating a task in a checklist and in the Task Menu involves different processes and contexts:

1. **Creating a Task in a Checklist:**
   - You can create a task directly from any checklist question. To do this, tap the **Add Task** icon at the bottom of the question view. Then, enter the task details such as Market, Description, Assignee, and Due Date. You can also optionally attach a photo. Once you tap **Save**, the task will appear in the task list after the checklist is completed [Source: Mcl Mobile App Faq Revised.pdf].

2. **Creating a Task in the Task Menu:**
   - The context does not provide specific steps for creating a task directly from the Task Menu. However, tasks created here are generally independent of checklists and can be managed separately in the Tasks section, where you can view all tasks you’ve created or that are assigned to you, depending on your role [Source: Mcl Mobile App Faq Revised.pdf].

If you need more detailed steps for creating tasks directly from the Task Menu, please let me know!

---

## 12. What is the difference between a Routine Inspection and a Special Inspection?
_topic=inspections · lang=en · expects_visual=False · ragie_chunks=3 · image_links=yes_

**Retrieved chunks:**
```
    score=0.191 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
You may be prompted to provide an electronic signature (if enabled).
Tap Finish Report to complete the checklist.
The report will...
    score=0.191 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
MCL Mobile App – FAQ Guide
How do I log in to the MCL Mobile App?
After opening the MCL App, enter your login credentials on the...
    score=0.083 doc=Visual Guide Checklist MCL APP.pdf links=yes
      Visual Guide Checklist MCL APP.pdf
Begins the checklist execution
Reason Selection
Routine – Standard scheduled execution
Special – Non-regular or event-based e...
```

**Ragie answer:**

The difference between a Routine Inspection and a Special Inspection in the MCL Mobile App is as follows:

- **Routine Inspection**: This is for recurring, scheduled checklists. It is used for regular audits that happen on a set schedule [Source: Mcl Mobile App Faq Revised.pdf].
- **Special Inspection**: This is for one-time or ad-hoc audits. It is used for inspections that are not part of the regular schedule and occur as needed. This distinction helps categorize checklist types for later data analysis and filtering [Source: Mcl Mobile App Faq Revised.pdf].

---

## 13. What does N.A. mean in a checklist?
_topic=terminology · lang=en · expects_visual=False · ragie_chunks=4 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=Visual Guide Checklist MCL APP.pdf links=yes
      Visual Guide Checklist MCL APP.pdf
Begins the checklist execution
Reason Selection
Routine – Standard scheduled execution
Special – Non-regular or event-based e...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
Select N.A. (Not Applicable) to make it always available.
Set signature and department options.
What does “Start scoring the check...
    score=0.091 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Planned/Actual
Number
How is the photo/comment requirement set?
During question creation, you can define whether a photo and/or c...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
lists.
How do I edit a task in the dashboard after it’s created?
Open a task marked as Not Started and select Edit.
Only the origi...
```

**Ragie answer:**

In a checklist, "N.A." stands for "Not Applicable." It is used when a checklist question cannot be answered, such as when the department in question does not exist. Questions marked as N.A. are excluded from evaluation [Source: Mcl Mobile App Faq Revised.pdf].

---

## 14. Why can't I see my tasks?
_topic=troubleshooting · lang=en · expects_visual=False · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=MCL Visual Guide Tasks.pdf links=yes
      MCL Visual Guide Tasks.pdf
→ Next Navigation Step:
Select “Tasks” from the menu page to open the Tasks Overview.
LEVEL 2 – TASKS SUB-MENU
[Screen Title]: Tasks...
    score=0.100 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Usually, the checklist is not shared with the required roles.
Check checklist permissions or contact your MCL Administrator.
Why...
    score=0.091 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Tap the question you want to copy. The text will automatically be highlighted and copied to the clipboard.
Can I duplicate conten...
    score=0.083 doc=Visual Guide Tasks MCL APP.pdf links=yes
      Visual Guide Tasks MCL APP.pdf
Function
The Markets screen is the central entry point of the mobile app. Users select a market to begin executing checklists.
→...
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Alternatively, use the Microphone Icon to dictate a comment (online mode required).
Red-bordered fields indicate mandatory commen...
```

**Ragie answer:**

If you can't see your tasks, there are a few things you can check to resolve the issue:

1. **Adjust Task Filter Settings:** Make sure your task filters are set correctly. You can filter tasks by Due Date or Created Date, and adjust the date range or filter by Responsible Person, Market, or Status [Source: Mcl Mobile App Faq Revised.pdf].

2. **Check Synchronization:** Ensure your device is online. If the app hasn't synchronized, your tasks might not appear. Try logging out and logging back in to force synchronization [Source: Mcl Mobile App Faq Revised.pdf].

3. **Task Creation Issues:** If a newly created task doesn't appear, it might not have been saved correctly, or the date filter might be incorrect. Double-check these settings [Source: Mcl Mobile App Faq Revised.pdf].

If these steps don't resolve the issue, you might want to contact your MCL Administrator for further assistance.

---

## 15. My checklist isn't syncing — what should I check?
_topic=troubleshooting · lang=en · expects_visual=False · ragie_chunks=2 · image_links=yes_

**Retrieved chunks:**
```
    score=0.200 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Usually, the checklist is not shared with the required roles.
Check checklist permissions or contact your MCL Administrator.
Why...
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Alternatively, use the Microphone Icon to dictate a comment (online mode required).
Red-bordered fields indicate mandatory commen...
```

**Ragie answer:**

To address checklist synchronization issues, you can follow these steps:

1. Ensure your device is online. A stable internet connection is necessary for synchronization [Source: Mcl Mobile App Faq Revised.pdf].
2. Return to the **Checklist Overview**. This can sometimes trigger the synchronization process [Source: Mcl Mobile App Faq Revised.pdf].
3. If synchronization doesn’t start automatically, try logging out of the app and then logging back in. This can help reset the sync process [Source: Mcl Mobile App Faq Revised.pdf].

If these steps do not resolve the issue, you may want to check with your MCL Administrator for further assistance.

---

## 16. How do I select roles in the Checklist Wizard?
_topic=roles · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.191 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
MCL Checklist Wizard – FAQ Guide
How do I access the Checklist Wizard in MCL?
To access the Checklist Wizard, open the Checklists...
    score=0.100 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
    score=0.091 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
Confirm that each recurrence period aligns with business needs.
How can I restrict or allow department selection for checklists?
I...
    score=0.083 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Set start-time and end-time for checklist to be completed within that duration.
Finally, you may select conf...
    score=0.077 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Creating
Checklists
In this tutorial, you will learn how to use the MCL Checklist Wizard to create new check...
```

**Ragie answer:**

To select roles in the Checklist Wizard, follow these steps:

1. During the checklist setup, navigate to the second panel, which is dedicated to Role Assignment.
2. In this panel, you can select which user roles will have access to the checklist in their MCL App.
3. Choose from the available roles: Author, Administrator, and Manager.
4. Only the roles you select will be able to see and interact with the checklist. Roles that are not selected will not have access [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

---

## 17. How do I view and analyze reports in the dashboard?
_topic=reports · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.183 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
How do I analyze specific questions or answers in the dashboard?
Go to Data Analysis. - Filter by market, checklist, and time peri...
    score=0.167 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
highlighting due dates, completions, and open cycles.
What do you see in the Task Overview section of the dashboard?
Total open an...
    score=0.148 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
4
Circulars: Displays company announcements.
How do I filter data by company, market, or time period in the
dashboard?
At the top...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
optional photos or PDFs. - Ensure the sender is configured as a market contact to publish successfully.
How do I assign reports to...
    score=0.083 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
A checklist appears only in stores where both the department and checklist settings match the assigned
markets. - If departments a...
```

**Ragie answer:**

To view and analyze reports in the MCL Dashboard, follow these steps:

1. **Access the Reports Section:** Navigate to the Reports section of the dashboard. Here, you can view summaries by department and export reports as PDF or Excel files [Source: Mcl Dashboard Faq Revised.pdf].

2. **Analyze Response Quality and Compliance Trends:** Use the analytics engine to visualize positive vs. negative responses, the frequency of N/A, and photo/comment counts. This helps identify areas for improvement [Source: Mcl Dashboard Faq Revised.pdf].

3. **Export Data for Further Analysis:** You can export checklist, task, and report data to Excel or PDF for more detailed reporting [Source: Mcl Dashboard Faq Revised.pdf].

4. **Interactive Charts:** Some charts, such as Checklist Not Completed, Recurring Checklist Status, and Tasks by Status, are interactive. Clicking on a bar or segment will open a detailed list of corresponding elements [Source: Mcl Dashboard Faq Revised.pdf].

These tools and features allow you to effectively analyze and interpret the data within your reports.

---

## 18. How do I manage markets?
_topic=markets · lang=en · expects_visual=True · ragie_chunks=4 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=MCL Visual Guide Markets.pdf links=yes
      MCL Visual Guide Markets.pdf
→ Next Navigation Step:
Select “Markets” from the menu page to open the Markets Overview.
LEVEL 2 – MARKETS OVERVIEW PAGE
[Screen T...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
MCL Dashboard – FAQ Guide
What is displayed in the MCL Dashboard and how do I navigate it?
The MCL dashboard provides an overview...
    score=0.091 doc=MCL Visual Guide Markets.pdf links=yes
      MCL Visual Guide Markets.pdf
Trash Icon:
Removes the market (if allowed by role permissions).
Functions Available
1. Create a new market using the “+” icon.
2....
    score=0.083 doc=MCL Visual Guide Markets.pdf links=yes
      MCL Visual Guide Markets.pdf
2. Company – Manage company-level information (global conﬁguration) and create
tags.
3. Markets – Manage stores (create, delete, ed...
```

**Ragie answer:**

To manage markets in the MCL Dashboard, follow these steps:

1. **Navigate to the Markets Section:**
   - From the main navigation menu, select **Markets** to open the Markets Overview page [Source: MCL Visual Guide Markets.pdf].

2. **Create a New Market:**
   - Click the **+** icon to open the market creation form. Here, you can enter details such as the market name, address, and email [Source: MCL Visual Guide Markets.pdf].

3. **Edit an Existing Market:**
   - Use the **Pencil Icon** to open the Market Configuration for the selected market. This allows you to modify market details and configurations [Source: MCL Visual Guide Markets.pdf].

4. **Assign or Modify Departments:**
   - Ensure each market is assigned to one or more departments. This is crucial as departments determine which checklists appear in the app [Source: MCL Visual Guide Markets.pdf].

5. **(Re)assign Tags:**
   - If using tags, assign them to markets for more detailed filtering and analysis in dashboard charts [Source: MCL Visual Guide Markets.pdf].

6. **Delete Markets:**
   - If necessary, use the **Trash Icon** to remove markets that are no longer active, provided your role permissions allow it [Source: MCL Visual Guide Markets.pdf].

7. **Batch Upload:**
   - For bulk operations, use the Batch Upload tab to import or update multiple markets at once using an Excel file. This is useful during initial setup or when updating large numbers of stores [Source: MCL Visual Guide Markets.pdf].

By following these steps, you can effectively manage markets within the MCL Dashboard.

---

## 19. How does photo management work in MCL?
_topic=photos · lang=en · expects_visual=True · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.183 doc=MCL Visual Guide Photo Management.pdf links=yes
      MCL Visual Guide Photo Management.pdf
→ Next Navigation Step:
Select “Photo Management” from the menu page to open the Photo Management Overview.
LEVEL 2 – PHOT...
    score=0.100 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Alternatively, use the Microphone Icon to dictate a comment (online mode required).
Red-bordered fields indicate mandatory commen...
    score=0.091 doc=MCL Visual Guide Photo Management.pdf links=yes
      MCL Visual Guide Photo Management.pdf
2. Company – Manage company-level information (global conﬁguration) and create
tags.
3. Markets – Manage stores (create, d...
    score=0.091 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
Red indicates incomplete questions.
The number of unanswered questions appears in brackets next to the department name.
What do t...
    score=0.083 doc=MCL Visual Guide Photo Management.pdf links=yes
      MCL Visual Guide Photo Management.pdf
The following MCL Visual Guide is based on MCL Web UI, the following is not repeatable through MCL APP UI.
Main Menu → Pho...
    score=0.077 doc=MCL Visual Guide Photo Management.pdf links=yes
      MCL Visual Guide Photo Management.pdf
→ Next Navigation Step:
None, stay on the same page. Next area will be focused on the Photo Display Area
LEVEL 3 – PHOTO R...
```

**Ragie answer:**

Photo management in MCL is handled through the MCL Dashboard's Photo Management module. This module allows users to manage and review photos uploaded during checklist completion. Here's how it works:

1. **Accessing Photo Management:**
   - From the MCL Dashboard, select **Photo Management** from the main navigation menu to open the Photo Management Overview [Source: MCL Visual Guide Photo Management.pdf].

2. **Photo Management Overview:**
   - The Photo Management screen displays all photos uploaded from the MCL App. You can filter these photos by market, department, date range, or checklist question [Source: MCL Visual Guide Photo Management.pdf].

3. **Functions of Photo Management:**
   - This module is used to verify checklist results visually, compare photos from specific events or topics across stores, inspect issues, and audit operations by viewing photo submissions [Source: MCL Visual Guide Photo Management.pdf].

4. **Photo Preview Area:**
   - Within the Photo Management page, the Photo Preview area allows you to view selected photos in a large format. You can navigate between photos using left and right arrows, but you cannot open photos in a new tab or enlarge them further [Source: MCL Visual Guide Photo Management.pdf].

5. **Metadata and Navigation:**
   - Below each photo, a metadata bar displays information such as the date submitted, department, checklist question, and the user who completed the checklist. You can also use a thumbnail strip to quickly scan through photos [Source: MCL Visual Guide Photo Management.pdf].

6. **Limitations:**
   - There is no option to edit, download, zoom, or view photos in a separate detail view within this module. Photo downloads can be requested from x2 solutions support [Source: MCL Visual Guide Photo Management.pdf].

This setup allows for efficient management and review of photos related to checklist tasks, ensuring that all visual evidence is easily accessible and organized.

---

## 20. Where do I see data analysis in the dashboard?
_topic=data_analysis · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.174 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
highlighting due dates, completions, and open cycles.
What do you see in the Task Overview section of the dashboard?
Total open an...
    score=0.100 doc=MCL Visual Guide Data Analysis.pdf links=yes
      MCL Visual Guide Data Analysis.pdf
→ Next Navigation Step:
Select “Data Analysis” from the menu page to open the Data Analysis Overview.
LEVEL 2 – DATA ANALYSIS...
    score=0.100 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
A checklist appears only in stores where both the department and checklist settings match the assigned
markets. - If departments a...
    score=0.091 doc=Mcl Dashboard Faq Revised.pdf links=yes
      Mcl Dashboard Faq Revised.pdf
optional photos or PDFs. - Ensure the sender is configured as a market contact to publish successfully.
How do I assign reports to...
    score=0.083 doc=MCL Visual Guide Data Analysis.pdf links=yes
      MCL Visual Guide Data Analysis.pdf
2. Company – Manage company-level information (global conﬁguration) and create
tags.
3. Markets – Manage stores (create, dele...
```

**Ragie answer:**

To access data analysis in the MCL Dashboard, you need to select "Data Analysis" from the main navigation menu. This will open the Data Analysis Overview, where you can choose between two tools: **Result Analysis** and **Comparative Analysis**. These tools allow you to analyze checklist results and compare performance across different markets and departments [Source: MCL Visual Guide Data Analysis.pdf].

---

## 21. How do notifications work across the app and web?
_topic=notifications · lang=en · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.182 doc=Visual Guide Notifications APP & WEB.pdf links=yes
      Visual Guide Notifications APP &amp; WEB.pdf
Visual Guide – MCL Notifications (App & Web)
Introduction
Notifications in MCL ensure users are informed about newl...
    score=0.167 doc=Visual Guide Notifications APP & WEB.pdf links=yes
      Visual Guide Notifications APP &amp; WEB.pdf
Tasks can be completed by swipe
Notification Triggers & Rules (Applies to Push Notifications)
Trigger Types
Event-B...
    score=0.155 doc=Visual Guide Notifications APP & WEB.pdf links=yes
      Visual Guide Notifications APP &amp; WEB.pdf
Example: Creating note in assigned task User clicks “+” icon on task, “Create a note”, enter in task note, upload i...
    score=0.083 doc=Mcl Mobile App Faq Revised.pdf links=yes
      Mcl Mobile App Faq Revised.pdf
From the App: All roles.
Which roles can see all markets?
Company Administration, Executive Management, Auditor, Checklist Manage...
    score=0.077 doc=Visual Guide Notifications APP & WEB.pdf links=yes
      Visual Guide Notifications APP &amp; WEB.pdf
Description
This notification appears in the device’s notification center outside of the MCL app. It informs the us...
```

**Ragie answer:**

Notifications in MCL are designed to keep users informed about tasks and performance insights through both the app and email system. Here's how they work across the app and web:

1. **Push Notifications (Device + In-App):**
   - **Device Push Notifications:** These appear in the device's notification center outside of the MCL app. They inform users about newly assigned or due tasks. Tapping the notification opens the MCL app [Source: Visual Guide Notifications APP & WEB.pdf].
   - **In-App Task Notifications:** When users open the MCL app, they see a dialog informing them about new or due tasks. This dialog includes options to view or dismiss the notification [Source: Visual Guide Notifications APP & WEB.pdf].

2. **Notification Triggers:**
   - Notifications are triggered by events such as task creation or when a task reaches its due date. They are also time-based, appearing at specific times like 8:30 AM and 2:30 PM for new tasks, and 7:00 AM for due tasks [Source: Mcl Mobile App Faq Revised.pdf].

3. **Notification Display:**
   - Outside the app, notifications appear in the device notification center. Inside the app, they appear as pop-ups in key areas like Market Overview, Checklist, Tasks, and Reports [Source: Mcl Mobile App Faq Revised.pdf].

4. **Email Notifications:**
   - Users receive periodic performance summaries via email, which include best and worst scoring questions, performance trends, and aggregated results. These are sent automatically and require role permission [Source: Visual Guide Notifications APP & WEB.pdf].

5. **Checklist Reports:**
   - After completing a checklist, users can choose how to handle the report, such as sending it via email or opening it as a PDF. Clicking a link in the email opens the report in a new browser tab [Source: Visual Guide Notifications APP & WEB.pdf].

These notifications ensure that users are kept up-to-date with their tasks and performance metrics, both in real-time and through scheduled summaries.

---

## 22. How are security and permissions managed?
_topic=security · lang=en · expects_visual=False · ragie_chunks=6 · image_links=yes_

**Retrieved chunks:**
```
    score=0.183 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
2. Company – Manage company-level information (global configuration) and create
tags.
3. Markets – Manage stores (create, delete,...
    score=0.177 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
Phone number
No
Optional contact number.
Role
Yes
Determines permissions and access rights within MCL (e.g., Admin, Store Manager,...
    score=0.168 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
→ Next Navigation Step:
Select “Security” from the menu page to open the Security Menu.
LEVEL 2 – SECURITY SUB-MENU (OVERVIEW)
[Sc...
    score=0.138 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
The following MCL Visual Guide is based on MCL Web UI, the following is not repeatable through MCL APP UI.
Main Menu → Security Me...
    score=0.091 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
Description:
The Users section is used to create and maintain user accounts.
Typical User Actions:
Create User – Add a new user ac...
    score=0.083 doc=MCL Visual Guide Security.pdf links=yes
      MCL Visual Guide Security.pdf
Controls whether a user has access to sensor monitoring and supervision data. If unchecked, the user may have the Sensors menu but...
```

**Ragie answer:**

In MCL, security and permissions are managed through the **Security** menu in the MCL Web interface. This menu allows you to control user access across the entire MCL system by managing users and configuring roles [Source: MCL Visual Guide Security.pdf].

Here's how it works:

1. **Users Management**: 
   - You can create, edit, and delete user accounts. Each user is assigned a role that determines their permissions and access rights within MCL [Source: MCL Visual Guide Security.pdf].
   - User details include fields like name, email, username, and optional contact numbers. A password is required when creating a new user [Source: MCL Visual Guide Security.pdf].

2. **Roles Configuration**:
   - Roles define permission profiles that control what features a user can access and which menu sections are visible to them. You can view existing roles, edit role configurations, and save changes to apply new permissions [Source: MCL Visual Guide Security.pdf].
   - Permissions control not only menu visibility but also feature access. If a permission is unchecked, the corresponding menu item may be hidden from the user [Source: MCL Visual Guide Security.pdf].

To manage these settings, navigate to the **Security** menu from the main navigation page, and then select either **Users** or **Roles** to proceed with the specific configurations [Source: MCL Visual Guide Security.pdf].

---

## 23. Wie erstelle ich eine Checkliste?
_topic=checklists · lang=de · expects_visual=True · ragie_chunks=5 · image_links=yes_

**Retrieved chunks:**
```
    score=0.100 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Creating
Checklists
In this tutorial, you will learn how to use the MCL Checklist Wizard to create new check...
    score=0.091 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
You can change the order of the questions by using icons. the
Select a department from the panel to select t...
    score=0.077 doc=Mcl Checklist Faq Revised.pdf links=yes
      Mcl Checklist Faq Revised.pdf
MCL Checklist Wizard – FAQ Guide
How do I access the Checklist Wizard in MCL?
To access the Checklist Wizard, open the Checklists...
    score=0.071 doc=Creating Checklists EN v5 05.22.2020_compressed.pdf links=yes
      Creating Checklists EN v5 05.22.2020_compressed.pdf
Set start-time and end-time for checklist to be completed within that duration.
Finally, you may select conf...
    score=0.067 doc=MCL Visual Guide Checklist.pdf links=yes
      MCL Visual Guide Checklist.pdf
Action Item:
“+” Icon – Opens the Checklist Wizard to create a new checklist.
Pencil Icon – Opens an existing checklist for editi...
```

**Ragie answer:**

Um eine Checkliste in MCL zu erstellen, folge diesen Schritten:

1. **Öffne den Checklisten-Tab** im Hauptmenü, um den Checklist Wizard zu starten [Source: Mcl Checklist Faq Revised.pdf].
2. **Klicke auf das **+**-Symbol**, um eine neue Checkliste zu erstellen [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
3. **Gib einen Namen für die Checkliste ein** und fülle alle Pflichtfelder aus. Dazu gehören in der Regel der Name der Checkliste sowie Start- und Enddaten, falls zutreffend [Source: Mcl Checklist Faq Revised.pdf].
4. **Lege die Start- und Enddaten fest** oder wähle eine Wiederholungsoption (z.B. täglich, wöchentlich) [Source: Mcl Checklist Faq Revised.pdf].
5. **Wähle "N/A"** für Start- und Enddaten, wenn die Checkliste immer verfügbar sein soll [Source: Mcl Checklist Faq Revised.pdf].
6. **Konfiguriere die Periodizität** der Checkliste, indem du eine Wiederholungsoption unter "Scheduling" auswählst [Source: Mcl Checklist Faq Revised.pdf].
7. **Wähle die Rollen aus**, die Zugriff auf die Checkliste haben sollen. Nur die ausgewählten Rollen können die Checkliste in ihrer MCL App sehen und verwenden [Source: Mcl Checklist Faq Revised.pdf].
8. **Wähle die Abteilungen aus**, die die Checkliste abdecken soll. Die Auswahl der Abteilungen bestimmt, in welchen Stores die Checkliste verfügbar ist [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].
9. **Ordne die Fragen innerhalb der Checkliste**, indem du die Fragen nach Abteilungen auswählst und die Reihenfolge mit den Pfeilen anpasst [Source: Mcl Checklist Faq Revised.pdf].
10. **Überprüfe die Checkliste** im letzten Panel und klicke auf "Complete", um die Erstellung abzuschließen [Source: Creating Checklists EN v5 05.22.2020_compressed.pdf].

Diese Schritte helfen dir, eine neue Checkliste im MCL-System zu erstellen und zu konfigurieren.

---

## 24. Was bedeutet N.Z. in einer Checkliste?
_topic=terminology · lang=de · expects_visual=False · ragie_chunks=1 · image_links=yes_

**Retrieved chunks:**
```
    score=0.091 doc=MCL Visual Guide Data Analysis.pdf links=yes
      MCL Visual Guide Data Analysis.pdf
Description:
The Result Analysis in the Data Analysis tab compares single question results across di(cid:431)erent stores, id...
```

**Ragie answer:**

In einer Checkliste steht "N.Z." für "Nicht Zutreffend", was gleichbedeutend mit "Not Applicable" ist. Diese Antwortoption wird verwendet, wenn eine Frage in der Checkliste auf eine bestimmte Situation nicht anwendbar ist [Source: MCL Visual Guide Data Analysis.pdf].

---
