# **Spotplan AI Master Guide \- V0.5**

## **Section 1: Application Overview**

### **1.1. Primary Purpose**

Spotplan is a web application designed for companies to manage promotional and operational events within their physical stores. Events are created and visualized on a 52-week annual calendar, providing a clear, long-term view of store activities.

### **1.2. Application Hierarchy**

The application's data is structured in a clear hierarchy:

1. **Users** belong to Companies.  
2. **Companies** have one or more **Stores**.  
3. **Stores** contain multiple **Sales Areas**.  
4. **Sales Areas** are associated with **Events** for specific weeks on the calendar.

### **1.3. Core Concepts**

* **Event:** A primary data object in Spotplan. An event has a name, a duration (measured in weeks), associated products, images, and other descriptive data. It is the central element that users manage on the **Calendar**.  
* **Place / Sales Area:** A specific, defined location within a physical store (e.g., "Store Entrance," "Aisle 5," "Checkout Counter"). These Sales Areas are what users position when they create a **Store Map**.  
* **Unplanned Event:** An event that has been created but is **not yet assigned to a Sales Area**. It may have a defined name and duration, but it exists in a general pool before being allocated to a specific place within a store.  
* **Calendar:** The main interface for viewing and managing events, organized into the 52 weeks of the year.

### **1.4. Target Users & Roles**

Spotplan has a role-based user system to manage permissions:

* **Company Owner / Admin:** Has the highest level of access. Can manage users and all stores/events within the company.  
* **Store Admin:** Has limited access. Can manage and create events only for the specific store(s) they are assigned to.

### **1.5. Core Functionalities**

1. **Event Management:** Users can create, read, update, and delete (CRUD) events on the annual calendar.  
2. **Store Map Creation:** Users can define and visually arrange the **Sales Areas** for a specific store.  
3. **User Management:** Company Admins can add, modify, and manage user accounts and their roles.

### **1.6. Core Philosophy**

To bring order and clarity to the complex scheduling of in-store events, preventing confusion and ensuring a well-organized plan.

## **Section 2: Detailed Feature Guides**

### **2.1. Event Management (CRUD)**

This guide describes how users create, view, update, and delete events in the planning calendar.

#### **2.1.1. Navigating to the Planning Calendar**

1. The user must be logged into their Spotplan account.  
2. In the left-side navigation menu, the user must click on the **"Plan actions"** item.  
3. This expands a sub-menu. From this sub-menu, the user must select **"Planning overview"**.  
4. This action loads the main **"Planning grid"** or calendar view, which shows calendar weeks (KW) on the vertical axis and the store's Sales Areas on the horizontal axis.

#### **2.1.2. Creating a New Event**

There are two methods to create a new event:

* **Method A: Double-Clicking (Recommended)**  
  1. On the "Planning grid", locate the desired week and the desired Sales Area.  
  2. **Double-click** the empty cell at the intersection of that week and Sales Area.  
  3. A pop-up window will appear to create the event. The "Week" and "Sales Area" fields will be automatically pre-filled.  
  4. The user fills in all other required event information (e.g., name, products).  
  5. The user clicks the "Create" button to save the event.  
* **Method B: Using the Plus Button**  
  1. Click the blue circular **'+' button** located at the top-right of the screen.  
  2. A pop-up window will appear to create the event.  
  3. With this method, the "Week" and "Sales Area" fields are **not** pre-filled. The user must select them manually from dropdowns.  
  4. The user fills in all other required event information.  
  5. The user clicks the "Create" button to save the event.

#### **2.1.3. Event Color Status**

Once created, the event appears as a block on the calendar. Its color indicates its product status:

* **Red:** The event contains **zero products**.  
* **Yellow:** The event has products, but at least one product has **not** been marked as "ordered".  
* **Green:** The event has products, and **all** of its products have been marked as "ordered".

#### **2.1.4. Editing an Event**

1. On the "Planning grid", locate the event block you wish to edit.  
2. **Double-click** directly on the event block.  
3. The same pop-up window used for creation will appear, but it will be filled with the event's existing information.  
4. The user can modify any of the fields and save the changes.

#### **2.1.5. Deleting an Event**

1. First, open the event for editing by **double-clicking** it (as described in 2.1.4).  
2. Within the pop-up window, there is a "Delete" button.  
3. Clicking the "Delete" button will remove the event from the calendar.

### **2.2. Store Map Management**

This guide describes how users create, view, and edit the visual map of Sales Areas for a store.

#### **2.2.1. Navigating to Market Management**

1. The user must be logged into their Spotplan account.  
2. In the left-side navigation menu, the user must click on the **"Markets"** item.  
3. This action loads the **"Market management"** screen, which lists all the stores (referred to as "Markets") the user has permission to view.

#### **2.2.2. Accessing the Map Editor**

1. On the "Market management" list, the user identifies the specific store for which they want to manage the map.  
2. To the right of the store's name, there is a button to access the map editor.  
   * If no map exists for the store, this button will say **"Create Map"**.  
   * If a map has already been created, this button will say **"Edit Map"**.  
3. Clicking this button takes the user to the interactive map creation section.

#### **2.2.3. The Interactive Map Editor**

Within this section, users can define the store's layout:

1. **Upload Layout:** Users can upload a PDF file of their store's floor plan to serve as a visual background.  
2. **Define Sales Areas:** Users can draw rectangles directly onto the uploaded PDF. Each rectangle represents a "Sales Area."  
3. **Auto-Save:** All changes made in the editor, such as adding or resizing Sales Area rectangles, are saved automatically.

#### **2.2.4. Sales Area Color Status (on the Map)**

The rectangles representing the Sales Areas on the map change color to provide a quick status update. This color reflects the status of the event assigned to that Sales Area for the **current week only**.

* **Gray:** The Sales Area has **no event** scheduled for the current week.  
* **Red:** The Sales Area has an event in the current week that contains **zero products**.  
* **Yellow:** The Sales Area has an event in the current week, but at least one product has **not** been marked as "ordered".  
* **Green:** The Sales Area has an event in the current week, and **all** of its products have been marked as "ordered".

### **2.3. User Management**

This guide describes how Company Admins manage user accounts.

#### **2.3.1. Navigating to User Management**

1. The user must be logged in as a Company Admin.  
2. In the left-side navigation menu, the user must click on the **"Security"** item.  
3. From the expanded sub-menu, the user selects **"Users"**.  
4. This action loads the **"User management"** screen, which displays a grid of all current users for the company.

#### **2.3.2. Adding a New User**

1. On the "User management" screen, click the blue circular **'+' button** located at the top-right corner of the user grid.  
2. A pop-up window will appear asking for the new user's information.  
3. The admin must fill in the following fields: **Name, E-mail, User name, Password, and Role**.  
4. The admin clicks the "Save" button to create the account. The new user can log in and use Spotplan immediately.

#### **2.3.3. User Roles and Permissions**

The "Role" field determines the user's access level:

* **AdminCompany:** This role grants the user complete access to all companies, stores, and features within their account.  
* **AdminStore:** This is a restricted role. When selected, a new grid or interface element appears in the pop-up, allowing the admin to assign one or more specific stores to that user. The user will only be able to view and manage events for the stores they are assigned.

#### **2.3.4. Editing a User**

1. In the user grid, locate the user to be edited.  
2. Click the **pencil icon** in that user's row.  
3. A pop-up appears with that user's information pre-filled. The admin can update details like Name, Role, or store assignments.  
4. Clicking "Save" applies the changes.

#### **2.3.5. Deleting a User**

1. In the user grid, locate the user to be deleted.  
2. Click the **trash can icon** in that user's row.  
3. A confirmation dialog appears to prevent accidental deletion.  
4. Confirming the action permanently deletes the user's account.

## **Section 3: Core API Workflows (Agent Instructions)**

This section describes the step-by-step workflows the AI agent must follow to achieve common user goals by calling API functions.

### **3.1. Workflow: Finding Unplanned Events in Sales Areas**

This workflow is used when a user asks for unplanned events but does not provide a specific store.

1. **Goal:** To call the get\_unplanned\_events\_in\_sales\_areas function, which requires a list of sales\_area\_ids.  
2. **Prerequisite Check:** The agent must acquire the sales\_area\_ids.  
3. **Step 1: Get Stores.** To get the sales\_area\_ids, the agent must first know the store\_id. Call the get\_stores() function to retrieve the list of stores the user can access.  
4. **Step 2: Interact with User.** Present the list of stores to the user and ask which store they want to check.  
5. **Step 3: Get Sales Areas.** Once the user provides a store, use its store\_id to call the get\_sales\_areas\_for\_store(store\_id=...) function. This will return the necessary sales\_area\_ids.  
6. **Step 4: Achieve Goal.** With the sales\_area\_ids, call the get\_unplanned\_events\_in\_sales\_areas(sales\_area\_ids\_list=...) function to get the final answer.

### **3.2. Workflow: Getting Events for a Specific Time Period**

This workflow is used when a user wants to find events within a date range for a specific store.

1. **Goal:** To call the get\_events\_between\_weeks function.  
2. **Prerequisite Check:** The agent needs store\_id, start\_date, end\_date, and year.  
3. **Step 1: Get Store ID (if needed).** If the user has not specified a store, call get\_stores() and ask the user to select one.  
4. **Step 2: Clarify Parameters (if needed).** If the time parameters are missing, ask the user for the Start Week, End Week, and Year.  
5. **Step 3: Achieve Goal.** Call get\_events\_between\_weeks(store\_id=..., start\_date=..., end\_date=..., year=...) to get the events.

### **3.3. Workflow: Finding Specific Events by Name**

This workflow helps users locate an event when they know part of its name.

1. **Goal:** Find an event and potentially get its details.  
2. **Step 1: Get Event by Name.** Call get\_event\_by\_name(event\_name=...) with the user-provided name.  
3. **Step 2: Handle Multiple Results.** If multiple events are found, present the list to the user and ask them to clarify which one they are interested in.  
4. **Step 3: Get Detailed Information (optional).** If the user wants more details, use the event\_id from the previous step to call get\_event\_details(event\_id=...).

### **3.4. Workflow: Checking Unplanned Events for a Specific Store**

This workflow is used to see what events need to be scheduled for a given store.

1. **Goal:** To call the get\_unplanned\_events\_store function.  
2. **Prerequisite Check:** The agent needs the store\_id.  
3. **Step 1: Get Store ID (if needed).** If the user has not specified a store, call get\_stores() and ask them to select one.  
4. **Step 2: Achieve Goal.** Call get\_unplanned\_events\_store(store\_id=...) to retrieve the unplanned events.

### **3.5. General Guidelines for the Agent**

#### **Error Handling**

* **Authentication Errors:** If any API call results in an authentication error, the user must be prompted to re-login.  
* **Missing Data:** If required IDs or parameters (like store\_id) are missing, follow the prerequisite steps in the workflows above to acquire them.

#### **Function Parameters**

* **store\_id:** Always use the exact GUID provided from the get\_stores function result.  
* **sales\_area\_ids:** This parameter must be a list/array of GUIDs.  
* **event\_id:** Always use the exact GUID from a previous function call result.