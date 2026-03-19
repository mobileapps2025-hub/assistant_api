# MCL Roles & Permissions Reference

This document is the authoritative reference for all MCL user roles and their permissions.

> **Important — Canonical Role Names:** Some older documents use abbreviated or translated role names. Always use the canonical names below. Mapping of old names to correct names:
> - "Company Management" or "Company manager" → **Company Administration**
> - "Management" or "Executive" → **Executive Management**
> - "District Management" → **Regional Management**
> - "Market Management" → **Store Management**
> - "Head of Department" → **Department Management**
> - "Examiner" or "Auditor (old)" → **Auditor**
> - "Team" → **Team Member**

---

## The 9 MCL Roles

The MCL system has exactly nine roles:

1. **Company Administration** — Full organisational control over all markets, users, checklists, tasks, and configuration.
2. **Executive Management** — View-only access to all markets, tasks, and checklists. Can answer checklists and view all tasks but can only manage tasks they created. Cannot receive tasks.
3. **Regional Management** — Access restricted to their assigned markets/regions.
4. **Store Management** — Access restricted to their own store(s). Can assign tasks only to roles within their own store.
5. **Department Management** — Identical permissions to Section Management except: can only view and manage their own tasks (not all market tasks).
6. **Section Management** — Identical permissions to Department Management except: can view all tasks in the market.
7. **Checklist Management** — Can create, edit, and delete checklists in the Dashboard. Cannot receive tasks. Cannot view Reports or Data Analysis tabs.
8. **Auditor** — Can create, edit, and delete checklists. Can see all markets. Cannot receive tasks.
9. **Team Member** — Standard user role. Can receive tasks and answer checklists.

---

## Permissions Matrix

### Checklist Management (Dashboard)
- **Can create, edit, delete checklists:** Company Administration, Auditor, Checklist Management
- **Can only view checklists:** Executive Management (view-only)

### Task Reception (who can be assigned tasks)
- **Can receive tasks:** Company Administration, Regional Management, Store Management, Department Management, Section Management, Team Member
- **Cannot receive tasks:** Auditor, Executive Management, Checklist Management

### Task Creation
- **From the Dashboard:** All roles except Checklist Management
- **From the MCL App:** All roles

### Market Visibility
- **Can see all markets:** Company Administration, Executive Management, Auditor, Checklist Management
- **Restricted to assigned markets:** Regional Management, Store Management, Department Management, Section Management, Team Member

### Reports & Analytics (Dashboard)
- **Can view Reports tab:** All roles except Checklist Management
- **Can view Data Analysis tab:** All roles except Checklist Management

### Configuration
- **Full configuration access:** Company Administration only

---

## Key Role Distinctions

### Executive Management vs Company Administration
- **Company Administration:** Full control — manages users, markets, departments, checklists, tasks, and configuration.
- **Executive Management:** View-only access. Can answer checklists and view all tasks but can only manage tasks they personally created. Cannot receive tasks assigned by others.

### Department Management vs Section Management
Both roles have identical permissions with one difference in task visibility:
- **Section Management:** Can view **all tasks** in the market.
- **Department Management:** Can only view and manage **their own tasks**.

### Store Management task assignment
Store Managers can assign tasks to other roles in the MCL app, but only to roles that are assigned to their own store. They cannot assign tasks to users from other stores.
