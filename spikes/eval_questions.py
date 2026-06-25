"""Evaluation question set for the Ragie spike.

Assembled from the FAQ PDFs, the visual guides' `helps_answer` frontmatter, and the
scenarios the current pipeline is tested against. `expects_visual=True` marks questions
where a screenshot should help — used to judge Ragie's multimodal value. A couple of German
items check the bilingual (EN/DE) behaviour the app must preserve.
"""

EVAL_QUESTIONS = [
    # --- Checklists (creation / editing / lifecycle) ---
    {"q": "How do I create a checklist?", "topic": "checklists", "lang": "en", "expects_visual": True},
    {"q": "How do I edit an existing checklist?", "topic": "checklists", "lang": "en", "expects_visual": True},
    {"q": "How do I delete a checklist?", "topic": "checklists", "lang": "en", "expects_visual": False},
    {"q": "How do I export a checklist to Excel?", "topic": "checklists", "lang": "en", "expects_visual": True},
    {"q": "How do I set how often a checklist repeats (its periodicity)?", "topic": "checklists", "lang": "en", "expects_visual": True},
    {"q": "How do I configure the email settings in the Checklist Wizard?", "topic": "checklists", "lang": "en", "expects_visual": True},

    # --- Navigation / Dashboard ---
    {"q": "Where do I find the checklists section in the dashboard?", "topic": "navigation", "lang": "en", "expects_visual": True},
    {"q": "Where can I see completed checklists and delivered tasks?", "topic": "navigation", "lang": "en", "expects_visual": True},
    {"q": "How do I navigate the MCL Dashboard?", "topic": "navigation", "lang": "en", "expects_visual": False},

    # --- Tasks (mobile / tablet specifics) ---
    {"q": "How do I complete a task on a tablet?", "topic": "tasks", "lang": "en", "expects_visual": False},
    {"q": "What is the difference between creating a task in a checklist and in the Task Menu?", "topic": "tasks", "lang": "en", "expects_visual": False},

    # --- Inspections / terminology ---
    {"q": "What is the difference between a Routine Inspection and a Special Inspection?", "topic": "inspections", "lang": "en", "expects_visual": False},
    {"q": "What does N.A. mean in a checklist?", "topic": "terminology", "lang": "en", "expects_visual": False},

    # --- Troubleshooting ---
    {"q": "Why can't I see my tasks?", "topic": "troubleshooting", "lang": "en", "expects_visual": False},
    {"q": "My checklist isn't syncing — what should I check?", "topic": "troubleshooting", "lang": "en", "expects_visual": False},

    # --- Roles & permissions ---
    {"q": "How do I select roles in the Checklist Wizard?", "topic": "roles", "lang": "en", "expects_visual": True},

    # --- Broader corpus (now that the full docs/ set is available) ---
    {"q": "How do I view and analyze reports in the dashboard?", "topic": "reports", "lang": "en", "expects_visual": True},
    {"q": "How do I manage markets?", "topic": "markets", "lang": "en", "expects_visual": True},
    {"q": "How does photo management work in MCL?", "topic": "photos", "lang": "en", "expects_visual": True},
    {"q": "Where do I see data analysis in the dashboard?", "topic": "data_analysis", "lang": "en", "expects_visual": True},
    {"q": "How do notifications work across the app and web?", "topic": "notifications", "lang": "en", "expects_visual": True},
    {"q": "How are security and permissions managed?", "topic": "security", "lang": "en", "expects_visual": False},

    # --- Bilingual (DE) ---
    {"q": "Wie erstelle ich eine Checkliste?", "topic": "checklists", "lang": "de", "expects_visual": True},
    {"q": "Was bedeutet N.Z. in einer Checkliste?", "topic": "terminology", "lang": "de", "expects_visual": False},
]
