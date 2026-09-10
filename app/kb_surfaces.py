"""Which surface a knowledge unit describes, and which surfaces are searchable.

The web platform was rebuilt, so its OLD documents are no longer accurate. They are tagged
`web_legacy` and kept out of search (archived, not deleted) until the new web map replaces
them. The new web map is tagged `web` and is what web questions will search. The mobile app
was not rebuilt, so `app` documents stay live. `both` documents (notifications, sync) hold
real app value and are not the source of the outdated-screen problem.
"""

APP = "app"
WEB = "web"                 # the new web map — the only web truth (built from the live app)
WEB_LEGACY = "web_legacy"   # old web documents — archived, never searched
BOTH = "both"

# No platform router yet, so search behaves as the app view: app + shared, never any web.
SEARCHABLE_BY_DEFAULT = frozenset({APP, BOTH})

# A web question searches ONLY the new web map. The shared documents are app-oriented
# (device sync, notifications) and some hide app-only procedures, so they never answer a web
# question. Until the map exists, a web question finds nothing and MarieClaire says so.
WEB_SEARCH = frozenset({WEB})

# An app question searches the app documents plus the shared ones.
APP_SEARCH = frozenset({APP, BOTH})

# The web map is small and curated, so a web answer requires a genuinely relevant unit.
# Below this cosine score the map has nothing on the topic and MarieClaire says so, instead
# of stitching an answer from a near-but-wrong screen (e.g. tasks for a checklist question).
WEB_MIN_SCORE = 0.5


def min_score_for(surface: str) -> float:
    return WEB_MIN_SCORE if surface == WEB else 0.0


# Names of screens/flows that exist only in the rebuilt web app. A shared ("both") unit that
# mentions one is really old web content, so it is demoted to web_legacy and kept out of search.
WEB_ONLY_SIGNALS = (
    "checklist wizard", "dashboard", "web portal", "web configuration", "web-configuration",
    "in the web app", "on the web", "web interface", "browser",
)


def classify(document_name: str) -> str:
    name = document_name.lower()

    if "mcl app" in name or "mobile app" in name:
        return APP
    if "synchronization" in name or "routine inspections" in name or "special vs" in name:
        return APP

    if "app & web" in name or "app and web" in name or "mobile & web" in name:
        return BOTH
    if "user_guide_qa" in name or "user guide qa" in name:
        return BOTH
    if "notifications" in name and "email" in name:
        return BOTH
    if "after its due date" in name:
        return BOTH

    return WEB_LEGACY


def classify_unit(document_name: str, text: str) -> str:
    """Per-unit surface. Shared documents mix app and web; a shared unit that describes a
    web-only screen is old web content and must not be searchable."""
    surface = classify(document_name)
    if surface == BOTH and any(signal in (text or "").lower() for signal in WEB_ONLY_SIGNALS):
        return WEB_LEGACY
    return surface
