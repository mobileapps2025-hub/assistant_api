"""The surface MCL.Api talks to. The browser never reaches it.

MCL.Api owns the session; it forwards each user turn here with the actor it derived from the
validated cookie, authenticated by a shared service secret. MarieClaire never sees a user
credential on this path.
"""
