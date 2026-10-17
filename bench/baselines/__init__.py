"""Detectors to compare airlock_pii with.

Every module here is a detector that ``evaluate.py --detector <module name>`` can run. It has
one function, ``detect(text)``, which returns the spans it finds as ``(type, start, end)``
tuples. The type is the name of an ``airlock_pii.EntityType``: a baseline maps its own names
to ours, and what it cannot detect it leaves out.
"""
