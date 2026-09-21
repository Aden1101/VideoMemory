"""Progress lines of the rollouts, flushed as they are printed so that the log of a running
job is current and its lines stay in order with the ones torch writes to stderr."""
import functools

log = functools.partial(print, flush=True)
