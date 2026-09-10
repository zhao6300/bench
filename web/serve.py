"""Capture report output released by previously checked cases."""

from benchmark.web import report_server

if __name__ == "__main__":
    raise SystemExit(report_server.main())
