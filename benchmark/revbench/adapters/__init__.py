"""Out-of-process adapters.

Each adapter runs in its *own* conda environment -- the competitor's
dependencies do not co-install with ``subcellspace_env`` (python 3.9 /
numpy 1.23.5 vs python 3.12 / torch 2.3) -- so an adapter must not import
``revbench``. They are standalone scripts that read a control parquet and write
an embedding dump in the format ``revbench.representations.read_dump``
understands. The ``schema`` field in the sidecar JSON is what keeps the
duplicated writer honest.
"""
