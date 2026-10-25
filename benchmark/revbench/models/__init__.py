"""Model variants benchmarked as ablations of SubCellSpace.

Kept under ``src/`` so ``paths.add_src_to_path()`` makes them importable from
every notebook and every script, which is the same problem the repo-root
``CVAE`` module already has: a checkpoint that pickles a model instance can only
be loaded where that instance's class is importable.
"""
