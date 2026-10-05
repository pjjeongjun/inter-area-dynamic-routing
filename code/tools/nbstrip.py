#!/usr/bin/env python
"""git "clean" filter: strip notebook outputs on the way into the index.

The notebooks carry ~1.4 MB of embedded figures each. That is fine in your
working copy, but committing it makes the repo balloon and the diffs
unreadable. This filter stores a stripped copy in git while leaving the file
on disk exactly as you saved it.

Uses only nbformat, which the capsule and the conda env already provide, so
there is nothing extra to install.

Enable once per clone (run from the repo root):

    git config filter.nbstrip.clean "python code/tools/nbstrip.py"
    git config filter.nbstrip.smudge cat
"""

import sys

import nbformat

DROP_CELL_METADATA = ("execution", "collapsed", "scrolled")


def strip(nb):
    nb.metadata.pop("widgets", None)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        for key in DROP_CELL_METADATA:
            cell.metadata.pop(key, None)
    return nb


def main():
    nb = nbformat.read(sys.stdin, as_version=4)
    nbformat.write(strip(nb), sys.stdout)


if __name__ == "__main__":
    main()
