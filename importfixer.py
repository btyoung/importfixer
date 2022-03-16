#! /usr/bin/env python
"""
Library to detect unimported modules and optionally add imports to the file
"""
import argparse
import ast
import difflib
import importlib
import os
import re
import sys
from pathlib import Path

from pyflakes import checker, messages

DEFAULT_CONFIG_FILE = "~/.importfixer.txt"
DEFAULT_CONFIG = [
    "os",
    "sys",
    "pickle",
    "csv",
    "copy",
    "fnmatch",
    "glob",
    "json",
    "pprint",
    "re",
    "sqlite3",
    "socket",
    "subprocess",
    "textwrap",
    "scipy.optimize",
    "scipy.linalg",
    "scipy.interpolate",
    "zipfile",
    "numpy as np",
    "from datetime import date, time, datetime",
    "from typing import *",
]


# --------------------------------------------------------------------------------------
#  Main Function
# --------------------------------------------------------------------------------------
def fiximports(content, config=None, filename=None):
    """
    Add and remove imports in a python file

    Args:
        content (str): The python code to analyze
        config (str or [str], optional): The configuration for this run. Can be a file
            with configuration data, or a list of entries. See project docs for details
            on format. Default is from ~/.importfixer.txt if available, and
            DEFAULT_CONFIG if not.
        filename (str, optional): Filename of content. Used for error messages

    Returns:
        str: The updated content with imports added and removed
    """
    config = _get_config(config)

    import_errors = _detect_import_errors(content, config, filename)

    for unused_import in import_errors["unused"]:
        content = remove_import(content, unused_import, filename)

    for required_import in import_errors["required"]:
        content = add_import(content, required_import, filename)

    return content


def _get_config(config):
    """
    Get the config from inputs

    Args:
        config (str or [str] or None): If a string, reads path and converts to outputs.
            If a list of strings, returns self. If None, attempts to read, then returns
            DEFAULT_CONFIG

    Returns:
        [str]: The config entries
    """
    # First default behavior if None
    if config is None:
        try:
            return _get_config(os.path.expanduser(DEFAULT_CONFIG_FILE))
        except OSError:
            return DEFAULT_CONFIG

    # Check if its a file
    if isinstance(config, str):
        with open(config, "r") as filehandle:
            # Get all lines stripping comments and white space
            lines = [line.split("#", 1)[0].strip() for line in filehandle]

        return [line for line in lines if line != ""]

    # Otherwise it must be a list of strings
    return config


def _detect_import_errors(content, config, filename=None):
    """
    Find import errors in python code

    Args:
        content (str): The python code to analyze
        config (dict): The configuration for this config
        filename (str): Filename of content

    Returns:
        {'required": [str], "unused": [str]}: Lists of required new imports,
            and existing unused imports
    """
    # Parse and get flakes
    tree = ast.parse(content, filename or "<unknown>")
    flakes = checker.Checker(tree, filename or "<unknown>")

    extra_imports, new_imports = [], []
    for msg in flakes.messages:
        if isinstance(msg, messages.UnusedImport):
            extra_imports.append(msg.message_args[0])
        elif isinstance(msg, messages.UndefinedName):
            name = msg.message_args[0]
            imp = _find_import(name, config)
            if imp is not None:
                new_imports.append(imp)

    return {"required": new_imports, "unused": extra_imports}


def _find_import(name, config):
    """
    Determine if a specific name is defined in the configuration

    Args:
        name (str): The name of the item that is "undefined"
        config (dict): Configuration dictionary

    Returns:
        str or None: Import string if appropriate, or None if no match
    """
    for entry in config:
        if entry not in _PARSED_ENTRIES:
            _PARSED_ENTRIES[entry] = _parse_config_entry(entry)

        if name in _PARSED_ENTRIES[entry]:
            return _PARSED_ENTRIES[entry][name]

    return None


_PARSED_ENTRIES = {}

_FROM_PATTERN = re.compile(r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+(.+)")
_ALIAS_PATTERN = re.compile(
    r"([a-zA-Z_][a-zA-Z0-9_.]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)"
)
_BASIC_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.]*)")


def _parse_config_entry(entry):
    """
    Parse a single line in the configuration to get a mapping of names to
    import string definitions.

    Args:
        entry (str): The entry

    Returns:
        dict or str: Mapping from name string or string
    """
    entry = entry.strip()

    # First check from x import y
    if match := _FROM_PATTERN.match(entry):
        import_from, names = match.groups()
        if names == "*":
            try:
                names = list(
                    importlib.import_module(import_from).__all__
                )  # __dict__.keys())
            except ImportError:
                names = []

            return {name: f"from {import_from} import {name}" for name in names}

        # Recurse to handle 'from x import y as z'
        #  If recursed entry raises error, skip ti now to raise error on whole thing
        try:
            return {
                key: f"from {import_from} import {value}"
                for name in names.split(",")
                for key, value in _parse_config_entry(name).items()
            }
        except ValueError:
            pass

    # Then check "x as Y"
    elif match := _ALIAS_PATTERN.match(entry):
        alias, name = match.groups()
        return {name: f"{alias} as {name}"}

    # Then plain "x"
    elif match := _BASIC_PATTERN.match(entry):
        return {match.groups()[0]: match.groups()[0]}

    raise ValueError(f"Invalid settings entry {entry!r}")


# --------------------------------------------------------------------------------------
#  File Modification functions
# --------------------------------------------------------------------------------------
# These are also "public" functions, since you may want a tool to add or remove as
# needed.
def remove_import(content, name, filename=None):
    """
    Update content with extraneous imports removed

    Args:
        content (str): The python code
        name (str): The name of the imported module/object to remove
        filename (str, optional): Filename of content

    Returns:
        str: Updated content
    """
    tree = ast.parse(content, filename or "<unknown>")

    words = name.split()
    if len(words) == 3 and words[1] == "as":
        name, asname = words[0], words[2]
    elif len(words) == 1:
        name, asname = words[0], None
    else:
        raise ValueError("Got incorrect input for name")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports = [(n.name, n.asname) for n in node.names]
            if (name, asname) not in imports:
                continue

            if len(imports) == 1:
                # Remove entire entry
                #  Find start/end of node
                start, end = _loc(content, node)

                # Look for other items around it
                #  ";  {import  x}  "
                #  "{import x;  }"
                #  "{import  #   \n}"
                leading_semicolon_match = re.search(r"\s*;\s*$", content[:start])
                trailing_semicolon_match = re.search(r"^\s*;\s*", content[end:])
                trailing_line_end_match = re.search(r"^\s*(#.*)?\n?", content[end:])
                if leading_semicolon_match:
                    span = leading_semicolon_match.span()
                    start -= span[1] - span[0]
                elif trailing_semicolon_match:
                    span = trailing_semicolon_match.span()
                    end += span[1] - span[0]
                elif trailing_line_end_match:
                    span = trailing_line_end_match.span()
                    end += span[1] - span[0]

                # Truncate interior
                return content[:start] + content[end:]
            else:
                # Remove just the name of interest
                stm_start, stm_end = _loc(content, node)
                stm = content[stm_start:stm_end]

                if asname is None:
                    trailing_comma_match = re.search(rf"{name}\s*,\s*", stm)
                    leading_comma_match = re.search(rf"\s*,\s*{name}", stm)
                else:
                    trailing_comma_match = re.search(
                        rf"{name}\s+as\s+{asname}\s*,\s*", stm
                    )
                    leading_comma_match = re.search(
                        rf"\s*,\s*{name}\s+as\s+{asname}", stm
                    )

                if trailing_comma_match:
                    span = trailing_comma_match.span()
                    newstm = stm[: span[0]] + stm[span[1] :]
                elif leading_comma_match:
                    span = leading_comma_match.span()
                    newstm = stm[: span[0]] + stm[span[1] :]
                else:
                    raise ValueError("You should never see this")

                return content[:stm_start] + newstm + content[stm_end:]
    else:
        raise ValueError(f"Import {name} not present")


def add_import(content, name, filename=None):
    """
    Update contents with new import added

    Args:
        content (str): The python code
        name (str): The name of the imported module/object to be added
        filename (str, optional): Filename of content

    Returns:
        str - Updated content
    """
    # Parse input
    if name.startswith("from"):
        _, import_from, _, import_name = name.split(maxsplit=3)
    else:
        import_from = None

    # Load tree, and find either import-from to add to, or an appropriate line
    #  for insertion
    tree = ast.parse(content, filename or "<unknown>")
    lines = content.splitlines()

    # First go past initial comments
    try:
        last_import_line = next(
            idx for idx, line in enumerate(lines) if not line.strip().startswith("#")
        )
    except StopIteration:
        last_import_line = len(lines)

    # Go past docstring
    if (
        len(tree.body) > 0
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        last_import_line = max(last_import_line, tree.body[0].end_lineno)

    for node in tree.body:
        # print(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)

        if (
            import_from is not None
            and isinstance(node, ast.ImportFrom)
            and import_from == node.module
        ):
            return _add_import_to_node(content, node, import_name)

    if import_from is None:
        newline = f"import {name}"
    else:
        newline = name

    lines.insert(last_import_line, newline)
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def _add_import_to_node(content, node, import_name):
    """
    Add a new import from to an existing list

    That is: 'from x import y, z' becomes 'from x import y, z, newitem'.

    Args:
        content (str): Full content of file
        node (ast.ImportFrom): The node of the import from
        import_name (str): The new import name or alias (e.g. 'x as a')

    Returns:
        str: Full content with new import added
    """
    start, end = _loc(content, node)
    substr = content[start : end + 1]

    if len(node.names) == 1:
        match = _alias_pattern(node.names[0]).search(substr)
        insertion_point = start + match.end()
        sep = ", "
    else:
        match0 = _alias_pattern(node.names[-2]).search(substr)
        match1 = _alias_pattern(node.names[-1]).search(substr)
        sep = substr[match0.end() : match1.start()]
        insertion_point = start + match1.end()

    return "".join(
        (
            content[:insertion_point],
            sep,
            import_name,
            content[insertion_point:],
        )
    )


def _alias_pattern(alias):
    ":returns: CompiledRegex: Regex pattern from ast alias"
    if alias.asname is None:
        return re.compile(alias.name)
    else:
        return re.compile(rf"{alias.name}\s+as\s+{alias.asname}")


def _loc(content, node):
    """
    Find the location of a node within ``content``

    Args:
        content (str): The file content
        node (ast.Node): Node to find

    Returns:
        (int, int): Start/end indices of string
    """
    start_line, start_col = node.lineno, node.col_offset
    end_line, end_col = node.end_lineno, node.end_col_offset
    line_lengths = [len(line) for line in content.splitlines(True)]

    idx0 = sum(line_lengths[: start_line - 1]) + start_col
    idx1 = sum(line_lengths[: end_line - 1]) + end_col

    return (idx0, idx1)


# --------------------------------------------------------------------------------------
#  Command Line Implementation
# --------------------------------------------------------------------------------------
def main(argv=None):
    """
    Main command line interface
    """
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", nargs="?", help="Input file. Uses stdin if not provided"
    )
    parser.add_argument(
        "output", nargs="?", help="Output file. Default prints to stdout"
    )
    parser.add_argument("-c", "--config", help="Input config")
    parser.add_argument("-d", "--diff", action="store_true", help="Show difference")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite input. Not compatible with <output>.",
    )
    args = parser.parse_args(argv)

    # Get input
    if args.file is not None:
        filename = args.file
        with open(filename, "r") as fh:
            content = fh.read()
    else:
        filename = "/dev/stdin"
        content = sys.stdin.read()

    # Run Updates
    update = fiximports(content, args.config, Path(filename))

    # Do output
    if args.overwrite:
        if args.output is not None:
            print("Cannot specify <output> and --overwrite")
            sys.exit(1)
        if args.file is None:
            print("Cannot overwrite stdin, provide <input>")
            sys.exit(1)

        output = filename
    else:
        output = args.output

    if output is not None:
        with open(output, "w") as fh:
            fh.write(update)

    if args.diff:
        diffs = difflib.unified_diff(content.splitlines(), update.splitlines())
        print("\n".join(diffs))
    elif output is None:
        print(update)
