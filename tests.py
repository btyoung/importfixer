"""
Tests for importfixerer
"""
import io
import os
from textwrap import dedent

import pytest

from importfixer import (
    DEFAULT_CONFIG,
    _detect_import_errors,
    _find_import,
    _get_config,
    add_import,
    fiximports,
    main,
    remove_import,
)

# --------------------------------------------------------------------------------------
#  File Modification Routines
# --------------------------------------------------------------------------------------
# NOTE: All these routines use the "wrap" fixture that adds combinations of shebang
# lines, docstrings, and post-import code to ensure these work in all possible
# configurations. This is the "wrap" function used to apply the same wrapping code in
# the input and output code.

# d = docstring, s=shebang line, w=trailing whitespace, c=trailing code
@pytest.fixture(params=["", "d", "s", "w", "c", "sdc", "sdw", "sdwc", "sc", "swc"])
def wrap(request):
    pre = []
    post = []
    if "s" in request.param:
        pre.append("#! /usr/bin/env mpython")
    if "d" in request.param:
        pre.extend(['"""', "Docstring here", '"""'])
    if "w" in request.param:
        post.extend(["", ""])
    if "c" in request.param:
        post.extend(["cwd = __name__", "print(cwd)"])

    return lambda content: "\n".join(
        pre + ([dedent(content).rstrip()] if content else []) + post
    )


class TestImportRemoval:
    def test_remove_single_line(self, wrap):
        src = """\
        import os
        import sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_second_line(self, wrap):
        src = """\
        import sys
        import os
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_multi_import_first(self, wrap):
        src = """\
        import os, sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_multi_import_last(self, wrap):
        src = """\
        import sys, os
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(
            wrap(src),
            "os",
        ) == wrap(expected)

    def test_remove_multi_import_middle(self, wrap):
        src = """\
        import sys, os, glob
        print(glob.glob(sys.argv[1]))
        """
        expected = """\
        import sys, glob
        print(glob.glob(sys.argv[1]))
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_import_from(self, wrap):
        src = """\
        from os.path import abspath
        import sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os.path.abspath") == wrap(expected)

    def test_remove_multiple_import_from(self, wrap):
        src = """\
        from os.path import abspath, realpath
        import sys
        print(realpath(sys.argv[1]))
        """
        expected = """\
        from os.path import realpath
        import sys
        print(realpath(sys.argv[1]))
        """
        assert remove_import(wrap(src), "os.path.abspath") == wrap(expected)

    def test_remove_single_line_with_comment(self, wrap):
        src = """\
        import os   # OS utilities
        import sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_semicolon_first(self, wrap):
        src = """\
        import os; import sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_remove_semicolon_last(self, wrap):
        src = """\
        import sys; import os
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os") == wrap(expected)

    def test_import_as(self, wrap):
        src = """\
        import os as o
        import sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os as o") == wrap(expected)

    def test_import_as_multi_import_first(self, wrap):
        src = """\
        import os as o, sys
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os as o") == wrap(expected)

    def test_import_as_multi_import_last(self, wrap):
        src = """\
        import sys, os as o
        print(sys.argv[1])
        """
        expected = """\
        import sys
        print(sys.argv[1])
        """
        assert remove_import(wrap(src), "os as o") == wrap(expected)

    def test_import_as_multi_import_middle(self, wrap):
        src = """\
        import sys, os as o, glob
        print(glob.glob(sys.argv[1]))
        """
        expected = """\
        import sys, glob
        print(glob.glob(sys.argv[1]))
        """
        assert remove_import(wrap(src), "os as o") == wrap(expected)

    def test_not_found(self, wrap):
        src = """\
        import sys
        print(sys.argv[1])
        """
        with pytest.raises(ValueError):
            remove_import(wrap(src), "os")

    def test_remove_from_first(self, wrap):
        src = """\
        from os import getcwd, environ, path

        print(environ.keys())
        """
        expected = """\
        from os import environ, path

        print(environ.keys())
        """
        assert remove_import(wrap(src), "os.getcwd") == wrap(expected)

    def test_remove_from_middle(self, wrap):
        src = """\
        from os import getcwd, environ, path

        print(getcwd())
        """
        expected = """\
        from os import getcwd, path

        print(getcwd())
        """
        assert remove_import(wrap(src), "os.environ") == wrap(expected)

    def test_remove_from_last(self, wrap):
        src = """\
        from os import getcwd, environ, path

        print(environ.keys())
        """
        expected = """\
        from os import getcwd, environ

        print(environ.keys())
        """
        assert remove_import(wrap(src), "os.path") == wrap(expected)

    def test_remove_from_only(self, wrap):
        src = """\
        from os import getcwd

        print('hello world')
        """
        expected = """\

        print('hello world')
        """
        assert remove_import(wrap(src), "os.getcwd") == wrap(expected)

    def test_remove_import_dot(self, wrap):
        src = """\
        import os.path

        print('hello world')
        """
        expected = """\

        print('hello world')
        """
        assert remove_import(wrap(src), "os.path") == wrap(expected)

    def test_remove_from_import_as(self, wrap):
        src = """\
        from os import path as P

        print('hello world')
        """
        expected = """\

        print('hello world')
        """
        assert remove_import(wrap(src), "os.path as P") == wrap(expected)

    def test_trailing_comment(self, wrap):
        # Issue #4
        src = """\
        import numpy

        # x = numpy.array([1, 2, 3])
        """
        expected = """\

        # x = numpy.array([1, 2, 3])
        """
        assert remove_import(wrap(src), "numpy") == wrap(expected)

    def test_trailing_comment_as(self, wrap):
        # Issue #4
        src = """\
        import numpy as np

        # x = np.array([1, 2, 3])
        """
        expected = """\

        # x = np.array([1, 2, 3])
        """
        assert remove_import(wrap(src), "numpy as np") == wrap(expected)


class TestAddImport:
    def test_add_empty(self, wrap):
        src = ""
        expected = "import os"
        assert add_import(wrap(src), "os") == wrap(expected)

    def test_add_to_existing(self, wrap):
        src = """\
        import sys
        from x import y
        """
        expected = """\
        import sys
        from x import y
        import os
        """
        assert add_import(wrap(src), "os") == wrap(expected)

    def test_alias(self, wrap):
        src = """\
        import sys
        """
        expected = """\
        import sys
        import os as O
        """
        assert add_import(wrap(src), "os as O") == wrap(expected)

    def test_new_from(self, wrap):
        src = """\
        import sys
        """
        expected = """\
        import sys
        from os import cwd
        """
        assert add_import(wrap(src), "from os import cwd") == wrap(expected)

    def test_from_single_no_paren(self, wrap):
        src = """\
        import os
        from numpy import array
        """
        expected = """\
        import os
        from numpy import array, linalg
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_from_single_with_paren(self, wrap):
        src = """\
        import os
        from numpy import (array)
        """
        expected = """\
        import os
        from numpy import (array, linalg)
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_from_single_newline(self, wrap):
        src = """\
        import os
        from numpy import (
            array
        )
        """
        expected = """\
        import os
        from numpy import (
            array, linalg
        )
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_from_single_as(self, wrap):
        src = """\
        import os
        from numpy import array as A
        """
        expected = """\
        import os
        from numpy import array as A, linalg
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_from_single_as_weird_spaces(self, wrap):
        src = """\
        import os
        from numpy import array   as  A
        """
        expected = """\
        import os
        from numpy import array   as  A, linalg
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_import_as(self, wrap):
        src = """\
        import os
        from numpy import array
        """
        expected = """\
        import os
        from numpy import array, linalg as LA
        """
        assert add_import(wrap(src), "from numpy import linalg as LA") == wrap(expected)

    def test_add_from_multiple_no_paren(self, wrap):
        src = """\
        import os
        from numpy import array, diag as D, sum
        """
        expected = """\
        import os
        from numpy import array, diag as D, sum, linalg
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_from_multiple_with_paren(self, wrap):
        src = """\
        import os
        from numpy import (array, diag as D, sum)
        """
        expected = """\
        import os
        from numpy import (array, diag as D, sum, linalg)
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_from_multiple_with_single_newline(self, wrap):
        src = """\
        import os
        from numpy import (
            array, diag as D, sum
        )
        """
        expected = """\
        import os
        from numpy import (
            array, diag as D, sum, linalg
        )
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_from_multiple_with_all_newlines(self, wrap):
        src = """\
        import os
        from numpy import (
            array,
            diag as D,
            sum
        )
        """
        expected = """\
        import os
        from numpy import (
            array,
            diag as D,
            sum,
            linalg
        )
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_from_multiple_with_all_newlines_trailing_comma(self, wrap):
        src = """\
        import os
        from numpy import (
            array,
            diag as D,
            sum,
        )
        """
        expected = """\
        import os
        from numpy import (
            array,
            diag as D,
            sum,
            linalg,
        )
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_add_import_from_semicolon(self, wrap):
        src = """\
        from numpy import array; import os
        """
        expected = """\
        from numpy import array, linalg; import os
        """
        assert add_import(wrap(src), "from numpy import linalg") == wrap(expected)

    def test_import_from_short_pattern_existing(self, wrap):
        src = "from mod import m"
        expected = "from mod import m, var"
        assert add_import(wrap(src), "from mod import var") == wrap(expected)

    def test_import_from_short_pattern_existing2(self, wrap):
        src = "from mod import oldvar, m"
        expected = "from mod import oldvar, m, var"
        assert add_import(wrap(src), "from mod import var") == wrap(expected)


# --------------------------------------------------------------------------------------
#  Main Script Utilities
# --------------------------------------------------------------------------------------
#  These confirm all the inputs get processed as expected
class TestFindImport:
    config = [
        "os",
        "sys",
        "numpy as np",
        "from os.path import *",
        "from typing import List, Dict as TDict",
    ]

    def test_simple(self):
        assert _find_import("os", self.config) == "os"

    def test_none(self):
        assert _find_import("glob", self.config) is None

    def test_alias(self):
        assert _find_import("np", self.config) == "numpy as np"

    def test_import_from_star(self):
        assert _find_import("abspath", self.config) == "from os.path import abspath"

    def test_import_from_star_noall(self, mocker):
        mocker.patch("importfixer._PARSED_ENTRIES", {})
        osall = os.path.__all__
        try:
            del os.path.__all__
            assert _find_import("abspath", self.config) == "from os.path import abspath"
        finally:
            os.path.__all__ = osall

    def test_import_from_spec(self):
        assert _find_import("List", self.config) == "from typing import List"

    def test_import_from_spec_alias(self):
        assert _find_import("TDict", self.config) == "from typing import Dict as TDict"


class TestDetection:
    config = ["os", "sys", "getpass", "numpy as np", "from datetime import *"]

    def test_missing_import(self):
        src = """\
        args = sys.argv[1:]
        print(args)
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": ["sys"],
            "unused": [],
        }

    def test_extra_import(self):
        src = """\
        import os
        print('Hello world')
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": [],
            "unused": ["os"],
        }

    def test_multiple_missing(self):
        src = """\
        with open(os.path.abspath(sys.argv[1]), 'w') as fh:
            fh.write('Hello world')
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": ["os", "sys"],
            "unused": [],
        }

    def test_multiple_extra_import(self):
        src = """\
        import os, sys
        print('Hello world')
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": [],
            "unused": ["os", "sys"],
        }

    def test_mixed(self):
        src = """\
        import os, sys
        print(f'Hello {getpass.getuser()}')
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": ["getpass"],
            "unused": ["os", "sys"],
        }

    def test_missing_not_in_config(self):
        src = """\
        prit("x")
        """
        assert _detect_import_errors(dedent(src), self.config) == {
            "required": [],
            "unused": [],
        }

    def test_extra_rename(self):
        assert _detect_import_errors("import os as o", self.config) == {
            "required": [],
            "unused": ["os as o"],
        }

    def test_extra_rename_extra_spaces(self):
        assert _detect_import_errors("import  os  as   o", self.config) == {
            "required": [],
            "unused": ["os as o"],
        }

    def test_missing_rename(self):
        assert _detect_import_errors("print(np.zeros(10))", self.config) == {
            "required": ["numpy as np"],
            "unused": [],
        }

    def test_missing_import_from(self):
        assert _detect_import_errors("datetime.now()", self.config) == {
            "required": ["from datetime import datetime"],
            "unused": [],
        }

    def test_extra_import_from(self):
        assert _detect_import_errors("from datetime import datetime", self.config) == {
            "required": [],
            "unused": ["datetime.datetime"],
        }

    def test_extra_import_submodule(self):
        assert _detect_import_errors("import os.path", [*self.config, "os.path"]) == {
            "required": [],
            "unused": ["os.path"],
        }

    def test_extra_from_import_as(self):
        assert _detect_import_errors("from os import path as P", self.config) == {
            "required": [],
            "unused": ["os.path as P"],
        }

    def test_from_in_function(self):
        src = """\
        import os
        import sys

        def printme():
            print(sqrt(int(sys.argv[1])))
        """
        config = ["os", "sys", "from numpy import array, sqrt, pi"]
        assert _detect_import_errors(dedent(src), config) == {
            "required": ["from numpy import sqrt"],
            "unused": ["os"],
        }

    def test_in_function(self):
        src = """\
        import os

        def printme():
            print(np.sqrt(int(sys.argv[1])))
        """
        config = ["os", "sys", "numpy as np"]
        assert _detect_import_errors(dedent(src), config) == {
            "required": ["numpy as np", "sys"],
            "unused": ["os"],
        }


class TestConfigParsing:
    def test_default_nofile(self, tmp_path, mocker):
        cfgfile = tmp_path / ".importfixer.txt"
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        config = _get_config(None)
        assert config == DEFAULT_CONFIG

    def test_default_withfile(self, tmp_path, mocker):
        cfgfile = tmp_path / ".importfixer.txt"
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        cfgfile.write_text(
            dedent(
                """\
                os
                numpy as np
                """
            )
        )

        config = _get_config(None)
        assert config == ["os", "numpy as np"]

    def test_regular_entries(self):
        config = _get_config(["os", "numpy as np", "sys"])
        assert config == ["os", "numpy as np", "sys"]

    def test_readfile(self, tmp_path):
        cfgfile = tmp_path / "myconfig.txt"
        cfgfile.write_text(
            dedent(
                """\
                os
                sys
                numpy as np
                from typing import *
                from scipy import linalg as LA, special
            """
            )
        )

        assert _get_config(str(cfgfile)) == [
            "os",
            "sys",
            "numpy as np",
            "from typing import *",
            "from scipy import linalg as LA, special",
        ]

    def test_readfile_comments(self, tmp_path):
        cfgfile = tmp_path / "myconfig.txt"
        cfgfile.write_text(
            dedent(
                """\
                os
                sys

                numpy as np
                from typing import *

                from scipy import linalg as LA, special
            """
            )
        )

        assert _get_config(str(cfgfile)) == [
            "os",
            "sys",
            "numpy as np",
            "from typing import *",
            "from scipy import linalg as LA, special",
        ]

    def test_readfile_blanks(self, tmp_path):
        cfgfile = tmp_path / "myconfig.txt"
        cfgfile.write_text(
            dedent(
                """\
                # System data
                os
                sys

                # Others
                numpy as np
                from typing import *   # Some extra comments

                from scipy import linalg as LA, special   # I like LA
            """
            )
        )

        assert _get_config(str(cfgfile)) == [
            "os",
            "sys",
            "numpy as np",
            "from typing import *",
            "from scipy import linalg as LA, special",
        ]


# --------------------------------------------------------------------------------------
#  Integration Tests
# --------------------------------------------------------------------------------------
def test_fiximports(wrap):
    src = """\
    import os
    import sys

    print(np.sqrt(int(sys.argv[1])))
    """
    config = ["os", "sys", "numpy as np"]
    expected = """\
    import sys
    import numpy as np

    print(np.sqrt(int(sys.argv[1])))
    """

    assert fiximports(dedent(wrap(src)), config) == dedent(wrap(expected))


def test_fiximports_ignore_unconfigured_mod(wrap):
    src = """\
    import os
    import sys
    import unused_mod

    print(np.sqrt(int(sys.argv[1])))
    """
    config = ["os", "sys", "numpy as np"]
    expected = """\
    import sys
    import unused_mod
    import numpy as np

    print(np.sqrt(int(sys.argv[1])))
    """

    assert fiximports(wrap(dedent(src)), config) == dedent(wrap(expected))


def test_func_import_inuse(wrap):
    src = """\
    import os

    def myfunc():
        import sys
        print(np.sqrt(int(sys.argv[1])))
    """
    config = ["os", "sys", "numpy as np"]
    expected = """\
    import numpy as np

    def myfunc():
        import sys
        print(np.sqrt(int(sys.argv[1])))
    """
    assert fiximports(dedent(wrap(src)), config) == dedent(wrap(expected))


def test_func_import_notinuse(wrap):
    src = """\
    import os
    import sys

    def myfunc():
        import unused_mod
        print(np.sqrt(int(sys.argv[1])))
    """
    config = ["os", "sys", "numpy as np"]
    expected = """\
    import sys
    import numpy as np

    def myfunc():
        import unused_mod
        print(np.sqrt(int(sys.argv[1])))
    """
    assert fiximports(dedent(wrap(src)), config) == dedent(wrap(expected))


# --------------------------------------------------------------------------------------
#  Command Line Tests
# --------------------------------------------------------------------------------------
class TestCommandLine:
    src = dedent(
        """\
        #! /usr/bin/env python
        "Compute a square root"
        import os
        import sys

        print(np.sqrt(int(sys.argv[1])))
        """
    ).strip()
    expected = dedent(
        """\
        #! /usr/bin/env python
        "Compute a square root"
        import sys
        import numpy as np

        print(np.sqrt(int(sys.argv[1])))
        """
    ).strip()

    @pytest.fixture
    def cfgfile(self, tmp_path):
        cfg_file = tmp_path / "import-config.txt"
        cfg_file.write_text(
            dedent(
                """\
                os
                sys
                numpy as np
                from typing import *
                from datetime import date, time, datetime
                """
            )
        )
        return cfg_file

    def test_stdin_stdout(self, cfgfile, capsys, mocker):
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))
        mocker.patch("sys.stdin", io.StringIO(self.src))
        main([])

        captured = capsys.readouterr()
        assert captured.out.strip() == self.expected

    def test_read_stdout(self, cfgfile, capsys, mocker, tmp_path):
        inputfile = tmp_path / "input.py"
        inputfile.write_text(self.src)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))
        main([str(inputfile)])

        captured = capsys.readouterr()
        assert captured.out.strip() == self.expected

    def test_read_write_different_file(self, cfgfile, mocker, tmp_path):
        inputfile = tmp_path / "input.py"
        outfile = tmp_path / "output.py"
        inputfile.write_text(self.src)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        main([str(inputfile), str(outfile)])

        assert inputfile.read_text() == self.src
        assert outfile.read_text() == self.expected

    def test_overwrite(self, cfgfile, mocker, tmp_path):
        inputfile = tmp_path / "input.py"
        inputfile.write_text(self.src)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        main([str(inputfile), "--overwrite"])

        assert inputfile.read_text() == self.expected

    def test_error_output_overwrite(self, cfgfile, tmp_path, capsys, mocker):
        inputfile = tmp_path / "input.py"
        inputfile.write_text(self.src)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        with pytest.raises(SystemExit):
            main([str(inputfile), str(tmp_path / "outfile"), "--overwrite"])

        assert (
            capsys.readouterr().out.strip() == "Cannot specify <output> and --overwrite"
        )
        assert not (tmp_path / "outfile").exists()

    def test_error_stdin_overwrite(self, cfgfile, mocker, capsys):
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))
        mocker.patch("sys.stdin", io.StringIO(self.src))

        with pytest.raises(SystemExit):
            main(["--overwrite"])

        assert (
            capsys.readouterr().out.strip() == "Cannot overwrite stdin, provide <input>"
        )

    def test_diff(self, cfgfile, mocker, capsys):
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))
        mocker.patch("sys.stdin", io.StringIO(self.src))

        main(["--diff"])
        captured = capsys.readouterr().out.strip()

        assert captured != self.expected

    def test_check_updates(self, tmp_path, cfgfile, mocker, capsys):
        inputfile = tmp_path / "input.py"
        inputfile.write_text(self.src)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        with pytest.raises(SystemExit) as exit:
            main([str(inputfile), "--check"])

        captured = capsys.readouterr()

        assert inputfile.read_text() == self.src
        assert captured.out.strip() == ""
        assert captured.err.strip() == ""
        assert exit.value.code == 1

    def test_check_no_updates(self, tmp_path, cfgfile, mocker, capsys):
        inputfile = tmp_path / "input.py"
        inputfile.write_text(self.expected)
        mocker.patch("importfixer.DEFAULT_CONFIG_FILE", str(cfgfile))

        with pytest.raises(SystemExit) as exit:
            main([str(inputfile), "--check"])

        captured = capsys.readouterr()

        assert inputfile.read_text() == self.expected
        assert captured.out.strip() == ""
        assert captured.err.strip() == ""
        assert exit.value.code == 0

    def test_config(self, cfgfile, tmp_path):
        inputfile = tmp_path / "input.py"
        outfile = tmp_path / "output.py"
        inputfile.write_text(self.src)

        main([str(inputfile), str(outfile), "-c", str(cfgfile)])

        assert inputfile.read_text() == self.src
        assert outfile.read_text() == self.expected

    def test_add(self, tmp_path, capsys):
        src = dedent(
            """\
            #! /usr/bin/env python
            "Compute a sqrt"
            import os
            import sys

            print(np.sqrt(int(sys.argv[1])))
            """
        ).strip()

        expected = dedent(
            """\
            #! /usr/bin/env python
            "Compute a sqrt"
            import os
            import sys
            import extramod

            print(np.sqrt(int(sys.argv[1])))
            """
        ).strip()

        inputfile = tmp_path / "input.py"
        inputfile.write_text(src)

        main([str(inputfile), "--add", "extramod"])

        captured = capsys.readouterr()
        assert captured.out.strip() == expected

    def test_remove(self, tmp_path, capsys):
        src = dedent(
            """\
            #! /usr/bin/env python
            "Compute a sqrt"
            import os
            import sys

            print(np.sqrt(int(sys.argv[1])))
            """
        ).strip()

        expected = dedent(
            """\
            #! /usr/bin/env python
            "Compute a sqrt"
            import os

            print(np.sqrt(int(sys.argv[1])))
            """
        ).strip()

        inputfile = tmp_path / "input.py"
        inputfile.write_text(src)

        main([str(inputfile), "--remove", "sys"])

        captured = capsys.readouterr()
        assert captured.out.strip() == expected
