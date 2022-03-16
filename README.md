# importfixer
This program adds and removes imports based on whats in use. It does not
autodetect available libraries, but relies on a configuration definition.

The primary interface is through the `importfixer` command line tool, with help
message:

```
usage: importfixer [-h] [-c CONFIG] [-d] [-o] [file] [output]

positional arguments:
  file                  Input file. Uses stdin if not provided
  output                Output file. Default prints to stdout

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Input config
  -d, --diff            Show difference
  -o, --overwrite       Overwrite input. Not compatible with <output>.
```

The config file defines the variables names that should always be understood as
things to be imported. The configuration supports normal imports, aliases, and
from imports, either with selections or in their entirety. That is an example
configuration file showing the four types of definitions is:

```
os           # Simple import
numpy as np  # Alias
from scipy import optimize, linalg   # Import specific names from library/module
from typing import *   # Import any name from library module
```

Note that the configuration file is a simple text file with comments defining
all the possible imports.

Given this configuration, the following input file:

```
import sys
import glob
from typing import Dict

def myfunc(arg: Dict) -> List:
  return list(arg.keys())

print(sys.argv)
print(os.getcwd())
print(linalg.cholesky(np.eye(6)))
```

would be rewritten as:

```
import sys
from typing import Dict, List
import os
from scipy import linalg
import numpy as np

def myfunc(arg: Dict) -> List:
  return list(arg.keys())

print(sys.argv)
print(os.getcwd())
print(linalg.cholesky(np.eye(6)))
```

Note that the ordering simply appends to the list of imports, in a way that will
minimize differences, but is usually not ideal. These can be manually reordered
before committing, or used in conjunction with `isort` to enforce a standard of
import order and style.


## API
In addition to the command line tool, the individual python functions can be
used individually or as a whole:

```python
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
```

## Developing
In order to set up a development environment, first create a virtual environment
(`python -m venv env` to create one in directory `./env`), then activate it
(`source ./env/bin/activate`), and then run `./setup-dev.sh`. This will install
the libraries required for development, set up the pre-commit hooks to enforce
black and isort, and install the library as editable.

Once installed you should add tests defining the expected behavior, and then
implement your update. Run `pytest` to run the automated tests.

Note that this project uses the `black` style enforcer, and `isort` to maintain
imports.


## Design Decisions
This is primarily intended to serve as a tool to run inside a text editor, since
the fixes are relatively trivial but annoying when working. There are two
particular choices to justify:

1. Writing a new tool instead of using `autoimport`. This tool relies on
   automatically detecting available libraries rather than explicitly
   whitelisting them, and doesn't provide a way to generically import all names
   from a library, which is one of my main needs. The latter could be addressed
   in a pull request, but the former probably couldn't be. I'd rather whitelist
   the few libraries I use all the time and not be suprised by (e.g.) using a
   variable named "time" incorrectly and getting the error masked in the linter
   by the auto-importing tool.

2. Using a configuration file in the home directory rather than
   `pyproject.toml`. This is because the tool is really a personal preference
   rather than something you specify on a per-project basis. Therefore a single
   config file in your home directory makes the most sense.  Given that the
   entries are a list of strings, it made more sense to eliminate boilerplate
   and use a simple list.
