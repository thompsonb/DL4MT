import os
import getpass

python_loc = '/path/to/python/with/lots/of/extra/packages'
cuda_loc = '/usr/local/cuda'
KENLM_PATH = '/todo/path/kenlm/build/'
PORT_NUMBER = 8091
temp_loc = None  # None==use system default
TEMP_DIR = None  # None==use system default

# Create a directory for the data, then cd into it and run the en-de command
# wget command for just en-de
# wget -r --cut-dirs=2 -e robots=off -nH -np -R index.html* http://data.statmt.org/rsennrich/wmt16_systems/en-de/

# if you wanted all of the language pairs...
# wget -r --cut-dirs=1 -e robots=off -nH -np -R index.html* http://data.statmt.org/rsennrich/wmt16_systems/

wmt16_systems_dir = '/todo/path/wmt16_Rico_systems/'
