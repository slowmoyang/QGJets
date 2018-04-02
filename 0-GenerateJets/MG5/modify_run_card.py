"""
https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
"""
import sys
import codecs
from tempfile import mkstemp
from shutil import move
from os import remove


def parse_line(line):
    # comment starting with "#"
    if line.startswith('#'):
        value = None
        var = None
        comment = line
    else:
        if "!" in line:
            value_var, comment = line.split("!")
            comment = comment.strip()
        else:
            value_var = line
            comment = None
        # value = var ! comment
        value_var = value_var.replace(" ", "")
        if "=" in value_var:
            value, var = value_var.split("=")
        # comment starting wtih "!"
        else:
            value = None
            var = None

    return [value, var, comment]


def write_line(entry):
    if entry[0] is None:
        line = ""
    else:
        line = "{var} = {value} ".format(
            var=entry[0],
            value=entry[1])
        
    if entry[2] is not None:
        line += " ! "
        line += entry[2]

    return line

def modify_run_card(source_file_path, **kargs):

    fh, target_file_path = mkstemp()

    with codecs.open(target_file_path, 'w', 'utf-8') as target_file:
        with codecs.open(source_file_path, 'r', 'utf-8') as source_file:
            for line in source_file:

                entry = parse_line(line)
                if entry[1] is not None:
                    if kargs.has_key(entry[1]):
                        entry[0] = kargs.pop(entry[1])
                        line = write_line(entry)

                if not line.endswith("\n"):
                    line += "\n"

                target_file.write(line)

    remove(source_file_path)

    move(target_file_path, source_file_path)

if __name__ == '__main__':
    path = sys.argv[1]

    if "=" in path:
        path = path.split("=")[0]


    kargs = dict(x.split('=', 1) for x in sys.argv[2:])
    modify_run_card(path, **kargs)
