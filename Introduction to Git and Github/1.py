import re

def rearrange_name(nama):

    result = re.search(r"^([\w .]*), ([\w .]*)$", nama)

    if result == None:

        return nama

    return "{} {}".format(result[2], result[1])

