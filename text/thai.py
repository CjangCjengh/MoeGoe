import re
from num_thai.thainumbers import NumThai


num = NumThai()


def num_to_thai(text):
    return re.sub(r'(?:\d+,?\d+)+(?:\.\d+,?\d+)?', lambda x: ''.join(num.NumberToTextThai(float(x.group(0).replace(',', '')))), text)
