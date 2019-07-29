import re
import string

VN_COMBINE_ACCENT_REPLACE = {
    'à': 'à',
    'á': 'á',
    'ã': 'ã',
    'ả': 'ả',
    'ạ': 'ạ',
    'è': 'è',
    'é': 'é',
    'ẽ': 'ẽ',
    'ẻ': 'ẻ',
    'ẹ': 'ẹ',
    'ì': 'ì',
    'í': 'í',
    'ĩ': 'ĩ',
    'ỉ': 'ỉ',
    'ị': 'ị',
    'ò': 'ò',
    'ó': 'ó',
    'õ': 'õ',
    'ỏ': 'ỏ',
    'ọ': 'ọ',
    'ờ': 'ờ',
    'ớ': 'ớ',
    'ỡ': 'ỡ',
    'ở': 'ở',
    'ợ': 'ợ',
    'ù': 'ù',
    'ú': 'ú',
    'ũ': 'ũ',
    'ủ': 'ủ',
    'ụ': 'ụ',
    'ỳ': 'ỳ',
    'ý': 'ý',
    'ỹ': 'ỹ',
    'ỷ': 'ỷ',
    'ỵ': 'ỵ',
    'â': 'â',
    'ầ': 'ầ',
    'ấ': 'ấ',
    'ẫ': 'ẫ',
    'ẩ': 'ẩ',
    'ậ': 'ậ',
    'ằ': 'ằ',
    'ắ': 'ắ',
    'ẵ': 'ẵ',
    'ẳ': 'ẳ',
    'ặ': 'ặ',
    'ừ': 'ừ',
    'ứ': 'ứ',
    'ữ': 'ữ',
    'ử': 'ử',
    'ự': 'ự',
    'ê': 'ê',
    'ề': 'ề',
    'ế': 'ế',
    'ễ': 'ễ',
    'ể': 'ể',
    'ệ': 'ệ',
    'ô': 'ô',
    'ồ': 'ồ',
    'ố': 'ố',
    'ỗ': 'ỗ',
    'ổ': 'ổ',
    'ộ': 'ộ'
}

VN_UPPERCASE = 'AẠẢÀÁÃ' \
               'ÂẬẨẦẤẪ' \
               'ĂẶẮẰẮẴ' \
               'BCDĐ' \
               'EẸẺÈÉẼ' \
               'ÊỆỂỀẾỄ' \
               'FGH' \
               'IỊỈÌÍĨ' \
               'JKLMN' \
               'OỌỎÒÓÕ' \
               'ÔỘỔỒỐỖ' \
               'ƠỢỞỜỚỠ' \
               'PQRST' \
               'UỤỦÙÚŨ' \
               'ƯỰỬỪỨỮ' \
               'VWX' \
               'YỴỶỲÝỸ' \
               'Z'

VN_LOWERCASE = 'aạảàáã' \
               'âậẩầấẫ' \
               'ăặẳằắẵ' \
               'bcdđ' \
               'eẹẻèéẽ' \
               'êệểềếễ' \
               'fgh' \
               'iịỉìíĩ' \
               'jklmn' \
               'oọỏòóõ' \
               'ôộổồốỗ' \
               'ơợởờớỡ' \
               'pqrst' \
               'uụủùúũ' \
               'ưựửừứữ' \
               'vwx' \
               'yỵỷỳýỹ' \
               'z'

DIGIT = '0123456789'
ADDITIONAL_CHARACTERS = '`~!@#$%^&*()-_=+\|]}[{"\';:/?.>,<“”‘’…'

_DIGIT = set([x for x in DIGIT])
_ADDITIONAL_CHARACTERS = set([x for x in ADDITIONAL_CHARACTERS])
_VN_LOWERCASE = set([x for x in VN_LOWERCASE])


def vn_isuppercase(char):
    """Check is uppercase for a vn character

    :param char: a unicode character
    :return:
    """
    if char in DIGIT or char in ADDITIONAL_CHARACTERS:
        return True

    return char in VN_UPPERCASE


def vn_islowercase(char):
    """Check is lowercase for a vn character

    :param char: a unicode character
    :return:
    """
    if char in _DIGIT or char in _ADDITIONAL_CHARACTERS:
        return True

    return char in VN_LOWERCASE


def vn_tolowercase(s):
    """To lower case a vn string

    :param s: a unicode vn string
    :return:
    """
    ls = list(s)
    for c in range(0, len(ls)):
        if ls[c] in _DIGIT or ls[c] in _ADDITIONAL_CHARACTERS:
            continue

        if vn_isuppercase(ls[c]):
            ic = VN_UPPERCASE.index(ls[c])
            ls[c] = VN_LOWERCASE[ic]

    return ''.join(ls)

def vn_combine_accent_replace(s):
    """
    convert ascii+combine_accent -> unicode_char
    :param s:
    :return:
    """
    ss = set([x for x in s])
    for k, v in VN_COMBINE_ACCENT_REPLACE.items():
        if k in ss:
            s = s.replace(k, v)
    return s


strip_chars = "`!#$%%^&*-_=+{}\|;:'\"<>/?,."

def query_preprocessing(query):
    """ Simple preprocessing query"""
    pquery = query.strip(strip_chars)
    pquery = re.sub("(\d+),(\d+)", "\\1.\\2", pquery)
    pquery = re.sub("([\!?,“”【】\"':/()…\-])", " \\1 ", pquery).strip()
    # pquery = re.sub(u"(?!\d+)\.(?!\d+)", "\\1 . \\2", pquery)
    pquery = " ".join(pquery.split())
    pquery = vn_tolowercase(pquery)
    pquery = "".join(filter(lambda x: x in string.printable or x in _VN_LOWERCASE, pquery))
    return pquery