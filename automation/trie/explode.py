EXPLICIT_JONGSUNG = True

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
# JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
#                  'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JONGSUNG_LIST = [chr(0) if EXPLICIT_JONGSUNG else '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
ALL_SET = set([*CHOSUNG_LIST,*JUNGSUNG_LIST,*JONGSUNG_LIST])

CHOSUNG_MAP = {CHOSUNG_LIST[i]:i for i in range(len(CHOSUNG_LIST))}
JUNGSUNG_MAP = {JUNGSUNG_LIST[i]:i for i in range(len(JUNGSUNG_LIST))}
JONGSUNG_MAP = {JONGSUNG_LIST[i]:i for i in range(len(JONGSUNG_LIST))}

# https://frhyme.github.io/python/python_korean_englished/
# def korean_to_be_englished(korean_word):
#     r_lst = []
#     for w in list(korean_word.strip()):
#         ## 영어인 경우 구분해서 작성함.
#         if '가' <= w <= '힣':
#             ## 588개 마다 초성이 바뀜.
#             ch1 = (ord(w) - ord('가')) // 588
#             ## 중성은 총 28가지 종류
#             ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
#             ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
#             r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
#         else:
#             r_lst.append([w])
#     return r_lst

def explode(korean_word):
    r_lst = []
    for w in list(korean_word.strip()):
        ## 영어인 경우 구분해서 작성함.
        if '가' <= w <= '힣':
            ## 588개 마다 초성이 바뀜.
            ch1 = (ord(w) - ord('가')) // 588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
            r_lst.extend([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        else:
            r_lst.extend([w])
    return ''.join(r_lst)

def _join_char(exploded_char):
    if len(exploded_char)<3 : exploded_char = exploded_char+(chr(0),)
    return chr(
        ord('가') +
        CHOSUNG_MAP[exploded_char[0]]*588 +
        JUNGSUNG_MAP[exploded_char[1]]*28 +
        JONGSUNG_MAP[exploded_char[2]])

def assemble(exploded_word):
    #cur = min([exploded_word.find(c) for c in JUNGSUNG_LIST])
    exploded_word = ''.join([c for c in exploded_word if c in ALL_SET])
    word = zip(exploded_word[:-1], exploded_word[1:], exploded_word[2:]+' ', exploded_word[4:]+'    ')
    word = [(w[0:3] if w[3] in JUNGSUNG_LIST else w[0:2]) for w in word if (w[1] in JUNGSUNG_LIST)]
    return ''.join(([_join_char(c) for c in word]))
