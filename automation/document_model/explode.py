# if Explicitly Jongsung, explode("때무") is not in explode("때문") because ㄸㅐ\x00ㅁㅜ\x00 not in ㄸㅐ\x00ㅁㅜㄴ
EXPLICIT_JONGSUNG = False

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
# JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
#                  'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JONGSUNG_LIST = [chr(0) if EXPLICIT_JONGSUNG else '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ',
                 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
HANGUL_COMPATIBILITY_JAMO = {4352:"ㄱ", 4353:"ㄲ", 4354:"ㄴ", 4355:"ㄷ", 4356:"ㄸ", 4357:"ㄹ", 4358:"ㅁ", 4359:"ㅂ", 4360:"ㅃ", 4361:"ㅅ", 4362:"ㅆ", 4363:"ㅇ", 4364:"ㅈ", 4365:"ㅉ", 4366:"ㅊ", 4367:"ㅋ", 4368:"ㅌ", 4369:"ㅍ", 4370:"ㅎ", 4449:"ᅡ", 4450:"ᅢ", 4451:"ᅣ", 4452:"ᅤ", 4453:"ᅥ", 4454:"ᅦ", 4455:"ᅧ", 4456:"ᅨ", 4457:"ᅩ", 4458:"ᅪ", 4459:"ᅫ", 4460:"ᅬ", 4461:"ᅭ", 4462:"ᅮ", 4463:"ᅯ", 4464:"ᅰ", 4465:"ᅱ", 4466:"ᅲ", 4467:"ᅳ", 4468:"ᅴ", 4469:"ᅵ", 4520:"ᆨ", 4521:"ᆩ", 4522:"ᆪ", 4523:"ᆫ", 4524:"ᆬ", 4525:"ᆭ", 4526:"ᆮ", 4527:"ᆯ", 4528:"ᆰ", 4529:"ᆱ", 4530:"ᆲ", 4531:"ᆳ", 4532:"ᆴ", 4533:"ᆵ", 4534:"ᆶ", 4535:"ᆷ", 4536:"ᆸ", 4537:"ᆹ", 4538:"ᆺ", 4539:"ᆻ", 4540:"ᆼ", 4541:"ᆽ", 4542:"ᆾ", 4543:"ᆿ", 4544:"ᇀ", 4545:"ᇁ", 4546:"ᇂ"}
ALL_SET = set([*CHOSUNG_LIST, *JUNGSUNG_LIST, *JONGSUNG_LIST])

CHOSUNG_MAP = {CHOSUNG_LIST[i]: i for i in range(len(CHOSUNG_LIST))}
JUNGSUNG_MAP = {JUNGSUNG_LIST[i]: i for i in range(len(JUNGSUNG_LIST))}
JONGSUNG_MAP = {JONGSUNG_LIST[i]: i for i in range(len(JONGSUNG_LIST))}


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

@staticmethod
def explode(korean_word, allow_nonunique_assemble=True):
    """
    explodes hangul.
    :param korean_word: string. allows space
    :param allow_nonunique_assemble: do not remove characters such as 'ㄱ', 'ㅎ'
    :return: exploded string
    """
    r_lst = []
    for w in list(korean_word.strip()):
        ## 영어인 경우 구분해서 작성함.
        if ('가' <= w <= '힣'):
            ## 588개 마다 초성이 바뀜.
            ch1 = (ord(w) - ord('가')) // 588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
            r_lst.extend([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        elif allow_nonunique_assemble and ('ㄱ' <= w <= 'ㅎ'): #초성만 있는 경우
            r_lst.append(w)
        else: # 'a, 1, ㅔ' ...
            if (32<= ord(w) <= 126) : #1바이트 문자 #ord(' ')==32
                r_lst.append(w)
            # 중성 종성만 있는 경우는 지운다
            if 4352<=ord(w)<=4546: # convert to utf8
                # FIXME : 'ᄡ' chr(4385) not in compatibility_jamo
                r_lst.append(HANGUL_COMPATIBILITY_JAMO[ord(w)])
        # if not ('가' <= w <= '힣'): # convenient for exception monitoring
        #     print(ord(w), w)
    return ''.join(r_lst)


def _join_char(exploded_char):
    """
    :param exploded_char: set of exploded chars e.g. ('ㅅ', 'ㅡ')
    :return: one character e.g. '스'
    """
    if len(exploded_char) == 1: return exploded_char
    if len(exploded_char) < 3: exploded_char = exploded_char + ('',)
    return chr(
        ord('가') +
        CHOSUNG_MAP[exploded_char[0]] * 588 +
        JUNGSUNG_MAP[exploded_char[1]] * 28 +
        JONGSUNG_MAP[exploded_char[2]])


## UTF-8 is lossy... So you can hide information in the characters. I take that loss
# 1100..11FF    ; Hangul # Lo [256] HANGUL CHOSEONG KIYEOK..HANGUL JONGSEONG SSANGNIEUN
# 302E..302F    ; Hangul # Mc   [2] HANGUL SINGLE DOT TONE MARK..HANGUL DOUBLE DOT TONE MARK
# 3131..318E    ; Hangul # Lo  [94] HANGUL LETTER KIYEOK..HANGUL LETTER ARAEAE
# 3200..321E    ; Hangul # So  [31] PARENTHESIZED HANGUL KIYEOK..PARENTHESIZED KOREAN CHARACTER O HU
# 3260..327E    ; Hangul # So  [31] CIRCLED HANGUL KIYEOK..CIRCLED HANGUL IEUNG U
# A960..A97C    ; Hangul # Lo  [29] HANGUL CHOSEONG TIKEUT-MIEUM..HANGUL CHOSEONG SSANGYEORINHIEUH
# AC00..D7A3    ; Hangul # Lo [11172] HANGUL SYLLABLE GA..HANGUL SYLLABLE HIH
# D7B0..D7C6    ; Hangul # Lo  [23] HANGUL JUNGSEONG O-YEO..HANGUL JUNGSEONG ARAEA-E
# D7CB..D7FB    ; Hangul # Lo  [49] HANGUL JONGSEONG NIEUN-RIEUL..HANGUL JONGSEONG PHIEUPH-THIEUTH
# FFA0..FFBE    ; Hangul # Lo  [31] HALFWIDTH HANGUL FILLER..HALFWIDTH HANGUL LETTER HIEUH
# FFC2..FFC7    ; Hangul # Lo   [6] HALFWIDTH HANGUL LETTER A..HALFWIDTH HANGUL LETTER E
# FFCA..FFCF    ; Hangul # Lo   [6] HALFWIDTH HANGUL LETTER YEO..HALFWIDTH HANGUL LETTER OE
# FFD2..FFD7    ; Hangul # Lo   [6] HALFWIDTH HANGUL LETTER YO..HALFWIDTH HANGUL LETTER YU
# FFDA..FFDC    ; Hangul # Lo   [3] HALFWIDTH HANGUL LETTER EU..HALFWIDTH HANGUL LETTER I
# # Total code points: 11739
@staticmethod
def assemble(exploded_word):
    """
    :param exploded_word: a word without space
    :return: assembled word string
    """
    if not exploded_word.strip() :
        print("Warning: assembling empty string")
        return ""
    # elif all([(c in ALL_SET) for c in exploded_word]): # 자모결합된 pure hangul인 경우 vectorize 가능
    #     #exploded_word = ''.join([c for c in exploded_word if c in ALL_SET])
    #     word = zip(exploded_word[:-1], exploded_word[1:], exploded_word[2:] + ' ', exploded_word[4:] + '    ')
    #     word = [(w[0:3] if w[3] in JUNGSUNG_LIST else w[0:2]) for w in word if (w[1] in JUNGSUNG_LIST)]
    #     if exploded_word[-1] in JONGSUNG_LIST and len(word[-1])==2 :
    #         word[-1] = word[-1] + (exploded_word[-1],)
    #     return ''.join(([_join_char(c) for c in word]))
    else : # contains non-hangul or jaeum-meoum only
        if ' ' in exploded_word:
            print("Warning: assembling space character")
        output = []
        state = -1 # -1=nonkor, 0=cho, 1=jung, 2=jong
        for c in exploded_word:
            if 4352<=ord(c)<=4546:
                c = HANGUL_COMPATIBILITY_JAMO[ord(c)]
            if (c in CHOSUNG_LIST) and ((state!=1) or (c not in JONGSUNG_LIST)):
                state = 0
                output.append(c)
            elif state==0 and (c in JUNGSUNG_LIST):
                state = 1
                output[-1] = chr( ord('가') + CHOSUNG_MAP[output[-1]] * 588 + JUNGSUNG_MAP[c] * 28 )
            elif state == 2 and (c in JUNGSUNG_LIST):
                prev = JONGSUNG_LIST[(ord(output[-1]) - ord('가'))%28]
                if prev in CHOSUNG_LIST :
                    state = 1
                    output[-1] = chr( ord(output[-1]) - (ord(output[-1])-ord('가'))%28 )
                    output.append( chr( ord('가') + CHOSUNG_MAP[prev] * 588 + JUNGSUNG_MAP[c] * 28 ) )
                else :
                    state = -1
                    output.append(c)
            elif state==1 and (c in JONGSUNG_LIST):
                state = 2
                output[-1] = chr( ord(output[-1]) + JONGSUNG_MAP[c] )
            else :
                state = -1
                output.append(c)
        return ''.join(output)
