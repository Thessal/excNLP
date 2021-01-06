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
HANGUL_COMPATIBILITY_JAMO = {4352: "ㄱ", 4353: "ㄲ", 4354: "ㄴ", 4355: "ㄷ", 4356: "ㄸ", 4357: "ㄹ", 4358: "ㅁ", 4359: "ㅂ",
                             4360: "ㅃ", 4361: "ㅅ", 4362: "ㅆ", 4363: "ㅇ", 4364: "ㅈ", 4365: "ㅉ", 4366: "ㅊ", 4367: "ㅋ",
                             4368: "ㅌ", 4369: "ㅍ", 4370: "ㅎ", 4449: "ᅡ", 4450: "ᅢ", 4451: "ᅣ", 4452: "ᅤ", 4453: "ᅥ",
                             4454: "ᅦ", 4455: "ᅧ", 4456: "ᅨ", 4457: "ᅩ", 4458: "ᅪ", 4459: "ᅫ", 4460: "ᅬ", 4461: "ᅭ",
                             4462: "ᅮ", 4463: "ᅯ", 4464: "ᅰ", 4465: "ᅱ", 4466: "ᅲ", 4467: "ᅳ", 4468: "ᅴ", 4469: "ᅵ",
                             4520: "ᆨ", 4521: "ᆩ", 4522: "ᆪ", 4523: "ᆫ", 4524: "ᆬ", 4525: "ᆭ", 4526: "ᆮ", 4527: "ᆯ",
                             4528: "ᆰ", 4529: "ᆱ", 4530: "ᆲ", 4531: "ᆳ", 4532: "ᆴ", 4533: "ᆵ", 4534: "ᆶ", 4535: "ᆷ",
                             4536: "ᆸ", 4537: "ᆹ", 4538: "ᆺ", 4539: "ᆻ", 4540: "ᆼ", 4541: "ᆽ", 4542: "ᆾ", 4543: "ᆿ",
                             4544: "ᇀ", 4545: "ᇁ", 4546: "ᇂ"}
HANGUL_COMPATIBILITY_JAMO = {4352: chr(12593), 4353: chr(12594), 4354: chr(12596), 4355: chr(12599), 4356: chr(12600),
                             4357: chr(12601), 4358: chr(12609), 4359: chr(12610), 4360: chr(12611), 4361: chr(12613),
                             4362: chr(12614), 4363: chr(12615), 4364: chr(12616), 4365: chr(12617), 4366: chr(12618),
                             4367: chr(12619), 4368: chr(12620), 4369: chr(12621), 4370: chr(12622), 4372: chr(12645),
                             4373: chr(12646), 4378: chr(12653), 4380: chr(12654), 4381: chr(12657), 4382: chr(12658),
                             4384: chr(12659), 4385: chr(12612), 4386: chr(12660), 4387: chr(12661), 4391: chr(12662),
                             4393: chr(12663), 4395: chr(12664), 4396: chr(12665), 4397: chr(12666), 4398: chr(12667),
                             4399: chr(12668), 4402: chr(12669), 4406: chr(12670), 4412: chr(12613), 4413: chr(12614),
                             4414: chr(12613), 4415: chr(12614), 4416: chr(12671), 4421: chr(12674), 4422: chr(12675),
                             4423: chr(12672), 4428: chr(12615), 4430: chr(12616), 4431: chr(12617), 4432: chr(12616),
                             4433: chr(12617), 4436: chr(12618), 4437: chr(12618), 4439: chr(12676), 4440: chr(12677),
                             4441: chr(12678), 4447: chr(12644), 4449: chr(12623), 4450: chr(12624), 4451: chr(12625),
                             4452: chr(12626), 4453: chr(12627), 4454: chr(12628), 4455: chr(12629), 4456: chr(12630),
                             4457: chr(12631), 4458: chr(12632), 4459: chr(12633), 4460: chr(12634), 4461: chr(12635),
                             4462: chr(12636), 4463: chr(12637), 4464: chr(12638), 4465: chr(12639), 4466: chr(12640),
                             4467: chr(12641), 4468: chr(12642), 4469: chr(12643), 4484: chr(12679), 4485: chr(12680),
                             4488: chr(12681), 4497: chr(12682), 4498: chr(12683), 4500: chr(12684), 4510: chr(12685),
                             4513: chr(12686), 4520: chr(12593), 4521: chr(12594), 4522: chr(12595), 4523: chr(12596),
                             4524: chr(12597), 4525: chr(12598), 4526: chr(12599), 4527: chr(12601), 4528: chr(12602),
                             4529: chr(12603), 4530: chr(12604), 4531: chr(12605), 4532: chr(12606), 4533: chr(12607),
                             4534: chr(12608), 4535: chr(12609), 4536: chr(12610), 4537: chr(12612), 4538: chr(12613),
                             4539: chr(12614), 4540: chr(12615), 4541: chr(12616), 4542: chr(12618), 4543: chr(12619),
                             4544: chr(12620), 4545: chr(12621), 4546: chr(12622), 4550: chr(12646), 4551: chr(12647),
                             4552: chr(12648), 4556: chr(12649), 4558: chr(12650), 4563: chr(12651), 4567: chr(12652),
                             4569: chr(12653), 4572: chr(12654), 4573: chr(12655), 4575: chr(12656), 4582: chr(12664),
                             4583: chr(12666), 4584: chr(12668), 4586: chr(12669), 4587: chr(12671), 4590: chr(12672),
                             4592: chr(12615), 4593: chr(12674), 4594: chr(12675), 4596: chr(12676), 4601: chr(12678)}

ALL_SET = set([*CHOSUNG_LIST, *JUNGSUNG_LIST, *JONGSUNG_LIST])

CHOSUNG_MAP = {CHOSUNG_LIST[i]: i for i in range(len(CHOSUNG_LIST))}
JUNGSUNG_MAP = {JUNGSUNG_LIST[i]: i for i in range(len(JUNGSUNG_LIST))}
JONGSUNG_MAP = {JONGSUNG_LIST[i]: i for i in range(len(JONGSUNG_LIST))}


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
        elif allow_nonunique_assemble and ('ㄱ' <= w <= 'ㅎ'):  # 초성만 있는 경우
            r_lst.append(w)
        else:  # 'a, 1, ㅔ' ...
            if (32 <= ord(w) <= 126):  # 1바이트 문자 #ord(' ')==32
                r_lst.append(w)
            # 중성 종성만 있는 경우는 지운다
            if 4352 <= ord(w) <= 4546:  # convert to utf8
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

def assemble(exploded_word):
    """
    :param exploded_word: a word without space
    :return: assembled word string
    """
    if not exploded_word.strip():
        print("Warning: assembling empty string")
        return ""
    # elif all([(c in ALL_SET) for c in exploded_word]): # 자모결합된 pure hangul인 경우 vectorize 가능
    #     #exploded_word = ''.join([c for c in exploded_word if c in ALL_SET])
    #     word = zip(exploded_word[:-1], exploded_word[1:], exploded_word[2:] + ' ', exploded_word[4:] + '    ')
    #     word = [(w[0:3] if w[3] in JUNGSUNG_LIST else w[0:2]) for w in word if (w[1] in JUNGSUNG_LIST)]
    #     if exploded_word[-1] in JONGSUNG_LIST and len(word[-1])==2 :
    #         word[-1] = word[-1] + (exploded_word[-1],)
    #     return ''.join(([_join_char(c) for c in word]))
    else:  # contains non-hangul or jaeum-meoum only
        # if ' ' in exploded_word:
        #     print("Warning: assembling space character")
        output = []
        state = -1  # -1=nonkor, 0=cho, 1=jung, 2=jong
        for c in exploded_word:
            if 4352 <= ord(c) <= 4601:
                c = HANGUL_COMPATIBILITY_JAMO[ord(c)]
            if (c in CHOSUNG_LIST) and ((state != 1) or (c not in JONGSUNG_LIST)):
                state = 0
                output.append(c)
            elif state == 0 and (c in JUNGSUNG_LIST):
                state = 1
                output[-1] = chr(ord('가') + CHOSUNG_MAP[output[-1]] * 588 + JUNGSUNG_MAP[c] * 28)
            elif state == 2 and (c in JUNGSUNG_LIST):
                prev = JONGSUNG_LIST[(ord(output[-1]) - ord('가')) % 28]
                if prev in CHOSUNG_LIST:
                    state = 1
                    output[-1] = chr(ord(output[-1]) - (ord(output[-1]) - ord('가')) % 28)
                    output.append(chr(ord('가') + CHOSUNG_MAP[prev] * 588 + JUNGSUNG_MAP[c] * 28))
                else:
                    state = -1
                    output.append(c)
            elif state == 1 and (c in JONGSUNG_LIST):
                state = 2
                output[-1] = chr(ord(output[-1]) + JONGSUNG_MAP[c])
            else:
                state = -1
                output.append(c)
        return ''.join(output)
