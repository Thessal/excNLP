from konlpy.tag import Kkma
kkma = Kkma()


def morph(input_data):  # 형태소 분석
    preprocessed = kkma.pos(input_data)
    print(preprocessed)


morph("우리가 가지고 있는 수학 외적인 모든 지식과 표현 가능한 논리들은 추측으로 이루어져 있다.")
