from LSTM.src import nlg_eye
from LSTM.src import nlg_hair


def get_output(in_gender, eye, hair):
    in_eye = 'eye color is ' + eye + '.'
    in_hair = 'hair color is ' + hair + '.'
    n = 15  # 문장 길이

    out_eye = nlg_eye.sentence_generation(in_eye, n)
    out_eye = out_eye.split('.')

    out_hair = nlg_hair.sentence_generation(in_hair, n)
    out_hair = out_hair.split('.')

    out_eye = out_eye[1]
    out_hair = out_hair[1]
    out_sentence = out_eye + '. ' + out_hair + '.'

    temp = out_sentence.split(' ')
    if in_gender == 0:  # 남성
        for i, data in enumerate(temp):
            if data == 'she':
                temp[i] = 'he'
            elif data == 'her':
                temp[i] = 'his'

    else: # 여성
        for i, data in enumerate(temp):
            if data == 'he':
                temp[i] = 'she'
            elif data == 'his':
                temp[i] = 'her'

    result = ''
    for i, data in enumerate(temp):
        result += data+' '

    return result
