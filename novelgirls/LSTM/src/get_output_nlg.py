import nlg_eye
import nlg_hair

inf= open('../class_output.txt', 'r')
cls= inf.read()
eye, hair= cls.split('\t')
inf.close()


def get_output(in_gender, eye, hair):
  in_eye= 'eye color is '+ eye +'.'
  in_hair= 'hair color is '+ hair +'.'
  n=15 #문장 길이

  out_eye= nlg_eye.sentence_generation(in_eye, n)
  out_eye= out_eye.split('.')

  out_hair= nlg_hair.sentence_generation(in_hair, n)
  out_hair= out_hair.split('.')

  out_eye= out_eye[1]
  out_hair= out_hair[1]
  out_sentence= out_eye +'. '+out_hair + '.'

  gender= {0:'male', 1:'female'}
  #성별을 0, 1로 입력받는다고 가정

  temp= out_sentence.split(' ')
  if gender[in_gender]=='male':
    for i, data in enumerate(temp):
      if data=='she':
        temp[i]='he'
      elif data=='her':
        temp[i]='his'

  else:
    for i, data in enumerate(temp):
      if data=='he':
        temp[i]='she'
      elif data=='his':
        temp[i]='her'

  result=''
  for i, data in enumerate(temp):
        result += data+' '

  return result

# result= get_output(1)
# print(result)


res = get_output(0, eye, hair)
print(res)