from novelgirls.BERT.src import eye_color_pred, hair_color_pred

# input을 어떻게 받으시는지 몰라 임시로 넣어두었습니다.
input_sentence=['', 'My eyes color is blue. and my hair color is brown.']

eye_color= eye_color_pred.get_prediction(input_sentence)
hair_color= hair_color_pred.get_prediction(input_sentence)

# print(eye_color[1][2])
# print(hair_color[1][2])

class_output= open('../class_output.txt', 'w')

eye_hair= eye_color[1][2] + '\t' + hair_color[1][2]
class_output.write(eye_hair)
class_output.close()

