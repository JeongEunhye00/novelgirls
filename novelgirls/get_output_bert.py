# -*- coding: utf-8 -*-
"""get_output_BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M0ZW792HZueVT3trH_IyNAlvnBVzshcK
"""

# BERT랑 LSTM의 tenserflow 버전이 안 맞아서 동시에 import가 불가능함. 
# 그래서 BERT output을 따로 파일로 넘겨 LSTM에 넣음


# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab\ Notebooks/novelgirls

# !pip install tensorflow==1.15.0
# !pip install --upgrade tb-nightly
# !pip install bert-tensorflow==1.0.1

import eye_color_pred
import hair_color_pred

# input을 어떻게 받으시는지 몰라 임시로 넣어두었습니다.
input_sentence=['', 'My eyes color is blue. and my hair color is brown.']

eye_color= eye_color_pred.get_prediction(input_sentence)
hair_color= hair_color_pred.get_prediction(input_sentence)

# print(eye_color[1][2])
# print(hair_color[1][2])

class_output= open('./BERT/class_output.txt', 'w')

eye_hair= eye_color[1][2] + '\t' + hair_color[1][2]
class_output.write(eye_hair)
class_output.close()

