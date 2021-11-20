from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QPixmap
import sys
# from novelgirls.BERT.src.eye_color_pred import get_prediction as eye_pred
# from novelgirls.BERT.src.hair_color_pred import get_prediction as hair_pred
# from novelgirls.LSTM.src.get_output_nlg import get_output

main_ui = 'UI/main.UI'
form_1, base_1 = uic.loadUiType(main_ui)

input_ui = 'UI/input_text.UI'
form_2, base_2 = uic.loadUiType(input_ui)

output_ui = 'UI/output.UI'
form_3, base_3 = uic.loadUiType(output_ui)


class main_page(base_1, form_1):
    def __init__(self):
        super(base_1, self).__init__()
        self.setupUi(self)
        pixmap = QPixmap('UI/r_girl.png')
        self.book_g.setPixmap(pixmap)
        self.show()
        self.male.clicked.connect(self.set_gender)
        self.female.clicked.connect(self.set_gender)
        self.start_btn.clicked.connect(self.change)
        self.end_btn.clicked.connect(self.quit)

    def set_gender(self):
        global gender
        if self.male.isChecked():
            gender = 0
        elif self.female.isChecked():
            gender = 1

    def change(self):
        self.main = input_text_page()
        self.main.show()
        self.close()

    def quit(self):
        self.close()


class input_text_page(base_2, form_2):
    def __init__(self):
        super(base_2, self).__init__()
        self.setupUi(self)
        self.text = 'input_text'
        pixmap = QPixmap('UI/girl.png')
        self.look_g.setPixmap(pixmap)
        self.show()
        self.in_btn.clicked.connect(self.button_event)

    """ 이 부분이 자꾸ㅜㅜ 막혀요ㅜㅜ"""
    def button_event(self):
        self.label.setText('*** 입력해주신 문장으로 글을 쓰는 중이에요! ***')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.repaint()
        text = self.lineEdit.text()
        l = list()
        l.append('')
        l.append(text)
        print(l[1])
        print(gender)
        # eye_color = eye_pred(l)
        # hair_color = hair_pred(l)
        global res
        res = '어떡해요ㅜㅜ'
        # res = get_output(gender, eye_color[1][2], hair_color[1][2])
        # print(res)
        self.change()

    def change(self):
        self.main = output_page()
        self.main.show()
        self.close()


class output_page(base_3, form_3):
    def __init__(self):
        super(base_3, self).__init__()
        self.setupUi(self)
        pixmap = QPixmap('UI/novelgirls.png')
        self.g_.setPixmap(pixmap)
        pixmap2 = QPixmap('UI/pen.png')
        self.pen1.setPixmap(pixmap2)
        self.pen2.setPixmap(pixmap2)
        self.show()
        self.exit_btn.clicked.connect(self.quit)
        self.back_btn.clicked.connect(self.change)
        self.output_text.setText(res)

    def change(self):
        self.main = input_text_page()
        self.main.show()
        self.close()

    def quit(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    op = main_page()
    op.show()
    sys.exit(app.exec_())