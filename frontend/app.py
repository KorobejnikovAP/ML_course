"""
 Copyright (c) 2023
 Created by Aleksey Korobeynikov
"""

import kivy
kivy.require('1.0.5')
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture

import cv2

class MainPageLayout(BoxLayout):
    pass

class DeepLearningExampleApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.play = False

    def build(self):
        self.widget = MainPageLayout()
        self.widget.ids.play_button.on_press = self.play_handler
        self.img = self.widget.ids.image
        return self.widget

    def start_camera(self):
        pass
        #opencv2 stuffs
        #self.capture = cv2.VideoCapture(1)
        #cv2.namedWindow("CV2 Image")
        # while(True):
        #     ret, frame = self.capture.read()
        #     # convert it to texture
        #     buf1 = cv2.flip(frame, 0)
        #     buf = buf1.tostring()
        #     texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #     texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #     # display image from the texture
        #     self.img.texture = texture1

    def play_handler(self):
        self.play = True
        #self.widget.ids.label.text = "Play now"
        if self.play:
            self.start_camera()
        print(self.play)


if __name__ == '__main__':
    Builder.load_file('view.kv')
    DeepLearningExampleApp().run()
