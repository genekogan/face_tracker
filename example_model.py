# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from PIL import Image
import dlib
import face_recognition
import numpy as np



def get_bounding_box(face, h, w):
    top, right, bottom, left = face
    z = [float(left)/w, float(top)/h, float(right)/w, float(bottom)/h]
    print(z)
    return z

class FaceTracker():

    def __init__(self, options):
        self.known_faces = {}
        self.index = 0

    # Generate an image based on some text.
    def process(self, input):

        img = np.array(input)
        h, w = img.shape[0:2]
        print("HW;,", h,w)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces([self.known_faces[f]["encoding"] for f in self.known_faces], face_encoding)
            if True not in matches:
                self.known_faces[self.index] = {"index": self.index, "encoding": face_encoding}
                match_index = self.index
                self.index += 1
            else:
                match_index = matches.index(True)
            
            print("look for", face_location)
            faces.append({"index": match_index, "location": get_bounding_box(face_location, h, w)})

        return faces
        