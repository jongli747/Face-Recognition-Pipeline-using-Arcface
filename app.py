from flask import Flask, render_template, request, redirect, url_for, Response
from register import reg
from face_verify import faceRec
import cv2
regclass = reg()

cam = faceRec()

app = Flask(__name__)

@app.route('/')
def index(name=None):
	if cap.isOpened():
		cap.release()
	return render_template('index.html',name=name)

@app.route('/registration', methods = ["GET", "POST"])
def parse(name=None):
    # import faces_test.py
    # print("done")
    if request.method == "POST":
    	req = request.form.getlist('Name')
    	NameOfPerson = ' '.join([str(elem) for elem in req])
    	a,b = regclass.main(NameOfPerson)
    	regclass.registration(a,b)
    	return redirect(url_for('index'))
    return render_template('registration.html',name="Name",mimetype='multipart/x-mixed-replace; boundary=frame')
#camera


def gen(cam):
	cap = cv2.VideoCapture(0)
	cap.set(3,500)
	cap.set(4,500)
	while True:
		frame = cam.main(cap)
		if frame != "":
			global_frame = frame
			yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		
			


@app.route('/identification')
def live_feed():
	return Response(gen(cam),mimetype='multipart/x-mixed-replace; boundary=frame')
	

if __name__ == '__main__':
	cap = cv2.VideoCapture(0)
	app.run()
	app.debug = True
