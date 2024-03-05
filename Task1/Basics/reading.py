import cv2 as cv

img = cv.imread('images/cats.jpg')
cv.imshow('Cats', img)

cv.waitKey(0)

# Reading Videos
ca = cv.VideoCapture('Video/video.mp4')

while True:
    isTrue, frame = ca.read()
    if isTrue:    
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break

ca.release()
cv.destroyAllWindows()
