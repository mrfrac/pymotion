import time
import cv2 as cv
from numpy import ndarray
from datetime import datetime


class ReadException(Exception):
    pass


class PyMotion:
    def __init__(self, camera=0, threshold=10, scale=0.5, show_window=False):
        self.show_window = show_window
        self.camera = camera
        self.open_camera()
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.scaled_width = int(self.width * scale)
        self.scaled_height = int(self.height * scale)

        self.garbage_area = (self.scaled_width * self.scaled_height) / 100
        self.threshold_area = self.garbage_area * threshold

        if self.show_window:
            cv.namedWindow("motion")

        self.run()

    def open_camera(self):
        self.cam = cv.VideoCapture(self.camera)

    def close_camera(self):
        if self.cam.isOpened():
            self.cam.release()

    def read_image(self):
        if self.cam.isOpened():
            status, image = self.cam.read()

        if not status:
            raise ReadException("read error")

        return image

    def process_image(self, image):
        status, image = cv.threshold(cv.cvtColor(image, cv.COLOR_RGB2GRAY), 10, 255, cv.THRESH_BINARY)

        if status:
            image = cv.resize(image, (self.scaled_width, self.scaled_height))
            image = cv.GaussianBlur(image, (21, 21), 0)
            image = cv.dilate(image, None, iterations=2)

        return image

    @staticmethod
    def calc_diff(image0, image1):
        return cv.absdiff(image1, image0)

    def run(self):
        previous = self.read_image()
        counter = 0

        while True:
            print(str(counter))
            try:
                current = self.read_image()
            except ReadException:
                print("read error! Trying to reopen device")
                # self.cam.reopen()
                self.close_camera()
                self.open_camera()

            diff = self.calc_diff(previous, current)
            diff = self.process_image(diff)

            cv.imshow("motion", diff)

            diff, moved = self.something_moved(diff)

            if moved:
                filename = datetime.now().strftime('%Y-%m-%d_%H%M%S%f') + '.jpg'
                cv.imwrite(filename, current)
                # cv.imwrite(filename + '.diff.jpg', diff)
                if self.show_window:
                    cv.imshow("motion", current)

            counter += 1

            previous = current
            time.sleep(0.3)
            cv.waitKey(81)

    def __del__(self):
        self.cam.release()

    def something_moved(self, image):
        _, contours, _ = cv.findContours(image=image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        affected_area = 0
        cv.drawContours(image, contours, 0, (255, 255, 255))
        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.garbage_area * 10:
                continue
            affected_area += area

            #(x, y, w, h) = cv.boundingRect(contour)
            #cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print(affected_area, self.garbage_area * 100)
        return image, affected_area > self.threshold_area and affected_area < self.garbage_area * 70


if __name__ == "__main__":
    motion = PyMotion(
        threshold=8,
        camera="",
        show_window=True)
