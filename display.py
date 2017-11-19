import cv2
from draw_frame import DrawFrame
from hand_detection import HandDetection


def loop():
    camera = cv2.VideoCapture(0)

    df = DrawFrame()
    hd = HandDetection()

    while True:
        # get frame
        (grabbed, frame_in) = camera.read()

        # original frame
        frame_orig = frame_in.copy()

        # shrink frame
        frame = df.resize(frame_in)

        # flipped frame to draw on
        frame_final = df.flip(frame)

        if cv2.waitKey(1) == ord('h') & 0xFF:
            # if pd.trained_paper and
            if not hd.trained_hand:
                hd.train_hand(frame)
        # click q to quit
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break

        if not hd.trained_hand:
            frame_final = hd.draw_hand_rect(frame_final)
        elif hd.trained_hand:
            frame_final = df.draw_final(frame_final, hd)

        # display frame
        cv2.imshow('image', frame_final)

    # cleanup
    camera.release()
    cv2.destroyAllWindows()
