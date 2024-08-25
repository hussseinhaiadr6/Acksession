import cv2
import numpy as np
from pathlib import Path
import torch
import os
from boxmot import DeepOCSORT
import re
from collections import Counter
from paddleocr import PaddleOCR

import time


def initialize_ocr():
    return PaddleOCR(lang="ch", ocr_version="PP-OCRv4",precision="fp8", det=False)

def initialize_model():
    model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\ALPR\yolov5', 'custom', path=r'.\LP_detection.onnx', source='local')
    model.conf = 0.4  # NMS confidence threshold
    model.iou = 0.5  # NMS IoU threshold
    model.agnostic = True  # NMS class-agnostic
    return model

def initialize_tracker():
    return DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to useLP_detection
        device='cuda:0',
        fp16=False,
        max_age=5,
        min_hits=2
    )

def process_frame(frame, model, tracker):
    start_time = time.time()
    resize_dim = (640, 640)
    resized_frame = cv2.resize(frame, resize_dim)


    results = model(resized_frame)
    df = results.pandas().xyxy[0]




    if not df.empty:
        highest_conf_idx = df['confidence'].idxmax()
        dets = df.loc[highest_conf_idx, ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].to_numpy().reshape(1, -1)
        dets = dets.astype(np.float32)

        track_results = tracker.update(dets, resized_frame)

    else:
        dets = np.empty((0, 6), dtype=np.float32)
        track_results = tracker.update(dets, resized_frame)
    end_time = time.time()
    print('time', end_time - start_time)
    return resized_frame, track_results

def update_text_storage_and_write_results(track,last_id, current_id, current_final_ocr, text_storage, output_dir, resized_frame, frame_counter,resized_frame_cropped,do_ocr):
    start_time = time.time()
    track_id = int(track[4])  # Extract the ID
    xmin, ymin, xmax, ymax = int(0.95 * track[0]), int(0.95 * track[1]), int(1.05 * track[2]), int(1.05 * track[3])
    id_dir = os.path.join(output_dir, str(track_id))
    resized_frame_cropped_path = os.path.join(id_dir, f'{frame_counter}.jpg')
    if track_id != current_id:
        text_storage[track_id]=[]

        last_id=current_id
        do_ocr=True
        if last_id is not None:
            with open("./ALPR_results_version2.txt", 'a') as file:
                file.write(f"ID:{last_id} Plate Number:{current_final_ocr}\n")
            print("last ID:",last_id)
            text_storage.pop(last_id)
        current_id=track_id
        current_final_ocr = "No detection"
        os.makedirs(id_dir, exist_ok=True)
        cv2.imwrite(os.path.join(id_dir, f'{frame_counter - 1}.jpg'), resized_frame)
        cv2.imwrite(resized_frame_cropped_path, resized_frame[ymin:ymax, xmin:xmax])
    resized_frame_cropped = resized_frame[ymin:ymax, xmin:xmax]
    end_time = time.time()

    return track_id, current_final_ocr,resized_frame_cropped,current_id,last_id,do_ocr,xmin,ymin


def perform_ocr(ocr, track_id, text_storage, current_final_ocr, resized_frame_cropped,do_ocr):
    start_time = time.time()
    if do_ocr:
        result = ocr.ocr(resized_frame_cropped)
        for res in result:
            if res:
                for line in res:
                    txt = line[1][0]
                    txt = re.sub(r'[^a-zA-Z0-9]', '', txt)
                    txt = txt.replace("0", "O")
                    print(f"text for {track_id} is {txt}" )
                    if track_id not in text_storage:
                        text_storage[track_id] = []
                    text_storage[track_id].append(txt)
                    text_counter = Counter(text_storage[track_id])
                    most_common_text, most_common_count = text_counter.most_common(1)[0]

                    if most_common_count > 5:
                        do_ocr = False
                        current_final_ocr = most_common_text
                        print("stopping OCR")
                    else:
                        significant = True
                        for text, count in text_counter.items():
                            if text != most_common_text and not ((most_common_count > 4 * count)):
                                significant = False
                                break

                        if significant:

                            current_final_ocr = most_common_text
                        else:
                            # Sort the texts first by length and then by count
                            longest_texts = sorted(text_counter.keys(), key=lambda t: (-len(t), -text_counter[t]))
                            for longest_text in longest_texts:
                                if text_counter[longest_text] >= 0.25 * most_common_count:
                                    final_text = longest_text
                                    current_final_ocr = longest_text
                                    break
                        print(f'Current final text for ID {track_id}: {current_final_ocr}')
        end_time = time.time()
        print('Total time taken to OCR, in seconds: ', end_time - start_time)
        return current_final_ocr, do_ocr


def pipline(video_path):
    frame_counter=0
    cap = cv2.VideoCapture(video_path)
    ocr = initialize_ocr()
    model = initialize_model()
    tracker = initialize_tracker()
    current_id=None
    last_id=None
    text_storage = {}
    current_final_ocr="No Detection"
    do_ocr = True
    output_dir = './output_version2'
    resized_frame_cropped=None
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every 5 frames
        if frame_counter % 4 != 0:
            frame_counter += 1
            continue

        resized_frame,track_results=process_frame(frame, model, tracker)
        for track in track_results:
            track_id, current_final_ocr, resized_frame_cropped, current_id, last_id, do_ocr, xmin, ymin= update_text_storage_and_write_results(track,last_id, current_id, current_final_ocr, text_storage, output_dir, resized_frame, frame_counter,resized_frame_cropped,do_ocr)
            #print(f"for frame {frame_counter} the id is {track_id} the current id is {current_id} the last id is {last_id}" )


            if do_ocr:
                current_final_ocr, do_ocr =perform_ocr(ocr, track_id, text_storage, current_final_ocr, resized_frame_cropped,do_ocr)
            cv2.putText(resized_frame, current_final_ocr, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)
        tracker.plot_results(resized_frame, show_trajectories=False)
        # break on pressing q or space

        cv2.imshow('BoxMOT detection', resized_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break

        frame_counter += 1
    with open("./ALPR_results_version2.txt", 'a') as file:
        print("code reached")
        file.write(f"ID:{current_id} Plate Number:{current_final_ocr}\n")
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    cap.release()
    cv2.destroyAllWindows()

video_path = r'C:\Users\HHR6\PycharmProjects\ALPR\Number(license) plate detection and recognition using CNN.mp4'

pipline(video_path)

