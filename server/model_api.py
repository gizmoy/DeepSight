import os
import cv2
import uuid
import random
import colorsys
import threading
import numpy as np

from werkzeug.utils import secure_filename

from model import Model


def get_model_api():
    """Returns lambda function for image and video model api"""

    # Initialize model once
    model = Model()

    # Initialize colors
    colors = []
    v_min = 0.8
    for i in range(90000):
        h = random.random() * 359
        s = random.random() * 2
        v = (v_min + (1. - v_min) * random.random());
        color = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
        colors.append(color)

    def image_api(file):
        
        # Decode image to numpy array
        img_str = file.stream.read()
        img_np = np.fromstring(img_str, np.uint8)
        img_in = cv2.imdecode(img_np, cv2.IMREAD_COLOR) 
        img_in = np.expand_dims(img_in, axis=0)

        # Perform prediction
        out = model.predict(img_in)

        # Add colors for client
        for obj in out[0]:
            r, g, b = colors[obj['id']]
            obj['color'] = 'rgb({},{},{})'.format(r,g,b)
        
        return out

    def video_api(file, folder):

        # Create new filename
        guid = str(uuid.uuid4().hex)
        extension = file.mimetype.split('/')[-1]
        new_filename = guid + '.' + extension
        new_filename = secure_filename(new_filename)

        # Save file
        new_path = os.path.join(folder, new_filename)
        file.save(new_path)

        # Create video capture
        cap = cv2.VideoCapture(new_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Create video writer
        dim = (int(v_w), int(v_h))            
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out_path = os.path.join(folder, 'p_' + new_filename)
        wrt = cv2.VideoWriter(out_path, fourcc, fps, dim, True)

        # Process video frame by frame
        frames = []
        for i in range(int(frame_num)):
            if cap.isOpened():
                correct, frame = cap.read()
                if correct:
                    frames.append(frame)
                
            # Check whether are some frames to process
            if len(frames) == 0:
                break
                
            # At 8 steps perform prediction
            if i % 23 == 0 and i > 0:
                input = np.array(frames)
                out = model.predict(input)
            
                # Draw bounding boxes and labels
                for j, frame in enumerate(frames):
                    objects = out[j]
                    for obj in objects:

                        # Unpack values
                        x = obj['bbox']['x']
                        y = obj['bbox']['y']
                        w = obj['bbox']['w']
                        h = obj['bbox']['h']

                        # Calculate top-left and bottom-right corner 
                        l = x
                        r = x + w   
                        t = y
                        b = y + h 

                        # Draw bounding box and label
                        color = colors[obj['id']]
                        cv2.rectangle(frame, (l,t), (r,b), color, 5)
                        cv2.putText(frame, obj['label'], (l, t-15), cv2.FONT_HERSHEY_SIMPLEX, v_h * 1.1e-3, color, int(v_h * 4e-3))

                    # Write frame with bounding boxes and labels
                    wrt.write(frame)
                
                # Clear frames
                frames = []

        # Release resources
        cap.release()
        wrt.release()

        return out_path, new_path

    return image_api, video_api