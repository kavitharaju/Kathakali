'''Kathakali mudra recognition using mediapipe and vector similarity'''
import mediapipe as mp
import numpy as np
from sqlalchemy import select
import cv2
import csv

import database

mp_hands = mp.solutions.hands

def solve_affine( p1, p2, p3, p4, s1, s2, s3, s4 ):
    '''Estimate the tanformtaion matrix, given two sets of 4 points'''
    x_mat = np.transpose(np.matrix([p1,p2,p3,p4]))
    y_mat = np.transpose(np.matrix([s1,s2,s3,s4]))
    # add ones on the bottom of x and y
    x_mat = np.vstack((x_mat,[1,1,1,1]))
    y_mat = np.vstack((y_mat,[1,1,1,1]))
    # solve for A2
    A2_ = y_mat * x_mat.I
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is 
    return lambda x: (A2_*np.vstack((np.matrix(x).reshape(3,1),1)))[0:3,:]

def transform_coordinates(hand_landmarks):
    '''Move points from one coordinate system to another'''
    primary_system1 = (
        hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z)
    primary_system2 = (
        hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z)
    primary_system3 = (
        hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z)
    primary_system4 = (
        hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z)
    secondary_system1 = [0.5,0.75,0]
    secondary_system2 = [0.6,0.5,-0.02]
    secondary_system3 = [0.4,0.45,-0.01]
    secondary_system4 = [0.42,0.7,-0.03]
    transform_fn = solve_affine( primary_system1, primary_system2, 
        primary_system3, primary_system4,
        secondary_system1, secondary_system2,
        secondary_system3, secondary_system4 )
    for landmark in hand_landmarks.landmark:
        new = transform_fn((landmark.x, landmark.y, landmark.z))
        landmark.x = new[0]
        landmark.y = new[1]
        landmark.z = new[2]
    return hand_landmarks

def pose_detection_on_single_image(image_path, max_trials=3):
    '''initialize a model object, get pose co-ordinated in image coordinates'''
    feat_vect_raw = np.zeros(63)
    feat_vect_norm = np.zeros(63)
    left_right = None
    hand_results = None
    with mp_hands.Hands(
        static_image_mode=True,max_num_hands=1, min_detection_confidence=0.5) as hands:

        image_cv = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.resize(image_rgb, (260, 460))

        hand_results = hands.process(image_rgb)
    
        trial = 0
        while trial<max_trials:
            if not hand_results.multi_hand_world_landmarks:
                log.warning(f"Repeating hand pose estimation on {image_path}")
                hand_results = hands.process(image_rgb)
                trial += 1
            else:
                break
        if hand_results.multi_hand_landmarks:
            for hand, side in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                vect_index = 0
                for landmark in hand.landmark:
                    feat_vect_raw[vect_index] = landmark.x
                    feat_vect_raw[vect_index+1] = landmark.y
                    feat_vect_raw[vect_index+2] = landmark.z
                    vect_index += 3
                try:
                    hand = transform_coordinates(hand)
                except np.linalg.LinAlgError as exce:
                    raise GenericException(
                        "Cannot normalize the coordinates as matrix inversing failed!",
                        name="Pose Estimation Error", status_code=500) from exce
                vect_index = 0
                for landmark in hand.landmark:
                    feat_vect_norm[vect_index] = landmark.x
                    feat_vect_norm[vect_index+1] = landmark.y
                    feat_vect_norm[vect_index+2] = landmark.z
                    vect_index += 3
                
                left_right = side.classification[0].label
                if left_right == "Left":
                    break
        return hand_results, feat_vect_raw, feat_vect_norm, left_right

def mudra_recognize(media_path):
    '''Find normalized pose on image, find most similar match in DB, 
    enter match details in DB, return match'''
    db_sess = database.get_db()
    _, _, norm_feat, _ = pose_detection_on_single_image(media_path)
    if not np.any(norm_feat):
        raise GenericException(
                "Cant detect pose coordinates of the image",
                name="Pose Estimation Error", status_code=503)
    match = db_sess.query(database.KathakaliMudraVectors).order_by(
                        database.KathakaliMudraVectors.embedding.l2_distance(norm_feat)).first()
    pred = database.Trials(mediaId=media_id, label=match.label,task=schema.Task.MUDRA.value)
    db_sess.add(pred)
    db_sess.commit()
    db_sess.refresh(pred)
    return pred

def add_data_to_db(filepath):
    '''Opens the seed data file i tsv format and load the (label, embedding) pair to db table'''
    db_sess = database.get_db()
    with open(filepath, 'r', encoding='utf-8') as vector_file:
        reader = csv.reader(vector_file, delimiter="\t")
        db_content = []
        for row in reader:
            db_content.append(database.KathakaliMudraVectors(label=row[0], embedding=eval(row[1])))
        db_sess.add_all(db_content)
        db_sess.commit()



# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np

# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# def draw_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image