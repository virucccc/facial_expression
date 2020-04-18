import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import dlib
import time

from utils import label_map_util
from utils import vis_util
from utils import func


# Это необходимо, поскольку записная книжка хранится в папке object_detection.
sys.path.append("..")
# Дополнительные параметры
MARKS_DISABLE = True
EMOTION_REC_DISABLE = False
DOWNSCALE = 1

# Загрузка landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Имя каталога, содержащего модуль обнаружения объекта, который мы используем.
MODEL_NAME = 'inference_graph'

# Получаем путь к текущему каталогу
CWD_PATH = os.getcwd()

# Путь к .pb файлу, который содержит используемую модель для обнаружения объекта.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, sys.argv[2])

# Путь к файлу с метками
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Путь к видео
PATH_TO_VIDEO = sys.argv[1]

# Количество классов, которые может определить детектор объекта
NUM_CLASSES = 8

# Загрузка карты с метками
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Загрузка TensorFlow в память.
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=config)

# Входные и выходные тензоры (данные) для классификатора обнаружения объекта
# Входной тензор изображения
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Выходными тензорами являются поля обнаружения, оценки и классы.
# Каждый блок представляет часть изображения, где был обнаружен конкретный объект.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Каждая оценка представляет собой уровень доверия для каждого из объектов.
# Оценки и метки отображаются на результате изображения.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Количество обнаруженных объектов
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Открываем видео
video = cv2.VideoCapture(PATH_TO_VIDEO)

# Загрузка TensorFlow до рендеринга видео
sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: np.expand_dims(np.zeros((1, 1, 3), np.uint8), axis=0)})

video_fps = video.get(cv2.CAP_PROP_FPS)
total_frame = 0
start = time.time()

while video.isOpened():
    ret, frame = video.read()
    h, w = frame.shape[:2]

    # Подсчет fps
    time1 = time.time()
    total_frame += 1

    # Масштабирование входного изображения
    resize = cv2.resize(frame, (int(w / DOWNSCALE), int(h / DOWNSCALE)))
    rh, rw = resize.shape[:2]

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) > 0:
        for face in faces:
            x1 = int(face.left())
            y1 = int(face.top())
            x2 = int(face.right())
            y2 = int(face.bottom())
            avg = int(((abs(x1 - x2) + abs(y1 - y2)) / 2) * 0.2)
            x1 = func.correct(x1 - avg, rw)
            y1 = func.correct(y1 - avg, rh)
            x2 = func.correct(x2 + avg, rw)
            y2 = func.correct(y2 + avg, rh)
            cropped = resize[y1:y2, x1:x2]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if not MARKS_DISABLE:
                landmarks = predictor(gray, face)
                # Левый глаз
                func.marks_draw(resize, landmarks, 36, 42)
                # Правый глаз
                func.marks_draw(resize, landmarks, 42, 48)
                # Рот
                func.marks_draw(resize, landmarks, 60, 68)

            if not EMOTION_REC_DISABLE:
                frame_expanded = np.expand_dims(cropped, axis=0)

                # t = time.time()
                # Обнаружение, запустив модель с изображением в качестве входных данных
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                # print(time.time() - t)

                # Результат обнаружения
                det = vis_util.get_boxes_and_labels(
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=0.7)

                # Выводится результат обнаружения
                if det is not None:
                    vis_util.draw(cropped, det)

    time2 = time.time()
    cv2.putText(resize, 'Time: {:.2f}s'.format(time2 - start), (1, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (255, 0, 255))
    cv2.putText(resize, 'FPS: {:.0f}'.format(1 / (time2 - time1)), (1, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (255, 0, 255))
    cv2.putText(resize, 'AVG: {:.0f}'.format(total_frame / (time2 - start)), (1, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (255, 0, 255))
    time1 = time.time()

    # Отображение результатов
    cv2.imshow('Facial expression', resize)

    # ESC, чтобы закрыть
    if cv2.waitKey(1) == 27:
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()
