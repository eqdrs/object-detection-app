# App Utils

import struct
import six
import collections
import cv2
import datetime
from threading import Thread
from matplotlib import colors


class FPS:
    def __init__(self):
        # Armazena a hora de início, o tempo de término e o número total de frames que foram examinados entre os intervalos de início e final
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # Start do timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # Stop do timer
        self._end = datetime.datetime.now()

    def update(self):
        # Incrementar o número total de frames examinados durante o intervalos de início e fim
        self._numFrames += 1

    def elapsed(self):
        # Devolve o número total de segundos entre o início e intervalo final
        return (self._end - self._start).total_seconds()

    def fps(self):
        # Computa os frames (aproximados) por segundo
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # Inicializa o fluxo da câmera de vídeo e lê o primeiro frame do fluxo
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # Inicializa a variável usada para indicar se o segmento deve seja parado
        self.stopped = False

    def start(self):
        # Inicia o segmento para ler os frames do fluxo de vídeo
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Mantém o loop infinitamente até a thread parar
        while True:
            # Se a variável da thread da linha estiver definida, pare a thread
            if self.stopped:
                return

            # Caso contrário, leia o próximo frame do fluxo
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Retorne o frame mais recentemente lido
        return self.frame

    def stop(self):
        # Indica que a thread deve ser interrompida
        self.stopped = True


def standard_colors():
    colors = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    return colors


def color_name_to_rgb():
    colors_rgb = []
    for key, value in colors.cnames.items():
        colors_rgb.append((key, struct.unpack('BBB', bytes.fromhex(value.replace('#', '')))))
    return dict(colors_rgb)


def draw_boxes_and_labels(
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        keypoints=None,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False):
    """Retorna as coordenadas das caixas, os nomes das classes e as cores

    Args:
      boxes: numpy array shape [N, 4]
      classes: numpy array shape [N]
      scores: numpy array shape [N] or None
      category_index: um dicionário contendo dicionários de categoria
      keypoints: numpy array shape [N, num_keypoints, 2]
      max_boxes_to_draw: número máximo de caixas a serem visualizadas. 
      min_score_thresh: Limite de score mínima para uma caixa a ser visualizada
      agnostic_mode: boolean (default: False) 
    """
    # Cria uma sequência de exibição (e cor) para cada local da caixa e agrupa todas as caixas que correspondem ao mesmo local.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100 * scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = standard_colors()[
                        classes[i] % len(standard_colors())]

    # Armazena todas as coordenadas das caixas, nomes de classe e cores
    color_rgb = color_name_to_rgb()
    rect_points = []
    class_names = []
    class_colors = []
    for box, color in six.iteritems(box_to_color_map):
        ymin, xmin, ymax, xmax = box
        rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
        class_names.append(box_to_display_str_map[box])
        class_colors.append(color_rgb[color.lower()])
    return rect_points, class_names, class_colors
