import cv2
import logging as log

def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)

class ClassificationVisualizer:
    def __init__(self) -> None:
        pass

    def draw_labels(self, frame, classifications):
        class_label = ""
        if classifications:
            class_label = classifications[0][1]
        font_scale = 0.7
        label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
        initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(classifications)) + 1)

        if (initial_labels_pos < 0):
            initial_labels_pos = label_height
            log.warning('Too much labels to display on this frame, some will be omitted')
        offset_y = initial_labels_pos

        header = "Label:     Score:"
        label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        put_highlighted_text(frame, header, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

        for idx, class_label, score in classifications:
            label = '{}. {}    {:.2f}'.format(idx, class_label, score)
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
            offset_y += int(label_height * 1.5)
            put_highlighted_text(frame, label, (frame.shape[1] - label_width, offset_y),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
        return frame

    @staticmethod
    def print_raw_results(classifications, frame_id):
        label_max_len = 0
        if classifications:
            label_max_len = len(max([cl[1] for cl in classifications], key=len))

        log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))

        if label_max_len != 0:
            log.debug(' Class ID | {:^{width}s}| Confidence '.format('Label', width=label_max_len))
        else:
            log.debug(' Class ID | Confidence ')

        for class_id, class_label, score in classifications:
            if class_label != "":
                log.debug('{:^9} | {:^{width}s}| {:^10f} '.format(class_id, class_label, score, width=label_max_len))
            else:
                log.debug('{:^9} | {:^10f} '.format(class_id, score))