import tempfile
from rest_framework.request import Request

import cv2
import base64
import numpy as np
from rest_framework.response import Response
from rest_framework.views import APIView

from app.algorithm import detect_differences, pixel_pairwise, align_with_phase_correlation, visualize_difference

def decode_and_save_image(base64_str):
    img_data = base64.b64decode(base64_str)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(img_data)
    temp_file.flush()
    return temp_file.name # путь к файлу

def add_legend(image):
    # Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    box_height = 30
    box_width = 20
    spacing = 10
    text_offset = 10
    text1 = "Pixels present in the first image and missing in the second image"
    text2 = "Pixels present in the second image and missing in the first image"

    legend_height = 2 * (box_height + spacing)

    # Create a white canvas for the legend
    legend = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255

    # First legend entry (blue)
    y1 = spacing
    cv2.rectangle(legend, (spacing, y1), (spacing + box_width, y1 + box_height), (255, 0, 0), -1)
    cv2.putText(legend, text1, (spacing + box_width + text_offset, y1 + box_height - 8), font, font_scale, (0, 0, 0), font_thickness)

    # Second legend entry (red)
    y2 = y1 + box_height + spacing
    cv2.rectangle(legend, (spacing, y2), (spacing + box_width, y2 + box_height), (0, 0, 255), -1)
    cv2.putText(legend, text2, (spacing + box_width + text_offset, y2 + box_height - 8), font, font_scale, (0, 0, 0), font_thickness)

    # Append the legend to the bottom of the image
    result = np.vstack((image, legend))
    return result

class AlgorithmsPostView(APIView):
    def post(self, request: Request):
        alignment_method = request.query_params.get("method")
        print(alignment_method)

        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        img2 = None
        if alignment_method == "phase_correlation":
            img2, _ = align_with_phase_correlation(img1_path, img2_path)
            if img2 is None:
                return Response({"error": "Failed to calculate difference"}, status=400)

        elif alignment_method == "sift":
            img2, _ = detect_differences(img1_path, img2_path)
            if img2 is None:
                return Response({"error": "Недостаточно совпадений для гомографии"}, status=400)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        result = visualize_difference(img1, img2)
        result = add_legend(result)
        _, changed_buffer = cv2.imencode('.jpg', result)
        result_b64 = base64.b64encode(changed_buffer).decode("utf-8")
        
        return Response({
            "images": {
                "changed": result_b64,
            }
        })

    def method_one(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        aligned_img, changed_area = detect_differences(img1_path, img2_path)

        if aligned_img is None:
            return Response({"error": "Недостаточно совпадений для гомографии"}, status=400)

        _, aligned_buffer = cv2.imencode('.jpg', aligned_img)
        aligned_b64 = base64.b64encode(aligned_buffer).decode("utf-8")

        _, changed_buffer = cv2.imencode('.jpg', changed_area)
        changed_b64 = base64.b64encode(changed_buffer).decode("utf-8")

        return Response({
            "images": {
                "aligned": aligned_b64,
                "changed": changed_b64,
            }
        })


    def method_two(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        changed_area = pixel_pairwise(img1_path, img2_path)

        if changed_area is None:
            return Response({"error": "Failed to calculate difference"}, status=400)

        # Конвертируем изображение в base64 для отправки в ответ
        _, buffer = cv2.imencode('.jpg', changed_area)
        changed_b64 = base64.b64encode(buffer).decode("utf-8")

        return Response({
            "images": {
                "changed": changed_b64,
            }
        })

    def method_three(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        # Совмещение изображений фазовой корреляцией
        aligned, _ = align_with_phase_correlation(img1_path, img2_path)

        if aligned is None:
            return Response({"error": "Failed to calculate difference"}, status=400)

        _, buffer = cv2.imencode('.jpg', aligned)
        changed_b64 = base64.b64encode(buffer).decode("utf-8")

        return Response({
            "images": {
                "aligned": changed_b64,
            }
        })
