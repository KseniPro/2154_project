import base64
import os
import tempfile
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from extract_raster import convert_pdf_to_images

# Streamlit UI
st.write("## Веб-сервис сравнения версий рабочей документации архитектурных решений для строительных объектов")

st.write("### 1. Загрузка файлов")

file_type = st.radio(
    "Выберите тип загружаемых данных",
    ["PDF-файлы", "Изображения"],
    index=0,
)

uploaded_files = []
if file_type == "PDF-файлы":
    uploaded_files = st.file_uploader("Выберите два PDF-файла", type=["pdf"],
                                      accept_multiple_files=True)
    page_number = st.text_input("Номер страницы для сравнения:")
    st.write(f"Выбранная страница: {page_number}")

elif file_type == "Изображения":
    uploaded_files = st.file_uploader("Выберите два изображения", type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)
# Если выбрано ровно 2 файла
if uploaded_files and len(uploaded_files) == 2:
    suffix = ".pdf" if file_type == "PDF-файлы" else ".png"

    temp_files = []
    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.read())
        temp_file.flush()  # убедимся, что данные записаны
        temp_files.append(temp_file)

    file_paths = [temp_file.name for temp_file in temp_files]

    # Если PDF — сначала конвертация нужных страниц в изображения
    if file_type == "PDF-файлы":
        if not page_number.isdigit() or int(page_number) < 1:
            st.error("Введите корректный номер страницы (целое число >= 1)")
            st.stop()

        page = int(page_number)

        converted_images = []

        for pdf_path in file_paths:
            try:
                output_dir = tempfile.mkdtemp()
                convert_pdf_to_images(pdf_path, output_dir, image_format="png", dpi=150, page=page)
                # Найти конвертированное изображение
                image_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
                if image_files:
                    converted_images.append(image_files[0])
                else:
                    st.error(f"Не удалось извлечь страницу {page} из {pdf_path}")
                    st.stop()
            except ValueError as e:
                st.error(f"Ошибка: {str(e)}")
                st.stop()

        # Отображаем изображения
        st.image(converted_images[0], caption="Страница из PDF 1", use_container_width=True)
        st.image(converted_images[1], caption="Страница из PDF 2", use_container_width=True)

    else:
        # Если изображения
        f1_path, f2_path = file_paths
        st.image([uploaded_files[0], uploaded_files[1]], caption=["Изображение 1", "Изображение 2"], width=300)
else:
    st.error("Необходимо выбрать ровно два файла.")
    st.stop()

st.write("### 2. Сравнение изображений")

selected_algorithm = st.radio(
    "Выберите алгоритм сравнения изображений",
    ["Гомография (SIFT) для поворота/масштаба", "Фазовая корреляция для точного сдвига", "Попиксельное наложение изображений"],
    index=0,
)

# Связь названий с методами API
algorithm_map = {
    "Гомография (SIFT) для поворота/масштаба": "one",
    "Попиксельное наложение изображений": "two",
    "Фазовая корреляция для точного сдвига": "three",
}

selected_method = algorithm_map[selected_algorithm]

captions = {
    "aligned": "Выравненное изображение",
    "changed": "Результат сравнения",
    "diff": "Разница между изображениями",
}

if st.button("Сравнить изображения"):
    if file_type == "PDF-файлы":
        # Используем уже сконвертированные пути
        payload = {
            "img1_path": converted_images[0],
            "img2_path": converted_images[1],
        }
    else:
        # Если обычные изображения — создаем временные файлы
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            temp_paths.append(temp_file.name)

        payload = {
            "img1_path": temp_paths[0],
            "img2_path": temp_paths[1],
        }

    response = requests.post(
        f"http://two154-project.onrender.com/api/methods/?method={selected_method}",
        # f"http://127.0.0.1:8000/api/methods/?method={selected_method}",
        json=payload  # <-- отправляем как JSON
    )

    if response.status_code == 200:
        data = response.json()
        for key, b64_img in data.get("images", {}).items():
            img_data = base64.b64decode(b64_img)
            img = Image.open(BytesIO(img_data))
            st.image(img, caption=captions.get(key, key), use_container_width=True)
    else:
        try:
            st.error(f"Ошибка: {response.json().get('error')}")
        except Exception:
            st.error("Ошибка при получении ответа от сервера.")




