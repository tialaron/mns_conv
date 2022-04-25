import streamlit as st

#import cv2
from PIL import Image #Отрисовка изображений
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")
st.title("Распознавание рукописных цифр искусственной нейронной сетью (ИНС)")

txt = st.text_area("Искусственная нейронная сеть", """Искусственная нейронная сеть - это математическая модель настоящей нейронной сети, 
    то есть мозга. На практике, это обучаемый под требуемую задачу инструмент.
    Искусственная нейронная сеть представляет собой набор матриц, с которыми работают по законам линейной алгебры. 
    Тем не менее, проще представить её как набор слоёв нейронов, связанных между собой 
    засчёт входных и выходных связей. Различают внешние слои - входной и выходной, и внутренние, находящиеся между ними.
    У каждого отдельного нейрона, например, перцептрона, может быть несколько входных связей, у каждой из связей - свой множитель усиления 
    (ослабления) влияния связи - весовой коэффициент, или вес. На выходе нейрона действует функция активации, засчёт нелинейностей 
    функций активации и подбора параметров-весов на входе нейрона, нейронная сеть и может обучаться.
    """)

header_names = ["Перцептрон", "Функции активации", "Полносвязная нейронная сеть", "Градиентный поиск минимума",
                "Скорость обучения (момент)",
                "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                "Картинка хранится как массив",
                "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                ]
subheader_names = ["Схема перцептрона:", "Функции активации", "Полносвязная нейронная сеть",
                   "Градиентный поиск минимума", "Скорость обучения (момент)",
                   "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                   "Картинка хранится как массив",
                   "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                   ]
file_names = ["perceptron", "activation_functions", "fully_connected_NN", "gradient_decay", "gradient_momentum",
              "digits", "one_digit", "digit28x28", "data_to_line",
              "model_2d", "accuracy", "categorical_crossentropy", "confusion_matrix_2d_model"
              ]
text_headers = ["Информация о перцептроне", "Функции активации", "Полносвязная нейронная сеть",
                "Градиентный поиск минимума", "Скорость обучения (момент)",
                "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                "Картинка хранится как массив",
                "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                ]
texts = ["Перцептрон - математический аналог нейрона. Из них состоит простейшая ИНС. На выходе перцептрона получаем математическую функцию нескольких подаваемых в него данных. И сам перцептрон по сути - математическая функция.Важные понятия: вес, смещение, функция активации. Вес - это число, регулирующее, насколько сильно воспринимается воздействие с данного входа. Смещение - дополнительная регулировка. Как в уравнении прямой, она позволяет нашей функции смещаться, чтобы она могла проходить не только через ноль. Функция активации добавляет перцептрону нелинейность на выходе, то есть зависимость выходных данных от входных становится не лежащей на одной прямой.Это позволяет сетям на основе перцептронов обучаться и находить сложные закономерности.",
         "Функции активации", "Полносвязная нейронная сеть", "Градиентный поиск минимума", "Скорость обучения (момент)",
         "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
         "Картинка хранится как массив",
         "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"

         ]
file_path = '/sysroot/home/user/Загрузки/PyProject/mnist_streamlit/venv/'

for header_name, subheader_name, file_name, text_header, text in zip(header_names, subheader_names, file_names, text_headers, texts):
    # st.subheader(header_name)
    with st.expander(header_name):
        col11, col12 = st.columns(2)

        with col11:
            with st.container():
                st.subheader(subheader_name)

                #image = cv2.imread(file_path + file_name + '.png')
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.open(file_path + file_name + '.png')
                st.image(image)

        with col12:
            with st.container():
                # st.subheader("Информация о перцептроне")

                txt = st.text_area(text_header, text, height=200)


def img_preprocess(img):
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    # st.write(type(img_array))

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    # st.write(img_array.shape)

    # make square shape
    img_height, img_width = img_array.shape[0], img_array.shape[1]
    img_center = int(img_width / 2)
    left_border = int(img_center - img_height / 2)
    right_border = int(img_center + img_height / 2)
    img_array1 = img_array[:, left_border:right_border, :]

    # Check the shape of img_array:
    # st.write(img_array1.shape)

    # convert n save
    im = Image.fromarray(img_array1)
    im.save("your_file_image.png")
    # image11 = Image.open('/sysroot/home/user/Загрузки/PyProject/mnist_streamlit/your_file_image.png')
    image11 = Image.open('your_file_image.png')
    img11 = image11.resize((28, 28), Image.ANTIALIAS)

    # convert image to one channel & Numpy array
    img12 = img11.convert("L")
    imgData = np.asarray(img12)

    # Calculate THRESHOLD_VALUE
    # assume dark digit & white sheet
    step_lobe = .4
    mid_img_color = np.sum(imgData) / imgData.size
    min_img_color = imgData.min()

    THRESHOLD_VALUE = int(mid_img_color - (mid_img_color - min_img_color) * step_lobe)

    print(mid_img_color)
    print(min_img_color)
    print(THRESHOLD_VALUE)

    thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
    imgData1 = np.expand_dims(thresholdedData, axis=0)
    return imgData1

def show_image(img):
  plt.imshow(Image.fromarray(img).convert('RGB')) #Отрисовка картинки .convert('RGB')
  plt.show()

col21 , col22 = st.columns(2)
with col21:
    with st.container():
        st.title("Примерно так выглядят цифры из базы")
        st.image('/sysroot/home/user/Загрузки/PyProject/mnist_streamlit/venv/mnist_example.png')
        #run = st.checkbox('Run')
        #FRAME_WINDOW = st.image([])
        #camera = cv2.VideoCapture(0)
        #Onclicktrue = st.button('Сделать снимок экрана', key=1676356)
        #while run:
        #    _, frame = camera.read()
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    FRAME_WINDOW.image(frame)
        #    if Onclicktrue:
        #        cv2.imwrite('test1.jpg', frame)
        #        Onclicktrue = False


        #else:
        #    st.write('Stopped')


with col22:
    with st.container():
        st.title('Снимок экрана')
        img_file_buffer = st.camera_input("Фото")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img_array = np.array(img)

            mnist_like = img_preprocess(img_array)

            model_2d = load_model('/sysroot/home/user/Загрузки/PyProject/mnist_streamlit/venv/mnist_2d.h5')

            #st.write(imgData1)

            y_predict1 = model_2d.predict(mnist_like)
            y_maxarg = np.argmax(y_predict1, axis=1)
            st.write(y_predict1)
            st.write('Нейронная сеть считает, что это ', )
            st.subheader(int(y_maxarg))













