# Detection-and-classification-brake-lights- 
[![cpp-linter](https://github.com/cpp-linter/cpp-linter-action/actions/workflows/cpp-linter.yml/badge.svg)](https://github.com/Lisa-creates/Detection-and-classification-brake-lights-/actions/workflows/linter.yml) 

### Подготовка к запуску
#### Установка opencv
##### Linux
1. Установите пакеты, требующиеся для сборки библиотеки.
   На Debian/Ubuntu:
   `sudo apt update && sudo apt install cmake g++ wget unzip`
   На Fedora:
   `sudo dnf install cmake g++ wget unzip`
2. Загрузите и распакуйте библиотеку:
   `wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip`
   `wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip`
   `unzip opencv.zipunzip opencv_contrib.zip`
3. Создайте директорию для сборки и перейдите туда:
   `mkdir -p build && cd build`
4. Сконфигурируйте сборку:
   `cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x`
5. Соберите библиотеку:
   `cmake --build .`
##### Windows
1. Загрузите exe-файл с [Sourceforge](https://sourceforge.net/projects/opencvlibrary/files/4.9.0/).
2. Убедитесь, что у вас есть права администратора. Распакуйте самораспаковывающийся архив.
3. Проверьте установку по выбранному пути.
#### Сборка проекта
1. Клонируйте проект
   `git clone https://github.com/Lisa-creates/Detection-and-classification-brake-lights-.git`
2.  Перейдите в созданную директорию
    `cd Detection-and-classification-brake-lights-`
4. Сконфигурируйте cmake, `path/to/opencv` - директория, куда была установлена OpenCV
   `cmake -DOPENCV_PATH=path/to/opencv .`
5. Запустите сборку
   `cmake --build .`
### Запуск
##### Linux
`./brake_lights_status_classification`
##### Windows
`.\brake_lights_status_classification.exe` 
##### Параметры командной строки 
-h, --help - встроенная справка  
-a, --action - запуск в одном из трёх режимов 
##### Режимы запуска 
1 - Для обработки видеофайлов, чтобы использовать нужно сначала обучить модель на датасете (команда 2) 

2 - Для обучения модели на датасете 

3 - Для тестирования работы детектора (нужен датасет с размеченными картинками) 
