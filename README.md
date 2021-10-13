# Introducción

En el siguiente documento se muestra un paso a paso de cómo ejecutar un programa de reconocimiento de imágenes en un
video, es importante mencionar que para el desarrollo de este proyecto se utiliza YOLO el cual es un algoritmo capas de
distinguir objetos en una imagen dada, este proyecto tiene como objetivo ayudar al problema de reconocimiento de
imágenes sensibles en una película, se espera que este programa pueda generar un gráfico en donde se muestre en qué
tiempo específico de la película se encuentran estas imágenes sensibles, para que los usuarios puedan tomar acción como
consideren una vez tienen esta información.

## Paso 1: Preparación de YOLOv3

En primer lugar se inicializa un nuevo proyecto en Python. Se puede utilizar la siguiente estructura como base.

```
project
│   README.md
│   main.py
│   .gitignore    
│
└───yolo-coco
│   │   obj.names
│   │   yolov3.cfg
│   │   yolov3.weights
│   
└───output
│    
└───videos
    │   sample_video1.mp4
    │   sample_video2.mp4
```

En donde en se guardan los videos en la carpeta videos y los resultados en la carpeta output.

### Instalación de paquetes

Se instalan los paquetes necesarios, opencv para poder utilizar YOLOv3 y matplotlib para graficar las secciones del
video donde se encontraron imágenes sensibles.

```
pip install opencv-python matplotlib
```

## Programación del sistema de detección

Se va a ocupar dos variables globales con las rutas de las carpetas de donde se debe cargar los vídeos y donde se deben
guardar.

```python
VIDEOS_PATH = './videos'
OUTPUT_PATH = './output'
```

Y después es necesario una lista para manejar los intervalos de los frames donde aparecen los objetos. Para detectar los
intervalos lo que se hará es que si se vuelve a detectar un objeto del mismo tipo en el siguiente frame se asumirá que
el mismo objeto que el frame anterior. Dado a que es posible que en algunos frames la imagen salga borrosa y que la
detección no funcione bien se dará un margen de error de 30 frames antes de abrir un nuevo intervalo de frames para el
objeto.

```python
FRAMES_ROOM = 30
intervals = []
```

Ejemplo de la variable intervals

```
intervals = [[[42,54],[65,85]],[[142,200]],[[21,72]]]
```

Las lista más inferiores son los intervalos de frames entre los que se dectecto un objeto y las lista más exteriores
representan cada una un objeto distinto.

Ya teniendo en claro esto, además son necesarias dos funciones que ayudaran a convertir frames a segundos, esto para que
la información de los gráficos sea más legible y fácil de entender.

```python
def frames_to_seconds(frames, movie_frames):
    return frames / movie_frames


def frames_in_movie(video):
    return video.get(cv2.CAP_PROP_FPS)
```

Depués se tiene la funcion que genera el gráfico a partir de la lista de intervalos, para esto se hace uso de la
librería matplotlib para generar un diagrama de Gantt. Este es un ejemplo para las tres variables que se hacen uso en
este sistema de detección, pero se puede adaptar la función según las necesidades

```python
def generate_graph(video_name, fps):
    # Declaring a figure "gnt"
    fig, gnt = plt.subplots()

    # Setting Y-axis limits
    gnt.set_ylim(0, 50)

    # Setting X-axis limits
    # gnt.set_xlim(0, 160)

    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Segundos desde el inicio')
    gnt.set_ylabel('Objetos')

    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Pistola', 'Fuego', 'Rifle'])

    # Setting graph attribute
    # gnt.grid(True)

    gun_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                  for range in intervals[0]]
    gnt.broken_barh(gun_tuples, (10, 9), facecolors='tab:blue')

    fire_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                   for range in intervals[1]]
    gnt.broken_barh(fire_tuples, (20, 9), facecolors=('tab:red'))

    rifle_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                    for range in intervals[2]]
    gnt.broken_barh(rifle_tuples, (30, 9), facecolors=('tab:orange'))

    folder_path = f"{OUTPUT_PATH}/{video_name.replace('.mp4', '')}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    # save image
    plt.savefig(f"{folder_path}/graph.png")
```

### Funciones para detección con YOLOv3

Primeramente se tiene la función que se encarga de inicializar el sistema de detección y los nombres de los objetos a
detectar, para esto se hacen uso de los archivos cargados en la parte anterior.

```python
def initialize_yolo():
    net = cv2.dnn.readNet("./yolo-coco/yolov3.weights", "./yolo-coco/yolov3.cfg")
    classes = []
    with open("./yolo-coco/obj.names", "r") as f:
        for line in f.readlines():
            classes.append(line.strip())
            intervals.append([])

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers
```

Después se tiene la función que se hace cargo de detectar los objetos en un frame dado. En esta función se debe
configurar el tamaño y la escala a la que se redimensionaran las imágenes antes de hacer la detección.

```python
def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs
```

Esta función se encarga de procesar los datos detectados para generar los cuadros de detección y de filtrar las
coincidencias que tienen muy baja probabilidad, en este caso esta configurado para que filtre aquellas menores a 0.3. En
esta función también se agregó el sistema de intervalos.

```python
def get_box_dimensions(curr_frame_idx, outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                if not intervals[class_id] or curr_frame_idx - intervals[class_id][-1][1] > FRAMES_ROOM:
                    intervals[class_id].append([curr_frame_idx, curr_frame_idx])
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
                else:
                    intervals[class_id][-1][1] = curr_frame_idx
    return boxes, confs, class_ids
```

Está otra función se encarga de generar una imagen con los cuadros de detección y de guardarla al disco duro para que
este pueda ver si se detectaron datos sensibles y pueda revisar si es un falso positivo o no. Esta función se llama solo
al inicio de cada intervalo de detección, esto con el fin de guardar imagenes innecesarias.

```python
def draw_labels(video_name, curr_frame_idx, fps, boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (800, 600))
    # create folder
    folder_path = f"{OUTPUT_PATH}/{video_name.replace('.mp4', '')}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    # save image
    cv2.imwrite(f"{folder_path}/{str(int(frames_to_seconds(curr_frame_idx, fps)))}.jpg", img)
```

Finalmente se tiene la función que se encarga de tomar un video, descomponerlo en frame y llamar a las otras según sea
necesario.

```python
def process_video(video):
    model, classes, colors, output_layers = initialize_yolo()
    cap = cv2.VideoCapture(VIDEOS_PATH + '/' + video)
    curr_frame_idx = 0
    fps = frames_in_movie(cap)

    while True:
        grabbed, frame = cap.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(curr_frame_idx, outputs, height, width)
        if boxes:
            draw_labels(video, curr_frame_idx, fps, boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        curr_frame_idx += 1

    cap.release()
    generate_graph(video, fps)
```

## Paso 2: Ejecución del código lineal

```python
videos = os.listdir(VIDEOS_PATH)

print("-----NO THREADS-----")
start = time.time()
for video in videos:
    process_video(video)
    for index, _ in enumerate(intervals):
        intervals[index] = []
end = time.time()
print(f"Time: {end - start}")
```

## Paso 3: Ejecución del código en paralelo

### Opción A

La primera opción que se consideró para optimizar el código mediante el uso de threads fue colocar todo el trabajado de
postprocesado de las imagenes en en un thread a parte, de esta manera no se interrumpe el proceso de detección. Sin
embargo no parece tener mayor impacto ya que el programa ya filtra la mayor parte del postprocesado y solo guarda
imágenes la primera vez que hay algún hit.

Para esta opción es necesario encapsular el postprocesado en un método aparte, el cual será utilizado para incializar el
thread.

```python
def postprocessing(video, curr_frame_idx, outputs, height, width, fps, colors, classes, frame):
    boxes, confs, class_ids = get_box_dimensions(curr_frame_idx, outputs, height, width)
    if boxes:
        draw_labels(video, curr_frame_idx, fps, boxes, confs, colors, class_ids, classes, frame)
```

Y también una versión modificada del process_video, en donde se haga uso del thread.

```python
def postproccessing_thread_process_video(video):
    model, classes, colors, output_layers = initialize_yolo()
    cap = cv2.VideoCapture(VIDEOS_PATH + '/' + video)
    curr_frame_idx = 0
    fps = frames_in_movie(cap)

    while True:
        grabbed, frame = cap.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        x = threading.Thread(target=postprocessing,
                             args=(video, curr_frame_idx, outputs, height, width, fps, colors, classes, frame))
        x.start()

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        curr_frame_idx += 1

    x.join()
    cap.release()
    generate_graph(video, fps)
```

Y se puede probar de la siguiente manera:

```python
print("-----POSTPROCESSING THREADS-----")
start = time.time()
for video in videos:
    postproccessing_thread_process_video(video)
    for index, _ in enumerate(intervals):
        intervals[index] = []
end = time.time()
print(f"Time: {end - start}")
```

### Opción B

La segunda opción fue iniciar un proceso por archivo hasta agotar los cores del CPU. De este manera se evitan problemas
de sincronización con OpenCV, la variable intervals y el postprocesado. En este caso en el entorno de prueba utilizado
fueron 8. En esta versión si se alcanzó a notar una mejoría de rendimiento.

Para esta versión solo es necesario el uso de pool.map() y asegurarse de que el código principal esté encapsulado en **if __name__ == '__main__':** con el fin de evitar creación recursiva de procesos.

```python
print("-----PROCESS PER FILE-----")
start = time.time()
pool = multiprocessing.Pool(processes=8)
pool.map(process_video, videos)
pool.close()
end = time.time()
print(f"Time: {end - start}")
```

## Resultados finales
