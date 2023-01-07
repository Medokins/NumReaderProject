## Analiza obrazów, projekt semestralny

- Nikodem Szwast
- Krzysztof Tłuszcz
- Jakub Ochman

---

### Obsługa programu

#### Uruchamianie programu

Projekt napisany został w języku python. W związku z tym pierwszym krokiem powinno być zainstalowanie wszystkich potrzebnych dependencies za pomocą komendy `pip install`.
Wszelkie potrzebne dependencies zawarte są w pliku `requirements.txt` zatem by poprawnie zainstalować zależności należy użyć komendy:

```
pip install requirements.txt
```

Po instalacji wszystkich zależności jedynym co pozostało do zrobienia jest uruchomienie pliku wejściowego `main.py`. Należy to zrobić przy użyciu komendy:

```
python main.py
```

#### Interfejs programu

Po uruchomieniu programu pojawia się okno z kilkoma funkcjami.

##### Browse

Funkcja ta pozwala wybrać zdjęcia poszczególnych cyfr lub ciągów cyfr, które później zostaną przeanalizowane przez program, po wyborze odpowiedniego zdjęcia należy kliknąć przycisk `Submit` by uruchomić funkcję. W terminalu pokaże się wtedy wynik działania programu w postaci odpowiedniej cyfry bądź ich ciągu.

##### Live View

Po uruchomieniu tej funkcji, program dostanie dostęp do kamery i ukaże podgląd zawierający obraz z `bounding boxami` wokół elementów rozpoznawanych jako cyfry. Po umieszczeniu w zakresie widoczności kartki z cyframi należy kliknąć przycisk `Q` by zakończyć podgląd kamery i pozwolić programowi przetworzyć znajdujące się na kamerze dane. Podobnie jak w funkcji `Browse` po przeanalizowaniu danych program w terminalu wypisze odpowiednią cyfrę lub ich ciąg.

---

### Działanie programu

W programie znajdują się dwa foldery z danymi:

- `handwrittenNumbers` - jest to folder w którym znajdują się przykładowe cyfry i ciągi cyfr których można użyć do testowania działania programu.
- `numReader.model` - automatycznie wygenerowany folder zawierający wytrenowany model sieci neuronowej.

Poza folderami w programie znajdują się również pliki w języku python:

- `settings.py` - zawiera zestaw globalnych ustawień sterujących działaniem programu.
- `model.py` - plik generujący wcześniej wspomniany folder z modelem sieci neuronowej.
- `data_preprocessor.py` - zawiera zestaw funkcji używanych w programie.
- `main.py` - plik wejściowy, zawiera cały `GUI` programu.

W związku z zawiłością plików `main.py`, `model.py`, `data_preprocessor.py` ich pełne opisy znajdują się poniżej.

#### main.py

Plik ten odpowiada za wygenerowanie `GUI` za pomocą biblioteki `PySimpleGUI` posiada on jedną funkcję `create_gui` która wywoływana jest po uruchomieniu pliku. To właśnie w tym pliku znajdziemy funkcje budujące opisywane wcześniej przyciski takie jak `Browse`, `Submit` oraz `Live View`. 
Przycisk `Live View` wykorzystuje funkcję `VideoCapture` z biblioteki `CV2` do przekazania obrazu z kamery do funkcji `getBoundingBoxes` znajdującej się w pliku `data_preprocessor.py`. 

```python

if event == "Live View":
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            _, frame = vid.read()
            last_frame = frame.copy()
            getBoundingBoxes(img = frame, visualize=True, live_view=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()
        cv2.destroyAllWindows()
        coordinates_array = getBoundingBoxes(last_frame, live_view=False)
        readDigits(coordinates_array, img=last_frame)

```

#### model.py

Jest to plik zawierający funkcję `trainNumReader` generującą i trenującą model sieci neuronowej.

Do wygenerowania modelu korzystamy z zbioru danych `mnist`. Po załadowaniu zbioru dzielimy go na dane testowe i trenujące.

```python

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

```

Podzielone dane normalizujemy w celu usunięcia możliwych ekstremów.

```python

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

```

Kolejnym krokiem jest utworzenie modelu i spłaszczenie danych, czyli z obrazków o wymiarach `28 x 28 px` utworzona zostaje jednowymiarowa tablica o wybierze `28^2` w celu zmniejszenia złożoności obliczeń.

```python

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))

```

Następnie stworzone zostają dwie warstwy neuronowe typu `Dense`każda po 128 neuronów z funkcjami aktywującymi `relu`.

```python

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

```

Ostatnim krokiem jest stworzenie ostatecznej warstwy wynikowej składającej się z 10 neuronów, każdy odpowiada za poszczególne cyfry `0, 1, ..., 9`.

```python

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 

```

Następnie następuje kompilacja modelu oraz jego trenowanie na wcześniej podzielonych danych.

```python

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=EPOCHS)

```

By otrzymywać informacje na temat procentowej skuteczności wytrenowanego modelu dodany został również system ewaluacji. Po ewaluacji model zostaje zapisany we wcześniej wspomnianym folderze `numReader.model`

```python	

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
model.save('numReader.model')

```

#### data_preprocessor.py

Plik ten składa się z trzech funkcji: `preprocessImage`, `getBoundingBoxes`oraz`readDigits`.

##### preprocessImage

Pierwszym krokiem działania tej funkcji jest utworzenie kwadratu z podanego jako argument obrazu poprzez wzięcie dłuższego boku prostokąta i dodanie odpowiedniej ilości białych pikseli by obraz miał równe boki.

```python

length = max(imageArray.shape[0:2]) + 5
squared_image = np.empty((length, length), np.uint8)
squared_image.fill(255)
ax,ay = (length - imageArray.shape[1]) // 2, (length - imageArray.shape[0])//2
squared_image[ay:imageArray.shape[0] + ay, ax:ax + imageArray.shape[1]] = imageArray

```

Kolejnym krokiem jest binaryzacja obrazu i jego inwersja w celu uzyskania odpowiedniego formatu obrazu, który zostanie przyjęty przez model.

```python

img = cv2.resize(squared_image, (28, 28))
_, blackAndWhiteImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
inverted = np.full(blackAndWhiteImage.shape,255) - blackAndWhiteImage
inverted = inverted.astype(np.uint8)

```

Ostatnim krokiem funkcji jest stworzenie i zwrócenie tablicy zawierającej piksele przygotowanego obrazu.

```python

tf_image = np.empty((1, 28, 28), dtype=np.double)
tf_image[0] = inverted

return tf_image

```

##### getBoundingBoxes

Pierwszym krokiem jest binaryzacja i inwersja obrazu.

```python
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
im = ~im
binaryIm = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -1)
```

Następnie znajdywane są obiekty, które posłużą do rysowania `bounding boxów`. 

```python

num_of_objects, labeledImage, componentStats, _ = cv2.connectedComponentsWithStats(binaryIm)
remainingComponentLabels = [i for i in range(1, num_of_objects) if componentStats[i][4] >= MIN_AREA_FILTER]
filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

```

Następnie za pomocą funkcji morfologicznych pozbywamy się niechcianych zakłóceń na obrazie.

```python

maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, CLOSING_INTERATIONS, cv2.BORDER_REFLECT101)

```

Kolejnym krokiem jest narysowanie `bounding boxów`.

```python

contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours_poly = [None] * len(contours)
boundRect = []

```

Ponieważ niektóre znalezione obramowania obiektów były niepoprawne, należy przefiltrować otrzymane obramowania.

```python

for i, c in enumerate(contours):
	if hierarchy[0][i][3] == -1:
		contours_poly[i] = cv2.approxPolyDP(c, 3, True)
		area = cv2.boundingRect(contours_poly[i])[2] * cv2.boundingRect(contours_poly[i])[3] 
		
		if area > 2000 and area < 10000 and live_view:
               	boundRect.append(cv2.boundingRect(contours_poly[i]))
		
		elif not live_view:
               	boundRect.append(cv2.boundingRect(contours_poly[i]))

```

Kolejny element kodu dotyczy widoku na żywo z kamerki, mianowicie rysuje on zielone obramowania wokół obiektów podczas gdy używamy funkcji `Live View`

```python

if visualize:
	imCopy = img.copy()
	
	for i in range(len(boundRect)):
		color = (0, 255, 0)
		cv2.rectangle(imCopy, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
	cv2.imshow('bounding boxes', imCopy)

```

Ostatnim krokiem funkcji jest utworzenie i zwrócenie tablicy obramowań zawierającej informacje o punkcie początkowym obramowania, jego szerokości i długości.

```python

coordinates_array = np.empty((len(boundRect), 4), dtype=np.int16)

	for i in range(len(boundRect)):       
		x_start, y_start, width, height = boundRect[i]

		coordinates_array[i][0] = x_start
		coordinates_array[i][1] = y_start
		coordinates_array[i][2] = x_start + width 
		coordinates_array[i][3] = y_start + height
    
    return coordinates_array

```

##### readDigits

Działanie funkcji rozpoczyna się od wczytania zapisanego modelu i zmiany koloru przekazanego jako argument obrazu na skale szarości

```python

model = tf.keras.models.load_model('numReader.model')
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

```

Następnie sortujemy `bounding boxy` znajdujące się na obrazie

```python

sortedBoundigBoxes = coordinates_array[coordinates_array[:,0].argsort()]

```

W kolejnym kroku wycinamy z obrazka zawartość `bounding boxów` i przekazujemy wycięte cyfry do wcześniej opisanej funkcji `preprocessImage`. Przetworzone w ten sposób obrazy wkładamy do modelu a wynik działania sieci neuronowej zapisujemy do wcześniej utworzonej tablicy `nums`.

```python

nums = []

for index in range(len(sortedBoundigBoxes)):
	single_digit = im[sortedBoundigBoxes[index][1]:sortedBoundigBoxes[index][3], sortedBoundigBoxes[index][0]:sortedBoundigBoxes[index][2]]
        boundingBoxConverted = preprocessImage(single_digit)
        nums.append(np.argmax(model.predict([boundingBoxConverted])))

```

Ostatnim krokiem jest wypisanie rozpoznanych cyfr.

```python

for number in nums:
	print(number, end=' ')

```

### Co nie działa

Pomimo wysokiego procentu skuteczności wytrenowanego modelu sieci neuronowej, faktyczna skuteczność działania programu jest niska. 
Na przykład używając używając zdjęcia cyfry `3` program odczytuje ją jako cyfrę `5`, podobne zjawisko jest obecne przy ciągu cyfry gdzie tylko niektóre z nich odczytywane są poprawnie.

Podczas działania funkcji `preprocessImage` obraz w związku z wieloma przekształceniami ulega nieznacznej dystorsji co znacząco wpływa na wyniki działania programu. 

### Podział pracy

- Nikodem Szwast - Autor funkcji logicznych w pliku `data_preprocessor.py`
- Jakub Ochman - Autor `GUI` oraz `Live View` w pliku `main.py`
- Krzysztof Tłuszcz - Autor dokumentacji `README.md`