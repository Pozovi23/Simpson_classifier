# Классификатор симпсонов

Моя задача — создать нейросеть, классифицирующую симпсонов. Для выполнения данного задания у меня есть датасет (https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/code). С него и начнем наши рассуждения.

-------------------------------------------------------------------------------------------------------------------------------------------
## Датасет
Для тренировочной + валидационной выборок использовались фото из директории data/simpson_dataset/, а для тестовой из data/kaggle_simpson_testset

Первым делом хотелось бы оценить общую картину, сколько фоток в каждом классе (в тренировочном и тестовом датасетах соответственно).

Тренировочный    |   Тестовый
:-------------------------:|:-------------------------:
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/distribution%20of%20photos%20in%20train%2Bvalidation%20BEFORE%20adding%20new%20photos.png)  |  ![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/distribution%20of%20photos%20in%20testset%20BEFORE%20adding%20new%20photos.png)


Такая ситуация мне не понравилась. Я решил добавить картинок в тренировочный датасет в 10, 19, 26, 40 классы, в тестовый в классы где было 0 картинок. Ситуация после добавления: 

Тренировочный   |   Тестовый
:-------------------------:|:-------------------------:
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/distribution%20of%20photos%20in%20train%2Bvalidation%20AFTER%20adding%20new%20photos.png)  |  ![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/distribution%20of%20photos%20in%20testset%20AFTER%20adding%20new%20photos.png)

После этого был написан скрипт для поиска повторок в тренировочном + валидационном датасете. Искал с помощью хеш-функции. Оказалось 6 пар повторяющихся:

| Герой                | Путь до картинки 1                                        | Путь до картинки 2                                        |
|----------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| Professor John Frink | `data/simpsons_dataset/professor_john_frink/pic_0014.jpg` | `data/simpsons_dataset/professor_john_frink/pic_0012.jpg` |
| Lenny Leonard        | `data/simpsons_dataset/lenny_leonard/pic_0262.jpg`        | `data/simpsons_dataset/lenny_leonard/pic_0257.jpg`        |
| Cletus Spuckler      | `data/simpsons_dataset/cletus_spuckler/pic_0010.jpg`      | `data/simpsons_dataset/cletus_spuckler/pic_0012.jpg`      |
| Rainier Wolfcastle   | `data/simpsons_dataset/rainier_wolfcastle/pic_0016.jpg`   | `data/simpsons_dataset/rainier_wolfcastle/pic_0011.jpg`   |
| Mayor Quimby         | `data/simpsons_dataset/mayor_quimby/pic_0054.jpg`         | `data/simpsons_dataset/mayor_quimby/pic_0176.jpg`         |
| Waylon Smithers      | `data/simpsons_dataset/waylon_smithers/pic_0038.jpg`      | `data/simpsons_dataset/waylon_smithers/pic_0051.jpg`      |

Далее исключая повторки разделил весь датасет на тренировочную и валидационную. При этом я брал из каждой папки с конкретным симпсоном рандомно 85% фото в тренировочную, а остальные 15% в валидационную. Таким образом, добился равномерности в процентном соотношении фото для каждого героя между train и validation.

После этого высчитал среднее по тренировочному датасету и стандартное отклонение.

-------------------------------------------------------------------------------------------------------------------------------------------
## Выбор архитектуры нейросети

Итак, с датасетом разобрались, теперь нужно писать саму нейросеть. За основу была взята сверточная нейронная сеть resnet50. Был выбор: использовать натренированную на ImageNet и дотренировывать или тренировать с нуля самому. Я решил попробовать взять веса с ImageNet’а, так как с ними модель уже всё-таки доведена до ума и как-никак фичи она умеет извлекать. Были мысли, что на симпсонах не сработает, так как всё-таки цветовая палитра мультика не особо совпадает с реальным миром. Но в итоге, всё получилось.

В нашем датасете 42 класса. Поэтому я заменил полносвязный слой на такой, чтобы он выдавал вектор длины 42.

-------------------------------------------------------------------------------------------------------------------------------------------
## Тренировка модели и тесты

Для мониторинга процесса тренировки использовал tensorboard.

Loss / train    |
:-------------------------:|
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/loss_train.png)  |  

Loss / validation    |
:-------------------------:|
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/loss_validation.png)  |  

### Результаты на тестовом датасете

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| 0                | 1.00      | 0.92   | 0.96     |
| 1                | 1.00      | 0.79   | 0.88     |
| 2                | 0.88      | 1.00   | 0.93     |
| 3                | 0.88      | 0.82   | 0.85     |
| 4                | 0.98      | 1.00   | 0.99     |
| 5                | 1.00      | 0.46   | 0.63     |
| 6                | 0.94      | 0.96   | 0.95     |
| 7                | 0.98      | 1.00   | 0.99     |
| 8                | 0.91      | 0.91   | 0.91     |
| 9                | 0.87      | 0.96   | 0.91     |
| 10               | 1.00      | 0.64   | 0.78     |
| 11               | 0.91      | 0.96   | 0.93     |
| 12               | 1.00      | 0.92   | 0.96     |
| 13               | 1.00      | 0.60   | 0.75     |
| 14               | 1.00      | 1.00   | 1.00     |
| 15               | 0.91      | 1.00   | 0.95     |
| 16               | 0.98      | 1.00   | 0.99     |
| 17               | 0.93      | 1.00   | 0.96     |
| 18               | 1.00      | 0.98   | 0.99     |
| 19               | 1.00      | 0.82   | 0.90     |
| 20               | 0.98      | 0.92   | 0.95     |
| 21               | 1.00      | 0.73   | 0.84     |
| 22               | 0.98      | 0.94   | 0.96     |
| 23               | 1.00      | 0.86   | 0.92     |
| 24               | 1.00      | 0.94   | 0.97     |
| 25               | 0.89      | 1.00   | 0.94     |
| 26               | 0.77      | 0.91   | 0.83     |
| 27               | 0.98      | 1.00   | 0.99     |
| 28               | 0.91      | 1.00   | 0.95     |
| 29               | 0.91      | 0.96   | 0.93     |
| 30               | 0.91      | 0.91   | 0.91     |
| 31               | 0.91      | 1.00   | 0.95     |
| 32               | 0.96      | 1.00   | 0.98     |
| 33               | 0.86      | 1.00   | 0.92     |
| 34               | 1.00      | 0.80   | 0.89     |
| 35               | 0.85      | 1.00   | 0.92     |
| 36               | 1.00      | 0.90   | 0.95     |
| 37               | 0.92      | 1.00   | 0.96     |
| 38               | 1.00      | 0.50   | 0.67     |
| 39               | 0.89      | 0.80   | 0.84     |
| 40               | 0.80      | 0.36   | 0.50     |
| 41               | 1.00      | 0.90   | 0.95     |
| **macro avg**    | **0.94** | **0.88** | **0.90** |
| **weighted avg** | **0.94** | **0.94** | **0.94** |

Visualisation    |
:-------------------------:|
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/pictures/test.png)  | 