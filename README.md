# Классификатор симпсонов
NSU Artificial intelligence task

Моя задача — создать нейросеть, классифицирующую симпсонов. Для выполнения данного задания у меня есть датасет (https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/code). С него и начнем наши рассуждения.

Для тренировочной + валидационной выборок использовались фото из директории data/simpson_dataset/, а для тестовой из data/kaggle_simpson_testset

Первым делом хотелось бы оценить общую картину, сколько фоток в каждом классе (в тренировочном и тестовом датасетах соответственно).

Тренировочный    |   Тестовый
:-------------------------:|:-------------------------:
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20train%2Bvalidation%20BEFORE%20adding%20new%20photos.png)  |  ![](https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20testset%20BEFORE%20adding%20new%20photos.png)


Такая ситуация мне не понравилась. Я решил добавить картинок в тренировочный датасет в 10, 19, 26, 40 классы, в тестовый в классы где было 0 картинок. Ситуация после добавления: 

Тренировочный   |   Тестовый
:-------------------------:|:-------------------------:
![](https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20train%2Bvalidation%20AFTER%20adding%20new%20photos.png)  |  ![](https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20testset%20AFTER%20adding%20new%20photos.png)

После этого был написан скрипт для поиска повторок в тренировочном + валидационном датасете. Искал с помощью хеш-функции. Оказалось 6 пар повторяющихся:

| Герой                | Путь до картинки 1                                        | Путь до картинки 2                                        |
|----------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| Professor John Frink | `data/simpsons_dataset/professor_john_frink/pic_0014.jpg` | `data/simpsons_dataset/professor_john_frink/pic_0012.jpg` |
| Lenny Leonard        | `data/simpsons_dataset/lenny_leonard/pic_0262.jpg`        | `data/simpsons_dataset/lenny_leonard/pic_0257.jpg`        |
| Cletus Spuckler      | `data/simpsons_dataset/cletus_spuckler/pic_0010.jpg`      | `data/simpsons_dataset/cletus_spuckler/pic_0012.jpg`      |
| Rainier Wolfcastle   | `data/simpsons_dataset/rainier_wolfcastle/pic_0016.jpg`   | `data/simpsons_dataset/rainier_wolfcastle/pic_0011.jpg`   |
| Mayor Quimby         | `data/simpsons_dataset/mayor_quimby/pic_0054.jpg`         | `data/simpsons_dataset/mayor_quimby/pic_0176.jpg`         |
| Waylon Smithers      | `data/simpsons_dataset/waylon_smithers/pic_0038.jpg`      | `data/simpsons_dataset/waylon_smithers/pic_0051.jpg`      |

Далее исключая повторки разделил весь датасет на тренировочную и валидационную. При этом я брал из каждой папки с конкретным симпсоном рандомно 85% фото в тренировочную, а остальные 15% в валидационную. Таким образом, добился равномерности в процентном соотношении фото для каждого героя между train и validation в каждом классе.

После этого высчитал среднее по тренировочному датасету и стандартное отклонение.