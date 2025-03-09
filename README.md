# Классификатор симпсонов
NSU Artificial intelligence task

Моя задача — создать нейросеть, классифицирующую симпсонов. Для выполнения данного задания у меня есть датасет (https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/code). С него и начнем наши рассуждения.

Первым делом хотелось бы оценить общую картину, сколько фоток в каждом классе (в тренировочном и тестовом датасетах соответственно).

Для тренировочной + валидационной выборок использовались фото из директории data/simpson_dataset/, а для тестовой из data/kaggle_simpson_testset


<div style="display: flex; justify-content: center; align-items: flex-start;">
      <div>"Тренировочная + валидационная</div>
      <img align="left" src="https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20train%2Bvalidation%20BEFORE%20adding%20new%20photos.png" width="400" />
      <img align="right" src="https://github.com/Pozovi23/Simpson_classifier/blob/main/distribution%20of%20photos%20in%20testset%20BEFORE%20adding%20new%20photos.png" width="400" />
</div>

