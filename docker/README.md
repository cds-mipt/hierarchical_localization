# Сборка докер-образа
```
docker build -t hloc:latest .
```
# Запуск докер-контейнера
cd <корень репозитория hierarchical-localization>
```
bash docker/start_and_into.sh <путь к папке с датасетами и моделью>
```

# Запуск jupyter notebook
```
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

## При необходимости можно войти в докер контейнер в другом терминале следующим образом:
```
docker exec -it <ID контейнера> /bin/bash
```
