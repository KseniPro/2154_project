How to run a web-service for image comparison?

Откройте проект в VS Code или PyCharm. Если появится предупреждение об установке зависимостей, нажмите "Install".
В терминале каждого проекта ввести соответсвующие команды.

Запуск сервера (backendCompare)
1. python -m venv venv
2. source venv/bin/activate
3. pip3 install django
4. pip3 install -r requirements.txt 
5. python3 manage.py runserver
6*. В терминале появится ссылка на сервер (например, http://127.0.0.1:8000/). Если порт отличается, измените его в проекте pythonStreamlit.

Запуск веб-сервиса (pythonStreamlit)
1. source .venv/bin/activate
2. pip3 install streamlit
3. python3 -m streamlit run main.py